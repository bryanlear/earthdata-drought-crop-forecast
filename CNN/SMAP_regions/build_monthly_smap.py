"""
Monthly SMAP 8-channel cube for Ethiopia (44×43 native window).
Steps: find common months across CHIRPS CSVs and SMAP archive → build composite lat/lon
       → extract Ethiopia H5 window per day → nanmean per month → save .npz.
"""

import h5py
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd


SMAP_DATA_DIR = Path('/Volumes/bryan_SSD/spl3smp')
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CHIRPS_DIR = PROJECT_DIR / 'reference_data_preprocessing'
OUT_DIR = SCRIPT_DIR

# CHIRPS CSVs used to determine target months
CHIRPS_CSVS = [
    'afar_chirps_monthly.csv',
    'amhara_chirps_monthly.csv',
    'oromia_chirps_monthly.csv',
    'snnpr_chirps_monthly.csv',
    'somali_chirps_monthly.csv',
    'west_arsi_chirps_monthly.csv',
]

# ── SMAP HDF5 groups / features ─────────────────────────────────────────────
AM_GROUP = 'Soil_Moisture_Retrieval_Data_AM'
PM_GROUP = 'Soil_Moisture_Retrieval_Data_PM'

FEATURES = [
    (AM_GROUP, 'soil_moisture',            'soil_moisture_am'),
    (PM_GROUP, 'soil_moisture_pm',         'soil_moisture_pm'),
    (AM_GROUP, 'surface_temperature',      'surface_temp_am'),
    (PM_GROUP, 'surface_temperature_pm',   'surface_temp_pm'),
    (AM_GROUP, 'vegetation_water_content', 'vegetation_water'),
    (AM_GROUP, 'vegetation_opacity',       'vegetation_opacity'),
    (AM_GROUP, 'tb_polarization_diff',     'tb_polarization_diff'),
    (AM_GROUP, 'bulk_density',             'bulk_density'),
]
N_FEATURES = len(FEATURES)


ETH_ROW_START, ETH_ROW_END = 149, 193
ETH_COL_START, ETH_COL_END = 569, 612



def parse_date(filename: str) -> datetime:
    return datetime.strptime(filename.split('_')[4], '%Y%m%d')


def _crop(arr, r0=ETH_ROW_START, r1=ETH_ROW_END,
          c0=ETH_COL_START, c1=ETH_COL_END):
    return arr[r0:r1, c0:c1].astype(np.float32)


def extract_ethiopia(file_path: Path) -> np.ndarray:
    H = ETH_ROW_END - ETH_ROW_START
    W = ETH_COL_END - ETH_COL_START
    result = np.full((N_FEATURES, H, W), np.nan, dtype=np.float32)

    with h5py.File(file_path, 'r') as f:
        am_qual = f[f'{AM_GROUP}/retrieval_qual_flag'][:]
        pm_qual = f[f'{PM_GROUP}/retrieval_qual_flag_pm'][:]

        for ch, (group, dataset, _) in enumerate(FEATURES):
            if dataset == 'tb_polarization_diff':
                tb_v = f[f'{group}/tb_v_corrected'][:]
                tb_h = f[f'{group}/tb_h_corrected'][:]
                mask = (tb_v != -9999.0) & (tb_h != -9999.0)
                data = np.where(mask, tb_v - tb_h, np.nan)
                result[ch] = _crop(data)
                continue

            data = f[f'{group}/{dataset}'][:]
            mask = data != -9999.0

            if 'soil_moisture' in dataset:
                qual = am_qual if group == AM_GROUP else pm_qual
                mask = mask & ((qual & 1) == 0)

            result[ch] = _crop(np.where(mask, data, np.nan))

    return result


def get_lat_lon_grids() -> tuple[np.ndarray, np.ndarray]:
    h5_files = sorted(SMAP_DATA_DIR.glob('SMAP_L3_SM_P_*.h5'),
                      key=lambda p: parse_date(p.name))
    H = ETH_ROW_END - ETH_ROW_START
    W = ETH_COL_END - ETH_COL_START
    lat_grid = np.full((H, W), np.nan, dtype=np.float32)
    lon_grid = np.full((H, W), np.nan, dtype=np.float32)

    # Sample 30 files spread across the archive
    indices = np.linspace(0, len(h5_files) - 1, 30, dtype=int)
    for idx in indices:
        with h5py.File(h5_files[idx], 'r') as f:
            lat = _crop(f[f'{AM_GROUP}/latitude'][:])
            lon = _crop(f[f'{AM_GROUP}/longitude'][:])
        valid = (lat != -9999.0) & (lon != -9999.0)
        lat_grid = np.where(valid & np.isnan(lat_grid), lat, lat_grid)
        lon_grid = np.where(valid & np.isnan(lon_grid), lon, lon_grid)

    filled = np.count_nonzero(~np.isnan(lat_grid))
    print(f'Lat/lon grid: {filled}/{H*W} cells filled ({100*filled/(H*W):.1f}%)')
    return lat_grid, lon_grid


def target_months() -> list[pd.Timestamp]:
    smap_start = pd.Timestamp('2015-04-01')
    month_sets = []
    for csv_name in CHIRPS_CSVS:
        df = pd.read_csv(CHIRPS_DIR / csv_name, parse_dates=['date'])
        months = set(df['date'])
        month_sets.append(months)

    common = sorted(set.intersection(*month_sets))
    overlap = [m for m in common if m >= smap_start]
    print(f'CHIRPS months: {len(common)}, overlap with SMAP: {len(overlap)}')
    print(f'  {overlap[0].date()} → {overlap[-1].date()}')
    return overlap


def build_monthly_cube(months: list[pd.Timestamp]) -> np.ndarray:
    h5_files = sorted(SMAP_DATA_DIR.glob('SMAP_L3_SM_P_*.h5'),
                      key=lambda p: parse_date(p.name))
    by_month: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for fp in h5_files:
        d = parse_date(fp.name)
        by_month[(d.year, d.month)].append(fp)

    H = ETH_ROW_END - ETH_ROW_START
    W = ETH_COL_END - ETH_COL_START
    T = len(months)
    cube = np.full((T, N_FEATURES, H, W), np.nan, dtype=np.float32)

    for t, month in enumerate(months):
        key = (month.year, month.month)
        files = by_month.get(key, [])
        if not files:
            print(f'  WARNING: no SMAP files for {month.date()}, leaving NaN')
            continue

        # Stack daily extractions → nanmean
        daily = np.stack([extract_ethiopia(fp) for fp in files], axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            cube[t] = np.nanmean(daily, axis=0)

        if (t + 1) % 12 == 0 or t == T - 1:
            valid_frac = np.count_nonzero(~np.isnan(cube[t])) / cube[t].size
            print(f'  {t+1:3d}/{T}  {month.date()}  '
                  f'{len(files):2d} files  {100*valid_frac:.0f}% valid')

    return cube



if __name__ == '__main__':
    print('=== Building monthly SMAP dataset for Ethiopia ===\n')

    months = target_months()
    lat_grid, lon_grid = get_lat_lon_grids()

    print(f'\nCompositing {len(months)} months from daily SMAP ...')
    cube = build_monthly_cube(months)

    # Summary
    print(f'\nCube shape: {cube.shape}  (T, C, H, W)')
    total = cube.size
    valid = np.count_nonzero(~np.isnan(cube))
    print(f'Overall coverage: {valid}/{total} ({100*valid/total:.1f}%)')

    for ch, (_, _, name) in enumerate(FEATURES):
        ch_valid = np.count_nonzero(~np.isnan(cube[:, ch]))
        ch_total = cube[:, ch].size
        print(f'  {name:25s}: {100*ch_valid/ch_total:.1f}% valid')

    # Save
    out_path = OUT_DIR / 'ethiopia_smap_monthly.npz'
    date_strings = np.array([m.strftime('%Y-%m-%d') for m in months])
    feature_names = np.array([f[2] for f in FEATURES])
    np.savez_compressed(
        out_path,
        cube=cube,
        dates=date_strings,
        feature_names=feature_names,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f'\nSaved {out_path.name} ({size_mb:.1f} MB)')

    # Quick verification
    loaded = np.load(out_path)
    print(f'Verified: cube={loaded["cube"].shape}, dates={loaded["dates"].shape}, '
          f'features={list(loaded["feature_names"])}')
