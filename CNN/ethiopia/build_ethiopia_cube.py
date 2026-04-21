"""
Monthly 64×64 SMAP cube + CHIRPS resampling + region masks for Ethiopia.
Steps: load target months from CSVs → build composite lat/lon → extract 64×64 SMAP window
       → nanmean per month → resample CHIRPS TIF (7×7 block avg) → polygon region masks → save .npz.
"""

import warnings
import json
import h5py
import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from matplotlib.path import Path as MplPath
from scipy.ndimage import uniform_filter


SMAP_DATA_DIR = Path('/Volumes/bryan_SSD/spl3smp')
SCRIPT_DIR    = Path(__file__).resolve().parent
CNN_DIR       = SCRIPT_DIR.parent
ROOT_DIR      = CNN_DIR.parent
CHIRPS_TIF_DIR = ROOT_DIR / 'reference_data'
GADM_ETH1      = CNN_DIR / 'gadm' / 'ethiopia' / 'gadm41_ETH_1.json'
WEST_ARSI_BOUNDARY = ROOT_DIR / 'reference_data' / 'reference_data_preprocessing' / 'west_arsi_boundary.geojson'
OUT_DIR       = SCRIPT_DIR


AM_GROUP  = 'Soil_Moisture_Retrieval_Data_AM'
PM_GROUP  = 'Soil_Moisture_Retrieval_Data_PM'
SMAP_FILL = -9999.0

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
WIN = 64

ETH_R0, ETH_R1 = 139, 203
ETH_C0, ETH_C1 = 558, 622

CHIRPS_NODATA_THRESHOLD = -100.0
CHIRPS_BLOCK = 7


REGIONS = {
    'afar':      ['Afar'],
    'amhara':    ['Amhara'],
    'oromia':    ['Oromia'],
    'somali':    ['Somali'],
    'west_arsi': None,
}


def parse_smap_date(filename: str) -> datetime:
    return datetime.strptime(Path(filename).name.split('_')[4], '%Y%m%d')


def build_composite_lat_lon() -> tuple[np.ndarray, np.ndarray]:
    all_files = sorted(SMAP_DATA_DIR.glob('SMAP_L3_SM_P_*.h5'),
                       key=lambda p: parse_smap_date(p.name))
    indices = np.linspace(0, len(all_files) - 1, 30, dtype=int)
    sample_files = [all_files[i] for i in indices]

    lat_grid = np.full((WIN, WIN), np.nan, dtype=np.float32)
    lon_grid = np.full((WIN, WIN), np.nan, dtype=np.float32)

    for fp in sample_files:
        with h5py.File(fp, 'r') as f:
            for grp, lk, lok in [
                (AM_GROUP, 'latitude',    'longitude'),
                (PM_GROUP, 'latitude_pm', 'longitude_pm'),
            ]:
                try:
                    lat_full = f[f'{grp}/{lk}'][ETH_R0:ETH_R1, ETH_C0:ETH_C1].astype(np.float32)
                    lon_full = f[f'{grp}/{lok}'][ETH_R0:ETH_R1, ETH_C0:ETH_C1].astype(np.float32)
                except KeyError:
                    continue
                valid = (lat_full != SMAP_FILL) & (lon_full != SMAP_FILL)
                lat_grid = np.where(valid & np.isnan(lat_grid), lat_full, lat_grid)
                lon_grid = np.where(valid & np.isnan(lon_grid), lon_full, lon_grid)

    filled = int(np.count_nonzero(~np.isnan(lat_grid)))
    print(f'Composite lat/lon: {filled}/{WIN*WIN} cells filled ({100*filled/(WIN*WIN):.1f}%)')
    return lat_grid, lon_grid


def extract_smap_window(file_path: Path) -> np.ndarray:
    result = np.full((N_FEATURES, WIN, WIN), np.nan, dtype=np.float32)

    with h5py.File(file_path, 'r') as f:
        am_qual = f[f'{AM_GROUP}/retrieval_qual_flag'][ETH_R0:ETH_R1, ETH_C0:ETH_C1]
        pm_qual = f[f'{PM_GROUP}/retrieval_qual_flag_pm'][ETH_R0:ETH_R1, ETH_C0:ETH_C1]

        for ch, (group, dataset, _) in enumerate(FEATURES):
            if dataset == 'tb_polarization_diff':
                tb_v = f[f'{group}/tb_v_corrected'][ETH_R0:ETH_R1, ETH_C0:ETH_C1]
                tb_h = f[f'{group}/tb_h_corrected'][ETH_R0:ETH_R1, ETH_C0:ETH_C1]
                mask = (tb_v != SMAP_FILL) & (tb_h != SMAP_FILL)
                result[ch] = np.where(mask, (tb_v - tb_h).astype(np.float32), np.nan)
                continue

            data = f[f'{group}/{dataset}'][ETH_R0:ETH_R1, ETH_C0:ETH_C1]
            mask = data != SMAP_FILL

            if 'soil_moisture' in dataset:
                qual = am_qual if group == AM_GROUP else pm_qual
                mask = mask & ((qual & 1) == 0)

            result[ch] = np.where(mask, data.astype(np.float32), np.nan)

    return result


def build_smap_cube(months: list) -> np.ndarray:
    h5_files = sorted(SMAP_DATA_DIR.glob('SMAP_L3_SM_P_*.h5'),
                      key=lambda p: parse_smap_date(p.name))
    by_month: dict = defaultdict(list)
    for fp in h5_files:
        d = parse_smap_date(fp.name)
        by_month[(d.year, d.month)].append(fp)

    T = len(months)
    cube = np.full((T, N_FEATURES, WIN, WIN), np.nan, dtype=np.float32)

    for t, month in enumerate(months):
        key = (month.year, month.month)
        files = by_month.get(key, [])
        if not files:
            print(f'  WARNING: no SMAP files for {month.date()}, leaving NaN')
            continue

        daily = np.stack([extract_smap_window(fp) for fp in files], axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            cube[t] = np.nanmean(daily, axis=0)

        if (t + 1) % 12 == 0 or t == T - 1:
            valid_frac = np.count_nonzero(~np.isnan(cube[t])) / cube[t].size
            print(f'  {t+1:3d}/{T}  {month.date()}  {len(files):2d} files  '
                  f'{100*valid_frac:.0f}% valid')

    return cube


def resample_chirps_to_smap(tif_path: Path,
                             lat_grid: np.ndarray,
                             lon_grid: np.ndarray) -> np.ndarray:
    result = np.full((WIN, WIN), np.nan, dtype=np.float32)
    valid_cells = ~np.isnan(lat_grid) & ~np.isnan(lon_grid)
    if not valid_cells.any():
        return result

    margin = (CHIRPS_BLOCK // 2 + 1) * 0.05
    lat_min = float(np.nanmin(lat_grid)) - margin
    lat_max = float(np.nanmax(lat_grid)) + margin
    lon_min = float(np.nanmin(lon_grid)) - margin
    lon_max = float(np.nanmax(lon_grid)) + margin

    with rasterio.open(tif_path) as src:
        win = rasterio.windows.from_bounds(
            lon_min, lat_min, lon_max, lat_max, src.transform
        )
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        chirps_arr = src.read(1, window=win).astype(np.float32)
        win_transform = src.window_transform(win)

    ch, cw = chirps_arr.shape
    if ch == 0 or cw == 0:
        return result

    chirps_arr = np.where(chirps_arr <= CHIRPS_NODATA_THRESHOLD, np.nan, chirps_arr)

    not_nan = (~np.isnan(chirps_arr)).astype(np.float32)
    filled  = np.where(np.isnan(chirps_arr), 0.0, chirps_arr)
    n2 = CHIRPS_BLOCK * CHIRPS_BLOCK
    sum_v = uniform_filter(filled,  size=CHIRPS_BLOCK, mode='constant', cval=0.0) * n2
    sum_n = uniform_filter(not_nan, size=CHIRPS_BLOCK, mode='constant', cval=0.0) * n2
    block_mean = np.full_like(sum_v, np.nan)
    valid_n = sum_n > 0
    block_mean[valid_n] = sum_v[valid_n] / sum_n[valid_n]

    CHIRPS_RES = abs(win_transform.a)
    win_left   = win_transform.c
    win_top    = win_transform.f

    c_rows = np.floor((win_top  - lat_grid) / CHIRPS_RES).astype(int)
    c_cols = np.floor((lon_grid - win_left) / CHIRPS_RES).astype(int)

    in_bounds = (valid_cells &
                 (c_rows >= 0) & (c_rows < ch) &
                 (c_cols >= 0) & (c_cols < cw))
    result[in_bounds] = block_mean[c_rows[in_bounds], c_cols[in_bounds]]
    return result


def build_chirps_cube(months: list,
                      lat_grid: np.ndarray,
                      lon_grid: np.ndarray) -> np.ndarray:
    T = len(months)
    cube = np.full((T, WIN, WIN), np.nan, dtype=np.float32)
    for t, month in enumerate(months):
        tif_path = CHIRPS_TIF_DIR / f'chirps-v3.0.{month.year}.{month.month:02d}.tif'
        if not tif_path.exists():
            print(f'  WARNING: CHIRPS TIF missing for {month.date()}')
            continue
        cube[t] = resample_chirps_to_smap(tif_path, lat_grid, lon_grid)
        if (t + 1) % 12 == 0 or t == T - 1:
            valid_frac = np.count_nonzero(~np.isnan(cube[t])) / cube[t].size
            print(f'  CHIRPS {t+1:3d}/{T}  {month.date()}  {100*valid_frac:.0f}% valid')
    return cube


# Region masks

def _polygons_from_geojson(geojson: dict) -> list[np.ndarray]:
    polys = []
    for feat in geojson.get('features', [geojson]):
        geom = feat.get('geometry', feat)
        if geom is None:
            continue
        if geom['type'] == 'Polygon':
            rings = geom['coordinates']
        elif geom['type'] == 'MultiPolygon':
            rings = [r for poly in geom['coordinates'] for r in poly]
        else:
            continue
        for ring in rings:
            polys.append(np.array(ring))
    return polys


def build_region_mask(lat_grid: np.ndarray,
                      lon_grid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    with open(GADM_ETH1) as fh:
        gadm_geojson = json.load(fh)

    with open(WEST_ARSI_BOUNDARY) as fh:
        west_arsi_geojson = json.load(fh)

    valid = ~np.isnan(lat_grid) & ~np.isnan(lon_grid)
    pts = np.column_stack([lon_grid[valid], lat_grid[valid]])

    region_names = sorted(REGIONS.keys())
    masks = []

    for slug in region_names:
        name1_values = REGIONS[slug]

        if name1_values is None:
            polys = _polygons_from_geojson(west_arsi_geojson)
        else:
            polys = []
            for feat in gadm_geojson['features']:
                if feat['properties']['NAME_1'] in name1_values:
                    polys.extend(_polygons_from_geojson({'features': [feat]}))

        if not polys:
            print(f'  WARNING: no polygons found for region {slug}')
            masks.append(np.zeros((WIN, WIN), dtype=bool))
            continue

        inside = np.zeros(pts.shape[0], dtype=bool)
        for ring in polys:
            if len(ring) < 4:
                continue
            path = MplPath(ring)
            inside |= path.contains_points(pts)

        region_mask = np.zeros((WIN, WIN), dtype=bool)
        region_mask[valid] = inside

        n_cells = int(region_mask.sum())
        print(f'  {slug:20s}: {n_cells} cells')
        masks.append(region_mask)

    return np.array(masks, dtype=bool), region_names


def target_months() -> list:
    smap_start = pd.Timestamp('2015-04-01')
    csv_dir = SCRIPT_DIR

    csv_files = sorted(csv_dir.glob('*_chirps_monthly.csv'))
    if not csv_files:
        raise FileNotFoundError(
            f'No CHIRPS CSVs found in {csv_dir}.\n'
            'Run build_ethiopia_chirps.py first.'
        )

    month_sets = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, parse_dates=['date'])
        month_sets.append(set(df['date']))
        print(f'  {csv_path.name}: {len(df)} rows')

    common = set.intersection(*month_sets)
    overlap = sorted(m for m in common if m >= smap_start)
    print(f'\nCommon months: {len(common)}, overlap with SMAP: {len(overlap)}')
    print(f'  {overlap[0].date()} → {overlap[-1].date()}')
    return overlap


if __name__ == '__main__':
    print('=== Building Ethiopia 64×64 SMAP monthly cube ===\n')

    print('--- Target months ---')
    months = target_months()

    print('\n--- Composite lat/lon ---')
    lat_grid, lon_grid = build_composite_lat_lon()

    print(f'\n--- Region masks ---')
    region_mask, region_names = build_region_mask(lat_grid, lon_grid)
    print(f'Regions: {region_names}')

    print(f'\n--- Building SMAP cube ({len(months)} months) ---')
    smap_cube = build_smap_cube(months)

    print(f'\n--- Resampling CHIRPS ---')
    chirps_cube = build_chirps_cube(months, lat_grid, lon_grid)

    print(f'\nSMAP cube shape:   {smap_cube.shape}')
    total = smap_cube.size
    valid = np.count_nonzero(~np.isnan(smap_cube))
    print(f'SMAP coverage:     {valid}/{total} ({100*valid/total:.1f}%)')
    for ch, (_, _, name) in enumerate(FEATURES):
        ch_valid = np.count_nonzero(~np.isnan(smap_cube[:, ch]))
        ch_total = smap_cube[:, ch].size
        print(f'  {name:25s}: {100*ch_valid/ch_total:.1f}% valid')

    print(f'\nCHIRPS cube shape: {chirps_cube.shape}')
    c_valid = np.count_nonzero(~np.isnan(chirps_cube))
    print(f'CHIRPS coverage:   {c_valid}/{chirps_cube.size} ({100*c_valid/chirps_cube.size:.1f}%)')


    out_path = OUT_DIR / 'ethiopia_smap_monthly.npz'
    date_strings  = np.array([m.strftime('%Y-%m-%d') for m in months])
    feature_names = np.array([f[2] for f in FEATURES])

    np.savez_compressed(
        out_path,
        smap_cube    = smap_cube,
        chirps_cube  = chirps_cube,
        region_mask  = region_mask,
        region_names = np.array(region_names),
        feature_names= feature_names,
        dates        = date_strings,
        lat_grid     = lat_grid,
        lon_grid     = lon_grid,
    )

    size_mb = out_path.stat().st_size / 1e6
    print(f'\nSaved {out_path.name} ({size_mb:.1f} MB)')

    loaded = np.load(out_path, allow_pickle=True)
    print(f'Verified:')
    print(f'  smap_cube    = {loaded["smap_cube"].shape}')
    print(f'  chirps_cube  = {loaded["chirps_cube"].shape}')
    print(f'  region_mask  = {loaded["region_mask"].shape}')
    print(f'  region_names = {list(loaded["region_names"])}')
    print(f'  dates        = {loaded["dates"][0]} → {loaded["dates"][-1]}')
