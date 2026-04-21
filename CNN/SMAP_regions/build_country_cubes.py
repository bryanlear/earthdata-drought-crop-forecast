"""
Monthly 64×64 SMAP cubes + CHIRPS resampling + region masks for 5 Horn of Africa countries.
Steps: build composite lat/lon (30 sampled files, AM+PM) → per country: crop 64×64 window
       → nanmean per month → resample CHIRPS TIF (7×7 block avg) → polygon region masks → save .npz.
"""

import h5py
import json
import warnings
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
from matplotlib.path import Path as MplPath
from scipy.ndimage import uniform_filter


SMAP_DATA_DIR  = Path('/Volumes/bryan_SSD/spl3smp')
SCRIPT_DIR     = Path(__file__).resolve().parent
CNN_DIR        = SCRIPT_DIR.parent
ROOT_DIR       = CNN_DIR.parent
CHIRPS_TIF_DIR = ROOT_DIR / 'reference_data'
CHIRPS_CSV_DIR = CNN_DIR / 'CHIRPS_processing'
GADM_DIR       = CNN_DIR / 'gadm'
OUT_DIR        = SCRIPT_DIR

# ── SMAP constants ─────────────────────────────────────────────────────────────
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

WINDOWS = {
    'eritrea':     (118, 182, 557, 621),
    'kenya':       (171, 235, 552, 616),
    'somalia':     (153, 217, 574, 638),
    'south_sudan': (144, 208, 531, 595),
    'sudan':       (116, 180, 531, 595),
}

CHIRPS_NODATA_THRESHOLD = -100.0
CHIRPS_BLOCK = 7

COUNTRIES = {
    'somalia': {
        'gadm': 'somalia/gadm41_SOM_1.json',
        'regions': {
            'northwest_agropastoral':        ['Awdal', 'WoqooyiGalbeed', 'Togdheer'],
            'northeast_dry_pastoral':        ['Sanaag', 'Sool', 'Bari', 'Nugaal'],
            'central_pastoral':              ['Mudug', 'Galguduud', 'Hiiraan'],
            'juba_riverine':                 ['JubbadaDhexe', 'JubbadaHoose', 'Gedo'],
            'shabelle_riverine':             ['ShabeellahaDhexe', 'ShabeellahaHoose', 'Banaadir'],
            'southern_rainfed_agropastoral': ['Bay', 'Bakool'],
        },
    },
    'eritrea': {
        'gadm': 'eritrea/gadm41_ERI_1.json',
        'regions': {
            'western_lowlands':      ['GashBarka'],
            'eastern_lowlands':      ['SemenawiKeyihBahri', 'DebubawiKeyihBahri'],
            'southwestern_lowlands': ['Anseba'],
        },
    },
    'kenya': {
        'gadm': 'kenya/gadm41_KEN_1.json',
        'regions': {
            'northern_arid_pastoral': ['Turkana', 'Marsabit', 'Mandera', 'Wajir', 'Samburu', 'Isiolo'],
            'eastern_semiarid':       ['Garissa', 'TanaRiver', 'Kitui', 'Makueni', 'Machakos',
                                       'Embu', 'Tharaka-Nithi', 'Meru'],
            'central_highlands':      ['Nyeri', 'Kirinyaga', "Murang'a", 'Kiambu',
                                       'Nyandarua', 'Nairobi', 'Laikipia'],
            'rift_valley_highlands':  ['Nakuru', 'Narok', 'Kajiado', 'Baringo',
                                       'Elgeyo-Marakwet', 'UasinGishu', 'Nandi',
                                       'Kericho', 'Bomet', 'TransNzoia', 'WestPokot'],
            'western_high_rainfall':  ['Kakamega', 'Bungoma', 'Busia', 'Vihiga',
                                       'Siaya', 'Kisumu', 'HomaBay', 'Migori', 'Kisii', 'Nyamira'],
            'coastal_lowlands':       ['Mombasa', 'Kilifi', 'Kwale', 'Lamu', 'TaitaTaveta'],
        },
    },
    'sudan': {
        'gadm': 'sudan/gadm41_SDN_1.json',
        'regions': {
            'desert':                ['Northern', 'RiverNile'],
            'semi_desert':           ['RedSea', 'Kassala', 'Khartoum'],
            'low_rainfall_savanna':  ['NorthKurdufan', 'NorthDarfur', 'AlQadarif'],
            'high_rainfall_savanna': ['SouthKurdufan', 'SouthDarfur', 'CentralDarfur',
                                      'EastDarfur', 'WestDarfur', 'WestKurdufan'],
            'gezira_irrigated_nile': ['AlJazirah', 'WhiteNile', 'Sennar'],
            'blue_nile_rainfed':     ['BlueNile'],
        },
    },
    'south_sudan': {
        'gadm': 'south_sudan/gadm41_SSD_1.json',
        'regions': {
            'greenbelt':                 ['WestEquatoria', 'CentralEquatoria'],
            'ironstone_plateau':         ['WestBahr-al-Ghazal'],
            'flood_plains':              ['Jungoli', 'Unity', 'Lakes'],
            'nile_sobat_river':          ['UpperNile'],
            'eastern_pastoral_drylands': ['EasternEquatoria'],
            'hills_and_mountains':       ['NorthBahr-al-Ghazal', 'Warap'],
        },
    },
}


def parse_smap_date(filename: str) -> datetime:
    return datetime.strptime(Path(filename).name.split('_')[4], '%Y%m%d')


def build_composite_lat_lon() -> tuple[np.ndarray, np.ndarray]:
    all_files = sorted(SMAP_DATA_DIR.glob('SMAP_L3_SM_P_*.h5'),
                       key=lambda p: parse_smap_date(p.name))
    indices = np.linspace(0, len(all_files) - 1, 30, dtype=int)
    sample_files = [all_files[i] for i in indices]

    H, W = 406, 964
    lat_comp = np.full((H, W), np.nan, dtype=np.float32)
    lon_comp = np.full((H, W), np.nan, dtype=np.float32)

    for fp in sample_files:
        with h5py.File(fp, 'r') as f:
            for grp, lk, lok in [(AM_GROUP, 'latitude',    'longitude'),
                                  (PM_GROUP, 'latitude_pm', 'longitude_pm')]:
                lat = f[f'{grp}/{lk}'][:]
                lon = f[f'{grp}/{lok}'][:]
                valid = (lat != SMAP_FILL) & (lon != SMAP_FILL)
                lat_comp = np.where(valid & np.isnan(lat_comp),
                                    lat.astype(np.float32), lat_comp)
                lon_comp = np.where(valid & np.isnan(lon_comp),
                                    lon.astype(np.float32), lon_comp)

    filled = int(np.count_nonzero(~np.isnan(lat_comp)))
    print(f'Composite lat/lon: {filled}/{H*W} cells filled ({100*filled/(H*W):.1f}%)')
    return lat_comp, lon_comp


# ─────────────────────────────────────────────────────────────────────────────
# SMAP extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_smap_window(file_path: Path,
                        r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    result = np.full((N_FEATURES, WIN, WIN), np.nan, dtype=np.float32)

    with h5py.File(file_path, 'r') as f:
        am_qual = f[f'{AM_GROUP}/retrieval_qual_flag'][:]
        pm_qual = f[f'{PM_GROUP}/retrieval_qual_flag_pm'][:]

        for ch, (group, dataset, _) in enumerate(FEATURES):

            if dataset == 'tb_polarization_diff':
                tb_v = f[f'{group}/tb_v_corrected'][r0:r1, c0:c1].astype(np.float32)
                tb_h = f[f'{group}/tb_h_corrected'][r0:r1, c0:c1].astype(np.float32)
                valid = (tb_v != SMAP_FILL) & (tb_h != SMAP_FILL)
                result[ch] = np.where(valid, tb_v - tb_h, np.nan)
                continue

            data = f[f'{group}/{dataset}'][r0:r1, c0:c1].astype(np.float32)
            mask = data != SMAP_FILL

            if 'soil_moisture' in dataset:
                qual = am_qual if group == AM_GROUP else pm_qual
                qual_crop = qual[r0:r1, c0:c1]
                mask = mask & ((qual_crop & 1) == 0)

            result[ch] = np.where(mask, data, np.nan)

    return result


def build_smap_cube(months: list[pd.Timestamp],
                    r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    h5_files = sorted(SMAP_DATA_DIR.glob('SMAP_L3_SM_P_*.h5'),
                      key=lambda p: parse_smap_date(p.name))
    by_month: dict[tuple[int, int], list[Path]] = defaultdict(list)
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

        daily = np.stack([extract_smap_window(fp, r0, r1, c0, c1)
                          for fp in files], axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            cube[t] = np.nanmean(daily, axis=0)

        if (t + 1) % 24 == 0 or t == T - 1:
            vf = np.count_nonzero(~np.isnan(cube[t])) / cube[t].size
            print(f'  SMAP {t+1:3d}/{T}  {month.date()}  '
                  f'{len(files):2d} files  {100*vf:.0f}% valid')

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
        bounds = src.bounds
        lat_min = max(lat_min, bounds.bottom)
        lat_max = min(lat_max, bounds.top)
        lon_min = max(lon_min, bounds.left)
        lon_max = min(lon_max, bounds.right)
        if lat_min >= lat_max or lon_min >= lon_max:
            return result

        win = rasterio.windows.from_bounds(
            lon_min, lat_min, lon_max, lat_max, src.transform
        ).round_offsets().round_lengths()
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


def build_chirps_cube(months: list[pd.Timestamp],
                      lat_grid: np.ndarray,
                      lon_grid: np.ndarray) -> np.ndarray:
    T = len(months)
    cube = np.full((T, WIN, WIN), np.nan, dtype=np.float32)
    for t, month in enumerate(months):
        tif = CHIRPS_TIF_DIR / f'chirps-v3.0.{month.year}.{month.month:02d}.tif'
        if not tif.exists():
            print(f'  WARNING: CHIRPS TIF missing → {tif.name}')
            continue
        cube[t] = resample_chirps_to_smap(tif, lat_grid, lon_grid)

        if (t + 1) % 24 == 0 or t == T - 1:
            vf = np.count_nonzero(~np.isnan(cube[t])) / cube[t].size
            print(f'  CHIRPS {t+1:3d}/{T}  {month.date()}  {100*vf:.0f}% valid')

    return cube


def build_region_mask(country: str,
                      lat_grid: np.ndarray,
                      lon_grid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    cfg = COUNTRIES[country]
    gadm_path = GADM_DIR / cfg['gadm']
    with open(gadm_path) as fh:
        gadm = json.load(fh)

    valid = ~np.isnan(lat_grid) & ~np.isnan(lon_grid)
    pts = np.column_stack([lon_grid[valid], lat_grid[valid]])

    region_names = sorted(cfg['regions'].keys())
    masks = []

    for slug in region_names:
        name1_vals = set(cfg['regions'][slug])

        mpl_paths = []
        for feat in gadm['features']:
            if feat['properties']['NAME_1'] not in name1_vals:
                continue
            geom = feat['geometry']
            if geom['type'] == 'Polygon':
                rings = geom['coordinates']
            elif geom['type'] == 'MultiPolygon':
                rings = [ring for poly in geom['coordinates'] for ring in poly]
            else:
                continue
            for ring in rings:
                if len(ring) >= 3:
                    mpl_paths.append(MplPath(ring))

        inside = np.zeros(pts.shape[0], dtype=bool)
        for path in mpl_paths:
            not_yet = np.where(~inside)[0]
            if not_yet.size:
                inside[not_yet] |= path.contains_points(pts[not_yet])

        region_mask_2d = np.zeros((WIN, WIN), dtype=bool)
        region_mask_2d[valid] = inside
        masks.append(region_mask_2d)
        print(f'    {slug}: {int(inside.sum())} SMAP cells')

    return np.stack(masks, axis=0), region_names


def target_months(country: str) -> list[pd.Timestamp]:
    smap_start = pd.Timestamp('2015-04-01')
    csv_dir = CHIRPS_CSV_DIR / country
    csvs = sorted(csv_dir.glob('*_chirps_monthly.csv'))
    if not csvs:
        raise FileNotFoundError(f'No CHIRPS CSVs found in {csv_dir}')

    month_sets = [set(pd.read_csv(c, parse_dates=['date'])['date']) for c in csvs]
    common  = sorted(set.intersection(*month_sets))
    overlap = [m for m in common if m >= smap_start]
    print(f'  Months: {len(common)} in all CSVs, {len(overlap)} overlap with SMAP '
          f'({overlap[0].date()} → {overlap[-1].date()})')
    return overlap


if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Building composite lat/lon grid (30 sampled files, AM + PM)...')
    lat_comp, lon_comp = build_composite_lat_lon()

    for country, (r0, r1, c0, c1) in WINDOWS.items():
        print(f'\n{"="*64}')
        print(f' {country.upper()}  —  SMAP window rows {r0}:{r1}, cols {c0}:{c1}')
        print('='*64)

        lat_grid = lat_comp[r0:r1, c0:c1].copy()
        lon_grid = lon_comp[r0:r1, c0:c1].copy()
        print(f'  Lat [{np.nanmin(lat_grid):.2f}, {np.nanmax(lat_grid):.2f}]  '
              f'Lon [{np.nanmin(lon_grid):.2f}, {np.nanmax(lon_grid):.2f}]')

        months = target_months(country)
        T = len(months)

        print(f'\n  Building SMAP cube ({T} months × 8 channels × {WIN}×{WIN}) ...')
        smap_cube = build_smap_cube(months, r0, r1, c0, c1)

        print(f'\n  Building CHIRPS cube ({T} months) ...')
        chirps_cube = build_chirps_cube(months, lat_grid, lon_grid)

        print(f'\n  Building region masks ...')
        region_mask, region_names = build_region_mask(country, lat_grid, lon_grid)

        smap_vf   = np.count_nonzero(~np.isnan(smap_cube))   / smap_cube.size
        chirps_vf = np.count_nonzero(~np.isnan(chirps_cube)) / chirps_cube.size
        print(f'\n  SMAP coverage:   {100*smap_vf:.1f}%')
        print(f'  CHIRPS coverage: {100*chirps_vf:.1f}%')
        print(f'  Regions ({len(region_names)}): {region_names}')

        out_path = OUT_DIR / f'{country}_smap_monthly.npz'
        np.savez_compressed(
            out_path,
            smap_cube    = smap_cube,
            chirps_cube  = chirps_cube,
            region_mask  = region_mask,
            region_names = np.array(region_names),
            feature_names= np.array([f[2] for f in FEATURES]),
            dates        = np.array([m.strftime('%Y-%m-%d') for m in months]),
            lat_grid     = lat_grid,
            lon_grid     = lon_grid,
        )
        size_mb = out_path.stat().st_size / 1e6
        print(f'\n  Saved → {out_path.name}  ({size_mb:.1f} MB)')

        loaded = np.load(out_path, allow_pickle=True)
        print(f'  Verified: smap={loaded["smap_cube"].shape}  '
              f'chirps={loaded["chirps_cube"].shape}  '
              f'mask={loaded["region_mask"].shape}  '
              f'T={loaded["dates"].shape[0]} months')

    print('\nDone.')
