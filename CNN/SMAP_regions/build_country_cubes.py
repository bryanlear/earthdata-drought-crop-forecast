"""
### Monthly 64×64 multi-feature country cubes for Horn of Africa CNN training

For each of 5 countries a single .npz is saved to CNN/SMAP_regions/:

    smap_cube    (T, 8, 64, 64)  float32  — 8 SMAP channels, monthly nanmean
    chirps_cube  (T, 64, 64)     float32  — precip mm resampled to SMAP grid
    region_mask  (R, 64, 64)     bool     — pixel membership per agro-zone
    region_names (R,)            str      — slug matching CHIRPS CSV names
    feature_names(8,)            str      — SMAP channel names
    dates        (T,)            str      — ISO date strings (YYYY-MM-DD)
    lat_grid     (64, 64)        float32  — SMAP cell-centre latitudes
    lon_grid     (64, 64)        float32  — SMAP cell-centre longitudes

Countries / regions (27 total, Djibouti excluded):
    somalia (6), eritrea (3), kenya (6), sudan (6), south_sudan (6)

SMAP source:  /Volumes/bryan_SSD/spl3smp/SMAP_L3_SM_P_*.h5
CHIRPS source: <project>/reference_data/chirps-v3.0.YYYY.MM.tif  (Africa 0.05°)
GADM source:   CNN/gadm/{country}/gadm41_{ISO}_1.json
CHIRPS CSVs:   CNN/CHIRPS_processing/{country}/*_chirps_monthly.csv

IMPORTANT: Run with the conda Python that has rasterio + scipy installed:
    ~/anaconda3/bin/python3 build_country_cubes.py
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

# ── paths ─────────────────────────────────────────────────────────────────────
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
WIN = 64  # fixed window size in SMAP grid pixels

# ── Pre-computed 64×64 SMAP grid windows ──────────────────────────────────────
# (row_start, row_end, col_start, col_end) on the 406×964 EASE-Grid 2.0.
# Derived from composite lat/lon (30 sampled files, AM+PM) — symmetric expansion
# of the natural per-country bbox to TARGET=64.
WINDOWS = {
    'eritrea':     (118, 182, 557, 621),
    'kenya':       (171, 235, 552, 616),
    'somalia':     (153, 217, 574, 638),
    'south_sudan': (144, 208, 531, 595),
    'sudan':       (116, 180, 531, 595),
}

# ── CHIRPS resampling constants ────────────────────────────────────────────────
CHIRPS_NODATA_THRESHOLD = -100.0  # values ≤ this are treated as missing
CHIRPS_BLOCK = 7                  # 7×7 CHIRPS pixels averaged per SMAP cell (~1 footprint)

# ── Country / region definitions ──────────────────────────────────────────────
# Mirrors CNN/CHIRPS_processing/chirps_regions_monthly_lvl_1.py exactly.
# Keys are region slugs; values are GADM NAME_1 admin units merged into that zone.
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


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def parse_smap_date(filename: str) -> datetime:
    """SMAP_L3_SM_P_YYYYMMDD_*.h5 → datetime"""
    return datetime.strptime(Path(filename).name.split('_')[4], '%Y%m%d')


# ─────────────────────────────────────────────────────────────────────────────
# Composite lat/lon
# ─────────────────────────────────────────────────────────────────────────────

def build_composite_lat_lon() -> tuple[np.ndarray, np.ndarray]:
    """
    Build full-grid (406×964) composite lat/lon arrays by sampling 30 SMAP
    files spread across the archive.  Both AM and PM passes are used to
    maximise grid coverage.
    Returns (lat_comp, lon_comp), each (406, 964) float32.
    """
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
    """
    Extract 8-channel SMAP patch from one daily H5 file.
    Returns (8, WIN, WIN) float32 with NaN for missing / low-quality values.
    Applies quality-flag bit-0 filter to both AM and PM soil moisture.
    """
    result = np.full((N_FEATURES, WIN, WIN), np.nan, dtype=np.float32)

    with h5py.File(file_path, 'r') as f:
        am_qual = f[f'{AM_GROUP}/retrieval_qual_flag'][:]
        pm_qual = f[f'{PM_GROUP}/retrieval_qual_flag_pm'][:]

        for ch, (group, dataset, _) in enumerate(FEATURES):

            # tb_polarization_diff is derived (not a stored field)
            if dataset == 'tb_polarization_diff':
                tb_v = f[f'{group}/tb_v_corrected'][r0:r1, c0:c1].astype(np.float32)
                tb_h = f[f'{group}/tb_h_corrected'][r0:r1, c0:c1].astype(np.float32)
                valid = (tb_v != SMAP_FILL) & (tb_h != SMAP_FILL)
                result[ch] = np.where(valid, tb_v - tb_h, np.nan)
                continue

            data = f[f'{group}/{dataset}'][r0:r1, c0:c1].astype(np.float32)
            mask = data != SMAP_FILL

            # Soil moisture: also filter by retrieval quality flag bit 0
            if 'soil_moisture' in dataset:
                qual = am_qual if group == AM_GROUP else pm_qual
                qual_crop = qual[r0:r1, c0:c1]
                mask = mask & ((qual_crop & 1) == 0)

            result[ch] = np.where(mask, data, np.nan)

    return result


def build_smap_cube(months: list[pd.Timestamp],
                    r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    """
    Composite daily SMAP files into monthly nanmean for a country window.
    Returns (T, 8, WIN, WIN) float32.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# CHIRPS resampling
# ─────────────────────────────────────────────────────────────────────────────

def resample_chirps_to_smap(tif_path: Path,
                             lat_grid: np.ndarray,
                             lon_grid: np.ndarray) -> np.ndarray:
    """
    Resample one CHIRPS monthly GeoTIFF to the SMAP 64×64 grid.

    For each SMAP cell the NaN-aware mean of the 7×7 CHIRPS pixel block
    centred on that cell is used (≈1 SMAP footprint at 0.05° CHIRPS res).
    Returns (WIN, WIN) float32; NaN where no CHIRPS data exists (ocean, OOB).
    """
    result = np.full((WIN, WIN), np.nan, dtype=np.float32)
    valid_cells = ~np.isnan(lat_grid) & ~np.isnan(lon_grid)
    if not valid_cells.any():
        return result

    # Geographic extent of the SMAP window + half a CHIRPS block margin
    margin = (CHIRPS_BLOCK // 2 + 1) * 0.05
    lat_min = float(np.nanmin(lat_grid)) - margin
    lat_max = float(np.nanmax(lat_grid)) + margin
    lon_min = float(np.nanmin(lon_grid)) - margin
    lon_max = float(np.nanmax(lon_grid)) + margin

    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        # Clamp to TIF extent (CHIRPS Africa: lon [-20,55], lat [-40,40])
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

    # Mask CHIRPS nodata
    chirps_arr = np.where(chirps_arr <= CHIRPS_NODATA_THRESHOLD, np.nan, chirps_arr)

    # NaN-aware 7×7 block mean via sum/count decomposition
    not_nan = (~np.isnan(chirps_arr)).astype(np.float32)
    filled  = np.where(np.isnan(chirps_arr), 0.0, chirps_arr)
    n2 = CHIRPS_BLOCK * CHIRPS_BLOCK
    sum_v = uniform_filter(filled,  size=CHIRPS_BLOCK, mode='constant', cval=0.0) * n2
    sum_n = uniform_filter(not_nan, size=CHIRPS_BLOCK, mode='constant', cval=0.0) * n2
    block_mean = np.full_like(sum_v, np.nan)
    valid_n = sum_n > 0
    block_mean[valid_n] = sum_v[valid_n] / sum_n[valid_n]

    # Map each SMAP cell centre to its row/col in the windowed CHIRPS array.
    # win_transform.c = left edge longitude, win_transform.f = top edge latitude.
    # Pixel (row r, col c) occupies the geographic interval:
    #   lat: [win_top - (r+1)*res, win_top - r*res]
    #   lon: [win_left + c*res,    win_left + (c+1)*res]
    # → row index containing lat = floor((win_top - lat) / res)
    CHIRPS_RES = abs(win_transform.a)  # ≈ 0.05°
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
    """
    Resample CHIRPS monthly TIFs for all months to the SMAP grid.
    Returns (T, WIN, WIN) float32.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Region masks
# ─────────────────────────────────────────────────────────────────────────────

def build_region_mask(country: str,
                      lat_grid: np.ndarray,
                      lon_grid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Build (R, WIN, WIN) boolean mask showing which SMAP cells fall inside each
    agro-ecological region.

    Uses matplotlib.path.Path.contains_points (lon, lat) for polygon testing.
    Returns (mask_array, region_names_sorted).
    """
    cfg = COUNTRIES[country]
    gadm_path = GADM_DIR / cfg['gadm']
    with open(gadm_path) as fh:
        gadm = json.load(fh)

    valid = ~np.isnan(lat_grid) & ~np.isnan(lon_grid)
    pts = np.column_stack([lon_grid[valid], lat_grid[valid]])  # (N, 2) in lon/lat order

    region_names = sorted(cfg['regions'].keys())
    masks = []

    for slug in region_names:
        name1_vals = set(cfg['regions'][slug])

        # Collect polygon rings for all admin units that belong to this region
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

        # Test all valid SMAP cells against every ring
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


# ─────────────────────────────────────────────────────────────────────────────
# Target months
# ─────────────────────────────────────────────────────────────────────────────

def target_months(country: str) -> list[pd.Timestamp]:
    """
    Return months that are present in ALL of the country's CHIRPS region CSVs
    AND fall within SMAP availability (>= 2015-04-01).
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

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

        # Summary
        smap_vf   = np.count_nonzero(~np.isnan(smap_cube))   / smap_cube.size
        chirps_vf = np.count_nonzero(~np.isnan(chirps_cube)) / chirps_cube.size
        print(f'\n  SMAP coverage:   {100*smap_vf:.1f}%')
        print(f'  CHIRPS coverage: {100*chirps_vf:.1f}%')
        print(f'  Regions ({len(region_names)}): {region_names}')

        # Save
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

        # Quick sanity check
        loaded = np.load(out_path, allow_pickle=True)
        print(f'  Verified: smap={loaded["smap_cube"].shape}  '
              f'chirps={loaded["chirps_cube"].shape}  '
              f'mask={loaded["region_mask"].shape}  '
              f'T={loaded["dates"].shape[0]} months')

    print('\nDone.')
