import h5py
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime

SMAP_DATA_DIR = Path('/Volumes/bryan_SSD/spl3smp')
AM_GROUP = 'Soil_Moisture_Retrieval_Data_AM'
PM_GROUP = 'Soil_Moisture_Retrieval_Data_PM'

# West Arsi, Ethiopia centroid
TARGET_LAT = 7.25
TARGET_LON = 39.0
WINDOW_SIZE = 64

# Feature channels selected for drought / crop-failure detection.
# Each tuple: (hdf5_group, dataset_name, output_name)
#
# Rationale per channel:
#   soil_moisture_am        — primary drought signal; low SM = moisture deficit
#   soil_moisture_pm        — 2nd daily pass on different swath; boosts coverage and
#                             captures diurnal SM change (AM–PM delta = evapotranspiration proxy)
#   surface_temp_am         — night-time surface temperature; baseline thermal state
#   surface_temp_pm         — daytime peak temperature; AM→PM delta = thermal inertia,
#                             which is lower for dry bare soil (crop failure signature)
#   vegetation_water        — canopy water content from SMAP ancillary; drops before
#                             visible wilting — early drought warning
#   vegetation_opacity      — microwave optical depth of canopy; proxy for above-ground
#                             biomass; declines during prolonged drought
#   sm_error                — retrieval uncertainty; lets the model down-weight noisy pixels
#                             (e.g. RFI-affected or mixed land/water grid cells)
#   tb_polarization_diff    — (tb_v - tb_h) brightness temperature polarization difference;
#                             sensitive to surface roughness, soil moisture, and vegetation
#                             structure; larger delta = smoother/wetter surface
#   bulk_density            — soil bulk density (semi-static); controls water-holding
#                             capacity and infiltration; spatial context layer
FEATURES = [
    (AM_GROUP, 'soil_moisture',            'soil_moisture_am'),
    (PM_GROUP, 'soil_moisture_pm',         'soil_moisture_pm'),
    (AM_GROUP, 'surface_temperature',      'surface_temp_am'),
    (PM_GROUP, 'surface_temperature_pm',   'surface_temp_pm'),
    (AM_GROUP, 'vegetation_water_content', 'vegetation_water'),
    (AM_GROUP, 'vegetation_opacity',       'vegetation_opacity'),
    (AM_GROUP, 'tb_polarization_diff',     'tb_polarization_diff'),  # computed: tb_v - tb_h
    (AM_GROUP, 'bulk_density',             'bulk_density'),
]

N_FEATURES = len(FEATURES)


def parse_date_from_filename(filename):
    '''Extracts date from SMAP: SMAP_L3_SM_P_YYYYMMDD_*.h5'''
    date_str = filename.split('_')[4]
    return datetime.strptime(date_str, '%Y%m%d')


def find_center_index(file_path, center_lat=TARGET_LAT, center_lon=TARGET_LON):
    '''Finds (row, col) grid index closest to target.

    Fill values (-9999) in lat/lon are set to inf so they never win the argmin.
    '''
    with h5py.File(file_path, 'r') as f:
        lat = f[f'{AM_GROUP}/latitude'][:]
        lon = f[f'{AM_GROUP}/longitude'][:]

    fill_mask = (lat == -9999.0) | (lon == -9999.0)
    distances_sq = (lat - center_lat)**2 + (lon - center_lon)**2
    distances_sq[fill_mask] = np.inf

    if np.all(np.isinf(distances_sq)):
        raise ValueError(f'No valid lat/lon pixels in {file_path}')

    center_row, center_col = np.unravel_index(
        np.argmin(distances_sq), distances_sq.shape
    )
    return int(center_row), int(center_col)


def _find_reference_file(data_dir):
    '''Returns a well-populated file for centre-pixel lookup (skips early commissioning files).'''
    h5_files = sorted(
        data_dir.glob('SMAP_L3_SM_P_*.h5'),
        key=lambda p: parse_date_from_filename(p.name)
    )
    if not h5_files:
        raise FileNotFoundError(f'No SMAP H5 files found in {data_dir}')
    return h5_files[min(50, len(h5_files) - 1)]


def _crop_window(full_array, center_row, center_col, window_size=WINDOW_SIZE):
    '''Extracts a fixed (window_size × window_size) patch from a 2D array with NaN edge-padding.'''
    half = window_size // 2
    row_start = center_row - half
    row_end = center_row + half
    col_start = center_col - half
    col_end = center_col + half

    window = np.full((window_size, window_size), np.nan, dtype=np.float32)
    global_rows, global_cols = full_array.shape

    g_r0 = max(0, row_start)
    g_r1 = min(global_rows, row_end)
    g_c0 = max(0, col_start)
    g_c1 = min(global_cols, col_end)

    w_r0 = max(0, -row_start)
    w_r1 = window_size - max(0, row_end - global_rows)
    w_c0 = max(0, -col_start)
    w_c1 = window_size - max(0, col_end - global_cols)

    window[w_r0:w_r1, w_c0:w_c1] = full_array[g_r0:g_r1, g_c0:g_c1]
    return window


def extract_multifeature_window(file_path, center_row, center_col, window_size=WINDOW_SIZE):
    '''Extracts all feature channels as a (C, 64, 64) array from one SMAP file.

    Each channel is independently masked:
        - fill values (-9999.0) → NaN
        - quality flag applied to soil_moisture channels (bit 0 == 0 = recommended)
    Other channels (temperature, vegetation, etc.) use fill-value masking only
    since they have their own internal QC.
    '''
    result = np.full((N_FEATURES, window_size, window_size), np.nan, dtype=np.float32)

    with h5py.File(file_path, 'r') as f:
        # Read quality flags once for AM and PM
        am_qual = f[f'{AM_GROUP}/retrieval_qual_flag'][:]
        pm_qual = f[f'{PM_GROUP}/retrieval_qual_flag_pm'][:]

        for ch_idx, (group, dataset, _) in enumerate(FEATURES):
            # Compute tb polarization difference (tb_v - tb_h)
            if dataset == 'tb_polarization_diff':
                tb_v = f[f'{group}/tb_v_corrected'][:]
                tb_h = f[f'{group}/tb_h_corrected'][:]
                mask = (tb_v != -9999.0) & (tb_h != -9999.0)
                data = np.where(mask, tb_v - tb_h, np.nan)
                result[ch_idx] = _crop_window(data, center_row, center_col, window_size)
                continue

            data = f[f'{group}/{dataset}'][:]
            mask = (data != -9999.0)

            # Apply quality filter only to soil moisture channels
            if 'soil_moisture' in dataset:
                qual = am_qual if group == AM_GROUP else pm_qual
                mask = mask & ((qual & 1) == 0)

            clean = np.where(mask, data, np.nan)
            result[ch_idx] = _crop_window(clean, center_row, center_col, window_size)

    return result


def composite_3day(feature_stacks):
    '''Composites a list of (C, 64, 64) arrays via per-channel nanmean.'''
    stack = np.stack(feature_stacks, axis=0)  # (3, C, 64, 64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(stack, axis=0).astype(np.float32)  # (C, 64, 64)


def load_multifeature_time_series(data_dir=SMAP_DATA_DIR, window_size=WINDOW_SIZE, composite_days=3):
    '''Loads all SMAP files into a (T', C, 64, 64) composited multi-feature cube.

    Output shape: (T', 8, 64, 64) — channels-first for PyTorch Conv2d / ViT patch embedding.
    '''
    h5_files = sorted(
        data_dir.glob('SMAP_L3_SM_P_*.h5'),
        key=lambda p: parse_date_from_filename(p.name)
    )
    if not h5_files:
        raise FileNotFoundError(f'No SMAP H5 files found in {data_dir}')

    ref_file = _find_reference_file(data_dir)
    center_row, center_col = find_center_index(ref_file)
    print(f'West Arsi centre index: row={center_row}, col={center_col}')
    print(f'Features ({N_FEATURES}): {[f[2] for f in FEATURES]}')

    daily_dates = [parse_date_from_filename(f.name) for f in h5_files]
    n_daily = len(h5_files)
    daily_windows = []

    for i, fpath in enumerate(h5_files):
        daily_windows.append(
            extract_multifeature_window(fpath, center_row, center_col, window_size)
        )
        if (i + 1) % 500 == 0:
            print(f'  Extracted {i + 1}/{n_daily} daily files')

    # Non-overlapping 3-day bins
    composites = []
    composite_dates = []
    for start in range(0, n_daily, composite_days):
        end = min(start + composite_days, n_daily)
        composites.append(composite_3day(daily_windows[start:end]))
        composite_dates.append(daily_dates[end - 1])

    cube = np.stack(composites, axis=0)  # (T', C, 64, 64)

    print(f'Composited {n_daily} daily → {cube.shape[0]} × {composite_days}-day frames')
    print(f'Shape: {cube.shape} (T, C, H, W)')
    print(f'Date range: {composite_dates[0]:%Y-%m-%d} to {composite_dates[-1]:%Y-%m-%d}')
    return cube, composite_dates


def save_cube(cube, dates, feature_names, out_path):
    '''Saves multi-feature cube to compressed .npz.

    Contents:
        cube           — (T', C, 64, 64) float32
        dates          — (T',) ISO date strings
        feature_names  — (C,) channel names for indexing
    '''
    date_strings = np.array([d.strftime('%Y-%m-%d') for d in dates])
    feature_arr = np.array(feature_names)
    np.savez_compressed(out_path, cube=cube, dates=date_strings, feature_names=feature_arr)
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f'Saved {out_path} ({size_mb:.1f} MB)')


OUT_DIR = Path('/Users/bry_lee/earthdata-drought-crop-forecast/3d_numpy_arrays/soil_moisture_array')


if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / 'smap_multifeature_west_arsi_3day.npz'

    cube, dates = load_multifeature_time_series()
    feature_names = [f[2] for f in FEATURES]
    save_cube(cube, dates, feature_names, out_path)

    # Verify
    loaded = np.load(out_path)
    print(f'\nVerified: cube={loaded["cube"].shape}, dates={loaded["dates"].shape}')
    print(f'Channels: {list(loaded["feature_names"])}')

    # Per-channel coverage stats for first composite
    sample = loaded['cube'][50]  # a mid-range composite
    for i, name in enumerate(loaded['feature_names']):
        valid = np.count_nonzero(~np.isnan(sample[i]))
        print(f'  {name:25s}: {100*valid/sample[i].size:.1f}% coverage')
