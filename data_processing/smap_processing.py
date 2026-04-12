import h5py
import numpy as np
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

SMAP_DATA_DIR = Path('/Volumes/bryan_SSD/spl3smp')
AM_GROUP = 'Soil_Moisture_Retrieval_Data_AM'

# West Arsi, Ethiopia centroid
TARGET_LAT = 7.25
TARGET_LON = 39.0
WINDOW_SIZE = 64


def find_center_index(file_path, center_lat=TARGET_LAT, center_lon=TARGET_LON):
    '''Finds (row, col) grid index closest to target.

    Squared Euclidean distance on EASE-2 lat/lon grids.
    Fill values (-9999) in lat/lon are set to inf so they never win the argmin.
    Squared distance avoids unnecessary sqrt — argmin is identical since sqrt is monotonic.
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
    '''Returns a file with sufficient lat/lon coverage for centre-pixel lookup.

    Early-mission files (March 2015) have very sparse coverage — most lat/lon
    cells are -9999. Skipping ahead to file index 50 avoids this.
    '''
    h5_files = sorted(
        data_dir.glob('SMAP_L3_SM_P_*.h5'),
        key=lambda p: parse_date_from_filename(p.name)
    )
    if not h5_files:
        raise FileNotFoundError(f'No SMAP H5 files found in {data_dir}')
    return h5_files[min(50, len(h5_files) - 1)]


def extract_centered_window(file_path, center_row, center_col, window_size=WINDOW_SIZE):
    '''Extracts fixed w x w soil moisture patch centred on West Arsi (center_row, center_col).

        1. Read soil_moisture and retrieval_qual_flag from AM group
        2. Apply fill-value mask INVALID (-9999.0) and quality-flag mask (bit 0 == 0 → recommended)
        3. Compute slice boundaries around centre pixel
        4. Pre allocate All-NaN (window_size × window_size) array — fixed tensor shape even when window overlaps grid edge (e.g. at
           high latitudes where EASE-2 rows are shorter)
        5. Out-of-bounds indices map to NaN padding (no IndexError)
        6. Copy VALID overlap into the pre allocated frame
    '''
    with h5py.File(file_path, 'r') as f:
        sm_data = f[f'{AM_GROUP}/soil_moisture'][:]
        qual_flag = f[f'{AM_GROUP}/retrieval_qual_flag'][:]

        valid_mask = (sm_data != -9999.0) & ((qual_flag & 1) == 0)
        sm_clean = np.where(valid_mask, sm_data, np.nan)

    half = window_size // 2
    row_start = center_row - half
    row_end = center_row + half
    col_start = center_col - half
    col_end = center_col + half

    # Pre allocate fixed-size output
    window = np.full((window_size, window_size), np.nan, dtype=np.float32)

    global_rows, global_cols = sm_clean.shape

    # Clip to VALID global array bounds
    g_r0 = max(0, row_start)
    g_r1 = min(global_rows, row_end)
    g_c0 = max(0, col_start)
    g_c1 = min(global_cols, col_end)

    # Offsets inside output window
    w_r0 = max(0, -row_start)
    w_r1 = window_size - max(0, row_end - global_rows)
    w_c0 = max(0, -col_start)
    w_c1 = window_size - max(0, col_end - global_cols)

    window[w_r0:w_r1, w_c0:w_c1] = sm_clean[g_r0:g_r1, g_c0:g_c1]

    return window


def parse_date_from_filename(filename):
    '''Extracts date from SMAP: SMAP_L3_SM_P_YYYYMMDD_*.h5'''
    date_str = filename.split('_')[4]
    return datetime.strptime(date_str, '%Y%m%d')


def composite_3day(windows):
    '''Composites list of (64, 64) windows via NaNmean

    Per-pixel nanmean across the 3-day stack:
        - All 3 days = NaN then pixel stays NaN
        - 1–3 days have data = mean of values fills pixel'''
    stack = np.stack(windows, axis=0)  # (3, 64, 64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(stack, axis=0).astype(np.float32)


def load_time_series(data_dir=SMAP_DATA_DIR, window_size=WINDOW_SIZE, composite_days=3):
    '''Loads all SMAP L3 files into a 3-day composited (T', 64, 64) array over West Arsi.

        1. Find centre pixel once — EASE-2 grid is static across all L3 granules.
        2. Glob + sort by parsed date (filesystem order unreliable on external SSD).
        3. Extract each daily window (fill + quality mask + edge clamp).
        4. Group into non-overlapping 3-day bins and nanmean each bin:
           - SMAP orbit shifts ~25° lon/day → 3 days tiles the window, boosting
             coverage from ~20% to ~50–60%.
           - nanmean averages over available passes, reducing single-pass noise
             by ~1/sqrt(n) while preserving NaN where all 3 days had gaps.
        5. Date assigned per composite = last day in the bin (the most recent
           observation that contributed).
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

    # Extract all daily windows
    daily_dates = [parse_date_from_filename(f.name) for f in h5_files]
    n_daily = len(h5_files)
    daily_windows = []

    for i, fpath in enumerate(h5_files):
        daily_windows.append(
            extract_centered_window(fpath, center_row, center_col, window_size)
        )
        if (i + 1) % 500 == 0:
            print(f'  Extracted {i + 1}/{n_daily} daily files')

    # Non-overlapping 3-day bins composite
    composites = []
    composite_dates = []
    for start in range(0, n_daily, composite_days):
        end = min(start + composite_days, n_daily)
        composites.append(composite_3day(daily_windows[start:end]))
        composite_dates.append(daily_dates[end - 1])  # label = last day in bin

    sm_cube = np.stack(composites, axis=0)

    print(f'Composited {n_daily} daily → {sm_cube.shape[0]} × {composite_days}-day frames')
    print(f'Date range: {composite_dates[0]:%Y-%m-%d} to {composite_dates[-1]:%Y-%m-%d}')
    return sm_cube, composite_dates


def save_cube(sm_cube, dates, out_path):
    '''Saves composited cube and date index to compressed .npz file
    Contents:
        sm_cube  — (T', 64, 64) float32 soil moisture array
        dates    — (T',) array of dates as ISO strings (YYYY-MM-DD)
    '''
    date_strings = np.array([d.strftime('%Y-%m-%d') for d in dates])
    np.savez_compressed(out_path, sm_cube=sm_cube, dates=date_strings)
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f'Saved {out_path} ({size_mb:.1f} MB)')


def visualize_window(sm_window, title=None):
    '''Visualizes a single (64, 64) soil moisture window.'''
    plt.figure(figsize=(8, 8))
    plt.imshow(sm_window, cmap='viridis_r', vmin=0.0, vmax=0.5)
    plt.colorbar(label='Volumetric Soil Moisture (cm³/cm³)')
    plt.title(title or f'SMAP L3 — West Arsi ({sm_window.shape[0]}×{sm_window.shape[1]})')
    plt.axis('off')
    plt.show()


OUT_DIR = Path(__file__).resolve().parent.parent / 'data'


if __name__ == '__main__':
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / 'smap_sm_west_arsi_3day.npz'

    sm_cube, dates = load_time_series()
    save_cube(sm_cube, dates, out_path)
    loaded = np.load(out_path)
    print(f'Verified: sm_cube={loaded["sm_cube"].shape}, dates={loaded["dates"].shape}')