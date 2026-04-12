'''Temporal interpolation imputation for SMAP 3-day composited cubes:

Strategy (per pixel, per channel):
    1. Linear interpolation along the time axis fills NaN timesteps that have
      VALID observations before AND after. Transient swath gaps that
       shift with SMAP's orbit. Linear interpolation rationale --> 3-day temporal temporal resolution needed for drought and hydroclimatologyt appl;ications
       
    2. Forward/backward fill for NaN runs at the start/end of time series where interpolation has no bracket. Uses nearest valid
       observation (≤ 5 composites ≈ 15 days) to avoid projecting stale data
       
    3. Permanently NaN pixels (ocean, lakes, ice) are filled with 0.0. These are non-land pixels at the
       window edge. Zero-fill is neutral sentinel --> won't bias learned features and model can learn to ignore them spatially

Outputs as new .npz files with '_imputed'.
'''

import numpy as np
from pathlib import Path

ARRAY_DIR = Path('/earthdata-drought-crop-forecast/3d_numpy_arrays/soil_moisture_array')

# Max gap (in composite steps) to forward/backward fill at series edges
MAX_EDGE_FILL = 5  # 5 × 3 days = 15 days


def temporal_interpolate_1d(series):
    '''Interpolates single 1D time series (T,) using linear interp + bounded edge fill

    1. np.interp with xp = valid indices fills interior gaps linearly
    2. Edge NaN runs (before first / after last valid obs) are filled only if gap ≤ MAX_EDGE_FILL steps. Prevention extrapolating stale data
    
    '''
    valid = ~np.isnan(series)
    if valid.sum() == 0:
        return series 
    if valid.all():
        return series

    indices = np.arange(len(series))
    valid_idx = indices[valid]
    valid_vals = series[valid]

    interpolated = np.interp(indices, valid_idx, valid_vals)

    # Restore NaN at edges beyond MAX_EDGE_FILL from nearest valid observation
    first_valid = valid_idx[0]
    last_valid = valid_idx[-1]

    # Leading edge: fill only up to MAX_EDGE_FILL steps before first observation
    if first_valid > 0:
        fill_start = max(0, first_valid - MAX_EDGE_FILL)
        interpolated[:fill_start] = np.nan

    # Trailing edge: fill only up to MAX_EDGE_FILL steps after last observation
    if last_valid < len(series) - 1:
        fill_end = min(len(series), last_valid + MAX_EDGE_FILL + 1)
        interpolated[fill_end:] = np.nan

    return interpolated


def impute_cube_3d(cube):
    '''Imputes a (T, H, W) cube — temporal interpolation per pixel.

    Returns (imputed_cube, nan_mask_before, nan_mask_after) for verification.
    '''
    T, H, W = cube.shape
    nan_before = np.isnan(cube).sum()
    imputed = cube.copy()

    for r in range(H):
        for c in range(W):
            series = imputed[:, r, c]
            if np.isnan(series).any():
                imputed[:, r, c] = temporal_interpolate_1d(series)

    # Fill permanently NaN pixels (never observed across full time series) with 0
    still_nan = np.isnan(imputed)
    always_nan_mask = np.all(np.isnan(cube), axis=0)  # (H, W) — permanent gaps
    for t in range(T):
        imputed[t][always_nan_mask] = 0.0

    # Any remaining NaN (edge-limited fill) → 0
    imputed = np.nan_to_num(imputed, nan=0.0)

    nan_after = np.isnan(imputed).sum()
    return imputed.astype(np.float32), nan_before, nan_after


def impute_cube_4d(cube):
    '''Imputes a (T, C, H, W) cube — temporal interpolation per channel per pixel.

    Each channel is treated independently because they have different NaN patterns
    (e.g. tb_polarization_diff is 97% filled vs soil_moisture at 63%).
    '''
    T, C, H, W = cube.shape
    nan_before = np.isnan(cube).sum()
    imputed = cube.copy()

    for ch in range(C):
        for r in range(H):
            for c in range(W):
                series = imputed[:, ch, r, c]
                if np.isnan(series).any():
                    imputed[:, ch, r, c] = temporal_interpolate_1d(series)

        # Fill permanently NaN pixels for this channel
        always_nan_ch = np.all(np.isnan(cube[:, ch]), axis=0)  # (H, W)
        for t in range(T):
            imputed[t, ch][always_nan_ch] = 0.0

    imputed = np.nan_to_num(imputed, nan=0.0)

    nan_after = np.isnan(imputed).sum()
    return imputed.astype(np.float32), nan_before, nan_after


def impute_single_channel():
    '''Imputes smap_sm_west_arsi_3day.npz → smap_sm_west_arsi_3day_imputed.npz'''
    src = ARRAY_DIR / 'smap_sm_west_arsi_3day.npz'
    dst = ARRAY_DIR / 'smap_sm_west_arsi_3day_imputed.npz'

    print(f'Loading {src.name}...')
    data = np.load(src)
    cube = data['sm_cube']
    dates = data['dates']
    print(f'  Shape: {cube.shape}, NaN: {np.isnan(cube).sum()}/{cube.size} ({100*np.isnan(cube).sum()/cube.size:.1f}%)')

    print('Imputing (temporal interpolation per pixel)...')
    imputed, nan_before, nan_after = impute_cube_3d(cube)

    print(f'  NaN before: {nan_before} → after: {nan_after}')
    print(f'  Filled: {nan_before - nan_after} pixels ({100*(nan_before - nan_after)/nan_before:.1f}% of NaNs)')

    np.savez_compressed(dst, sm_cube=imputed, dates=dates)
    size_mb = dst.stat().st_size / 1e6
    print(f'Saved {dst.name} ({size_mb:.1f} MB)\n')


def impute_multifeature():
    '''Imputes smap_multifeature_west_arsi_3day.npz → smap_multifeature_west_arsi_3day_imputed.npz'''
    src = ARRAY_DIR / 'smap_multifeature_west_arsi_3day.npz'
    dst = ARRAY_DIR / 'smap_multifeature_west_arsi_3day_imputed.npz'

    print(f'Loading {src.name}...')
    data = np.load(src)
    cube = data['cube']
    dates = data['dates']
    feature_names = data['feature_names']
    print(f'  Shape: {cube.shape}, NaN: {np.isnan(cube).sum()}/{cube.size} ({100*np.isnan(cube).sum()/cube.size:.1f}%)')

    for i, ch in enumerate(feature_names):
        ch_nan = np.isnan(cube[:, i]).sum()
        print(f'    {ch}: NaN={100*ch_nan/cube[:, i].size:.1f}%')

    print('Imputing (temporal interpolation per channel per pixel)...')
    imputed, nan_before, nan_after = impute_cube_4d(cube)

    print(f'  NaN before: {nan_before} → after: {nan_after}')
    print(f'  Filled: {nan_before - nan_after} pixels ({100*(nan_before - nan_after)/nan_before:.1f}% of NaNs)')

    for i, ch in enumerate(feature_names):
        ch_nan = np.isnan(imputed[:, i]).sum()
        vals = imputed[:, i][imputed[:, i] != 0.0]
        if len(vals) > 0:
            print(f'    {ch}: NaN={ch_nan}, range=[{vals.min():.4f}, {vals.max():.4f}]')

    np.savez_compressed(dst, cube=imputed, dates=dates, feature_names=feature_names)
    size_mb = dst.stat().st_size / 1e6
    print(f'Saved {dst.name} ({size_mb:.1f} MB)\n')


if __name__ == '__main__':
    impute_single_channel()
    impute_multifeature()
    print('Done. Imputed files saved alongside originals.')
