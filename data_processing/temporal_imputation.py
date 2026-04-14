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

ARRAY_DIR = Path('earthdata-drought-crop-forecast/3d_numpy_arrays/soil_moisture_array')

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


def _vectorized_temporal_interp(data_2d):
    '''Vectorized temporal interpolation for a (T, N) array.

    Equivalent to calling temporal_interpolate_1d on each column but uses
    forward/backward index accumulation + linear weighting instead of a
    Python loop — roughly 50-100× faster on large grids.
    '''
    T, N = data_2d.shape
    valid = ~np.isnan(data_2d)          # (T, N)
    any_valid = valid.any(axis=0)       # (N,)

    timesteps = np.arange(T, dtype=np.int32)[:, None]  # (T, 1)

    # Forward index: last valid timestep <= t  (−1 if none)
    fwd_idx = np.where(valid, timesteps, np.int32(-1))          # (T, N)
    np.maximum.accumulate(fwd_idx, axis=0, out=fwd_idx)

    # Backward index: next valid timestep >= t  (T if none)
    bwd_idx = np.where(valid, timesteps, np.int32(T))           # (T, N)
    bwd_idx = np.minimum.accumulate(bwd_idx[::-1], axis=0)[::-1].copy()

    # Look up values at nearest forward / backward valid timesteps
    fwd_safe = np.clip(fwd_idx, 0, T - 1)
    bwd_safe = np.clip(bwd_idx, 0, T - 1)
    pixel_idx = np.arange(N)[None, :]                           # (1, N)
    fwd_vals = data_2d[fwd_safe, pixel_idx]                     # (T, N)
    bwd_vals = data_2d[bwd_safe, pixel_idx]                     # (T, N)

    # Linear interpolation weight: 0 at fwd, 1 at bwd
    gap = (bwd_idx - fwd_idx).astype(np.float32)
    gap[gap == 0] = 1.0
    weight = (timesteps - fwd_idx).astype(np.float32) / gap

    result = fwd_vals * (1.0 - weight) + bwd_vals * weight      # (T, N)

    # Edge handling: constant fill (matches np.interp behaviour)
    no_prev = (fwd_idx == -1)   # before first valid obs
    no_next = (bwd_idx == T)    # after last valid obs
    result = np.where(no_prev & ~no_next, bwd_vals, result)
    result = np.where(no_next & ~no_prev, fwd_vals, result)
    result = np.where(no_prev & no_next, np.nan, result)

    # Bounded edge fill: truncate beyond MAX_EDGE_FILL from first/last valid
    first_valid_t = np.where(any_valid, np.argmax(valid, axis=0), T).astype(np.int32)
    last_valid_t  = np.where(any_valid, T - 1 - np.argmax(valid[::-1], axis=0), -1).astype(np.int32)

    fill_start = np.maximum(0, first_valid_t - MAX_EDGE_FILL)              # (N,)
    fill_end   = np.minimum(T, last_valid_t + MAX_EDGE_FILL + 1)           # (N,)

    result[timesteps < fill_start[None, :]] = np.nan
    result[timesteps >= fill_end[None, :]]  = np.nan

    # Keep original valid values untouched
    result[valid] = data_2d[valid]

    return result


def impute_cube_3d(cube):
    '''Imputes a (T, H, W) cube — vectorized temporal interpolation.

    Returns (imputed_cube, nan_before, nan_after) for verification.
    '''
    T, H, W = cube.shape
    nan_before = np.isnan(cube).sum()

    flat = cube.reshape(T, -1)                          # (T, H*W)
    imputed = _vectorized_temporal_interp(flat).reshape(T, H, W)

    # Permanently NaN pixels → 0
    always_nan_mask = np.all(np.isnan(cube), axis=0)     # (H, W)
    imputed[:, always_nan_mask] = 0.0
    imputed = np.nan_to_num(imputed, nan=0.0)

    nan_after = np.isnan(imputed).sum()
    return imputed.astype(np.float32), nan_before, nan_after


def impute_cube_4d(cube):
    '''Imputes a (T, C, H, W) cube — vectorized temporal interpolation per channel.

    Each channel is treated independently because they have different NaN patterns
    (e.g. tb_polarization_diff is 97% filled vs soil_moisture at 63%).
    Processes one channel at a time to limit memory to ~O(T × H × W).
    '''
    T, C, H, W = cube.shape
    nan_before = np.isnan(cube).sum()
    imputed = cube.copy()

    for ch in range(C):
        ch_flat = cube[:, ch].reshape(T, -1)                    # (T, H*W)
        interpolated = _vectorized_temporal_interp(ch_flat)
        imputed[:, ch] = interpolated.reshape(T, H, W)

        # Permanently NaN pixels for this channel → 0
        always_nan_ch = np.all(np.isnan(cube[:, ch]), axis=0)   # (H, W)
        imputed[:, ch][:, always_nan_ch] = 0.0
        print(f'    channel {ch} done')

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
    '''Imputes smap_multifeature_africa_3day.npz → smap_multifeature_africa_3day_imputed.npz'''
    src = ARRAY_DIR / 'smap_multifeature_africa_3day.npz'
    dst = ARRAY_DIR / 'smap_multifeature_africa_3day_imputed.npz'

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
