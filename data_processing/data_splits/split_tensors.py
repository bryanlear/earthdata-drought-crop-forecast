'''Temporal train / validation / test splits for SMAP tensors

Split boundaries (by year):
    Train : 2015-04-02  →  2022-12-31   (idx   0 – 915,   916 composites)
    Val   : 2023-01-01  →  2024-12-31   (idx 916 – 1159,  244 composites)
    Test  : 2025-01-02  →  2026-04-08   (idx 1160 – 1323, 164 composites)

Each 3-day composite = 1 time step

Per-channel normalization:
    - Zeros = invalid (ocean, lakes, coverage gaps) → masked from stat computation
    - Vegetation channels (Ch4, Ch5) clipped at p1/p99 on train set before norm
    - Per-channel z-score using train-set non-zero mean/std
    - Zeros preserved as zeros in output; spatial mask saved
'''

import torch
import numpy as np
from pathlib import Path

TENSOR_DIR = Path('earthdata-drought-crop-forecast/'
                  '3d_numpy_arrays/soil_moisture_array/3d_tensor_time_lat_long')
OUT_DIR = Path('earthdata-drought-crop-forecast/'
               'data_processing/data_splits')
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLIP_CHANNELS = {4, 5}  # vegetation_water, vegetation_opacity
CLIP_LO, CLIP_HI = 0.01, 0.99


def _year_index(dates, year_str, first=True):
    for i, d in enumerate(dates):
        if str(d)[:4] == year_str:
            return i
    raise ValueError(f'Year {year_str} not found in dates')


def split_indices(dates):
    val_start = _year_index(dates, '2023')
    test_start = _year_index(dates, '2025')
    return (
        slice(0, val_start),          # train
        slice(val_start, test_start), # val
        slice(test_start, None),      # test
    )


def build_spatial_mask(tensor):
    '''Identify always-zero (invalid) pixels per channel.

    Returns bool mask (C, H, W) — True = valid pixel.
    '''
    # tensor: (T, C, H, W)
    C, H, W = tensor.shape[1], tensor.shape[2], tensor.shape[3]
    mask = torch.ones(C, H, W, dtype=torch.bool)
    for c in range(C):
        mask[c] = tensor[:, c, :, :].abs().sum(dim=0) > 0
    return mask


def compute_channel_stats(tensor, time_slice, spatial_mask, feature_names):
    '''Per-channel mean/std from train set, masking zeros.

    For CLIP_CHANNELS, clips non-zero train values at p1/p99 first
    and applies the same clipping to the full tensor in-place.

    Returns dict {channel_name: {mean, std, clip_lo, clip_hi}}.
    '''
    train = tensor[time_slice]  # (T_train, C, H, W)
    C = tensor.shape[1]
    stats = {}

    for c in range(C):
        name = feature_names[c] if feature_names else f'ch{c}'
        mask = spatial_mask[c]                          # (H, W)
        train_valid = train[:, c][:, mask]              # (T_train, n_valid)
        nonzero = train_valid[train_valid != 0]

        clip_lo, clip_hi = None, None

        if c in CLIP_CHANNELS and nonzero.numel() > 0:
            nz_np = nonzero.numpy()
            clip_lo = float(np.percentile(nz_np, CLIP_LO * 100))
            clip_hi = float(np.percentile(nz_np, CLIP_HI * 100))
            # Clip full tensor (all splits) in-place for this channel
            ch = tensor[:, c]                           # (T, H, W)
            valid_vals = ch[:, mask]                     # (T, n_valid)
            valid_vals.clamp_(min=clip_lo, max=clip_hi)
            ch[:, mask] = valid_vals
            # Recompute nonzero after clipping
            nonzero = tensor[time_slice, c][:, mask]
            nonzero = nonzero[nonzero != 0]
            print(f'  Ch{c} ({name}): clipped to [{clip_lo:.4f}, {clip_hi:.4f}]')

        mean = nonzero.mean().item()
        std = nonzero.std().item()
        stats[name] = {'mean': mean, 'std': std,
                        'clip_lo': clip_lo, 'clip_hi': clip_hi}
        print(f'  Ch{c} ({name}): mean={mean:.6f}, std={std:.6f}, '
              f'valid_px={mask.sum().item()}/{mask.numel()}')

    return stats


def normalize_tensor(tensor, spatial_mask, channel_stats, feature_names):
    '''Z-score normalize each channel using precomputed stats.

    Zeros (invalid pixels) are left untouched.
    Returns new tensor.
    '''
    out = tensor.clone()
    C = tensor.shape[1]
    for c in range(C):
        name = feature_names[c] if feature_names else f'ch{c}'
        s = channel_stats[name]
        mask = spatial_mask[c]                          # (H, W)
        valid = out[:, c][:, mask]                      # (T, n_valid)
        out[:, c][:, mask] = (valid - s['mean']) / s['std']
    return out


def split_and_save_multifeature(name, pt_file):
    data = torch.load(TENSOR_DIR / pt_file, weights_only=False)
    tensor = data['tensor'].clone()      # (T, C, H, W)
    dates  = data['dates']
    feature_names = data.get('feature_names', None)

    tr_sl, va_sl, te_sl = split_indices(dates)

    # Spatial mask: True = valid pixel per channel
    spatial_mask = build_spatial_mask(tensor)

    # Per-channel stats from train set (clips vegetation channels in-place)
    channel_stats = compute_channel_stats(
        tensor, tr_sl, spatial_mask, feature_names)

    # Normalize full tensor then split
    tensor_norm = normalize_tensor(
        tensor, spatial_mask, channel_stats, feature_names)

    # Save stats
    stats_path = OUT_DIR / f'{name}_norm_stats.pt'
    torch.save({
        'channel_stats': channel_stats,
        'spatial_mask': spatial_mask,
        'feature_names': feature_names,
    }, stats_path)

    for split_name, sl in [('train', tr_sl), ('val', va_sl), ('test', te_sl)]:
        t = tensor[sl]
        tn = tensor_norm[sl]
        d = dates[sl]

        payload = {
            'tensor': t,
            'tensor_norm': tn,
            'dates': d,
            'spatial_mask': spatial_mask,
        }
        if feature_names is not None:
            payload['feature_names'] = feature_names

        out_path = OUT_DIR / f'{name}_{split_name}.pt'
        torch.save(payload, out_path)

        size_mb = out_path.stat().st_size / 1e6
        print(f'  {split_name:5s}  {str(d[0]):>10s} → {str(d[-1]):<10s}  '
              f'shape={tuple(t.shape)}  ({size_mb:.1f} MB)')

    print(f'  stats  → {stats_path.name}')


def split_and_save_single(name, pt_file):
    '''Single-channel tensor — no clipping, simple zero-masked z-score.'''
    data = torch.load(TENSOR_DIR / pt_file, weights_only=False)
    tensor = data['tensor'].clone()      # (T, H, W)
    dates  = data['dates']

    tr_sl, va_sl, te_sl = split_indices(dates)

    # Spatial mask: always-zero pixels
    spatial_mask = tensor.abs().sum(dim=0) > 0  # (H, W)

    train_valid = tensor[tr_sl][:, spatial_mask]
    nonzero = train_valid[train_valid != 0]
    mean = nonzero.mean().item()
    std = nonzero.std().item()
    print(f'  mean={mean:.6f}, std={std:.6f}, '
          f'valid_px={spatial_mask.sum().item()}/{spatial_mask.numel()}')

    tensor_norm = tensor.clone()
    valid = tensor_norm[:, spatial_mask]
    tensor_norm[:, spatial_mask] = (valid - mean) / std

    stats_path = OUT_DIR / f'{name}_norm_stats.pt'
    torch.save({
        'mean': mean, 'std': std,
        'spatial_mask': spatial_mask,
    }, stats_path)

    for split_name, sl in [('train', tr_sl), ('val', va_sl), ('test', te_sl)]:
        t = tensor[sl]
        tn = tensor_norm[sl]
        d = dates[sl]

        payload = {
            'tensor': t,
            'tensor_norm': tn,
            'dates': d,
            'spatial_mask': spatial_mask,
        }

        out_path = OUT_DIR / f'{name}_{split_name}.pt'
        torch.save(payload, out_path)

        size_mb = out_path.stat().st_size / 1e6
        print(f'  {split_name:5s}  {str(d[0]):>10s} → {str(d[-1]):<10s}  '
              f'shape={tuple(t.shape)}  ({size_mb:.1f} MB)')

    print(f'  stats  → {stats_path.name}')


if __name__ == '__main__':
    print('Multi-feature (8 channels):')
    split_and_save_multifeature('multifeature', 'multifeature_tensor_T_C_Lat_Lon.pt')

    print('\n── Verification ──')
    for pt in sorted(OUT_DIR.glob('*.pt')):
        d = torch.load(pt, weights_only=False)
        if 'tensor' in d:
            t = d['tensor']
            tn = d['tensor_norm']
            print(f'{pt.name}: raw={tuple(t.shape)} norm={tuple(tn.shape)} '
                  f'dtype={t.dtype} NaN={torch.isnan(t).sum().item()}')
        else:
            print(f'{pt.name}: keys={list(d.keys())}')
