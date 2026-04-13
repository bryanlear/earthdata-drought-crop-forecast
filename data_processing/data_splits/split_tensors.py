'''Temporal train / validation / test splits for SMAP tensors

Split boundaries (by year):
    Train : 2015-04-02  →  2022-12-31   (idx   0 – 915,   916 composites)
    Val   : 2023-01-01  →  2024-12-31   (idx 916 – 1159,  244 composites)
    Test  : 2025-01-02  →  2026-04-08   (idx 1160 – 1323, 164 composites)

Each 3-day composite = 1 time step

Saves per-split .pt files and trainset pixel wise normalization stats
'''

import torch
import numpy as np
from pathlib import Path

TENSOR_DIR = Path('/earthdata-drought-crop-forecast/'
                  '3d_numpy_arrays/soil_moisture_array/3d_tensor_time_lat_long')
OUT_DIR = Path('/earthdata-drought-crop-forecast/'
               'data_processing/data_splits')
OUT_DIR.mkdir(parents=True, exist_ok=True)


SPLIT_YEARS = {
    'train': (None, '2023'),   # everything before 2023
    'val':   ('2023', '2025'), # 2023-01-01 to 2024-12-31
    'test':  ('2025', None),   # 2025 onward
}

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

# normalization (stats train subset only)
def compute_pixel_stats(tensor, time_slice):
    '''Compute per-pixel mean and std from training
    ----------
    tensor : Tensor
        Shape (T, ..., H, W)
    time_slice : slice
        Which time indices belong to the train set
    '''
    train = tensor[time_slice]  # (T_train, [C,] H, W)
    mean = train.mean(dim=0, keepdim=True)   # (1, [C,] H, W)
    std  = train.std(dim=0, keepdim=True)    # (1, [C,] H, W)
    std[std == 0] = 1.0  # avoid division by zero for constant pixels
    return mean, std


def split_and_save(name, pt_file):
    data = torch.load(TENSOR_DIR / pt_file, weights_only=False)
    tensor = data['tensor']              # (T, [C,] H, W)
    dates  = data['dates']               # list[str]
    feature_names = data.get('feature_names', None)

    tr_sl, va_sl, te_sl = split_indices(dates)

    # Pixel-wise normalisation stats from train set
    mean, std = compute_pixel_stats(tensor, tr_sl)

    # Save
    stats_path = OUT_DIR / f'{name}_norm_stats.pt'
    torch.save({'mean': mean.squeeze(0), 'std': std.squeeze(0)}, stats_path)

    for split_name, sl in [('train', tr_sl), ('val', va_sl), ('test', te_sl)]:
        t = tensor[sl]
        d = dates[sl]

        # Normalise using train stats
        t_norm = (t - mean) / std

        payload = {
            'tensor': t,           # raw values
            'tensor_norm': t_norm, # normalised
            'dates': d,
        }
        if feature_names is not None:
            payload['feature_names'] = feature_names

        out_path = OUT_DIR / f'{name}_{split_name}.pt'
        torch.save(payload, out_path)

        size_mb = out_path.stat().st_size / 1e6
        print(f'  {split_name:5s}  {str(d[0]):>10s} → {str(d[-1]):<10s}  '
              f'shape={tuple(t.shape)}  ({size_mb:.1f} MB)')

    print(f'  stats  → {stats_path.name}  '
          f'mean_shape={tuple(mean.squeeze(0).shape)}')


if __name__ == '__main__':
    print('Single-channel (soil moisture):')
    split_and_save('sm', 'sm_tensor_T_Lat_Lon.pt')

    print('\nMulti-feature (8 channels):')
    split_and_save('multifeature', 'multifeature_tensor_T_C_Lat_Lon.pt')

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
