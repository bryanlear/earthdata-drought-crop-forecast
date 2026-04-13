'''Sliding-window PyTorch Dataset for SMAP drought forecasting.

Each sample = window of consecutive composites (default 10 = 30 days)
and target = composite following that window

Usage
-----
    from smap_dataset import SMAPDataset

    ds = SMAPDataset(split='train', feature='multifeature', window_size=10)
    x, y, date = ds[0]
    # x: (window_size, C, H, W)   normalised input sequence
    # y: (C, H, W)                normalised target (next step)
    # date: str                   date of the target composite
'''

import torch
from torch.utils.data import Dataset
from pathlib import Path

SPLIT_DIR = Path('/earthdata-drought-crop-forecast/'
                 'data_processing/data_splits')

# 30 days / 3 days per composite = 10 composites
DEFAULT_WINDOW = 10


class SMAPDataset(Dataset):
    '''Sliding-window dataset over temporal split tensors.

    Parameters
    ----------
    split : str
        One of 'train', 'val', 'test'.
    feature : str
        'sm' for single-channel or 'multifeature' for 8-channel.
    window_size : int
        Number of composites in the input sequence (default 10 → 30 days).
    normalize : bool
        If True (default), return pixel-wise normalised data.
    '''

    def __init__(self, split='train', feature='multifeature',
                 window_size=DEFAULT_WINDOW, normalize=True):
        path = SPLIT_DIR / f'{feature}_{split}.pt'
        data = torch.load(path, weights_only=False)

        key = 'tensor_norm' if normalize else 'tensor'
        self.tensor = data[key]   # (T, [C,] H, W)
        self.dates  = data['dates']
        self.feature_names = data.get('feature_names', None)
        self.window_size = window_size

    def __len__(self):
        # Each sample uses window_size inputs + 1 target
        return len(self.tensor) - self.window_size

    def __getitem__(self, idx):
        x = self.tensor[idx : idx + self.window_size]        # (W, [C,] H, W)
        y = self.tensor[idx + self.window_size]               # ([C,] H, W)
        date = str(self.dates[idx + self.window_size])
        return x, y, date


if __name__ == '__main__':
    for feat in ['sm', 'multifeature']:
        for split in ['train', 'val', 'test']:
            ds = SMAPDataset(split=split, feature=feat)
            x, y, d = ds[0]
            print(f'{feat:>14s}/{split:5s}  len={len(ds):>4d}  '
                  f'x={tuple(x.shape)}  y={tuple(y.shape)}  '
                  f'first_target_date={d}')
