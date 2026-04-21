"""
DroughtDataset: loads 6-country SMAP/CHIRPS cubes and aligns CSV drought labels.
Steps: load .npz per country → temporal split → stack SMAP+CHIRPS (9ch)
       → pad masks/labels to R_MAX=6 → compute training channel stats → z-score normalize.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

R_MAX = 6

COUNTRIES = ['eritrea', 'kenya', 'somalia', 'south_sudan', 'sudan', 'ethiopia', 'djibouti']

_CNN_DIR = Path(__file__).resolve().parent.parent
COUNTRY_NPZ_DIRS: dict[str, Path] = {
    'ethiopia': _CNN_DIR / 'ethiopia',
}
COUNTRY_CSV_DIRS: dict[str, Path] = {
    'ethiopia': _CNN_DIR / 'ethiopia',
}

SPLIT_DATES = {
    'train': ('2015-04-01', '2022-12-01'),
    'val':   ('2023-01-01', '2023-12-01'),
    'test':  ('2024-01-01', '2026-03-01'),
}


def load_labels(csv_dir: Path,
                region_names: list,
                dates: np.ndarray,
                label_col: str = 'drought_class_spi3') -> np.ndarray:
    T, R = len(dates), len(region_names)
    labels = np.full((T, R), -100, dtype=np.int64)
    date_to_t = {d: i for i, d in enumerate(dates)}

    for r, slug in enumerate(region_names):
        csv = csv_dir / f'{slug}_chirps_monthly.csv'
        if not csv.exists():
            continue
        df = pd.read_csv(csv, parse_dates=['date'])
        df = df.dropna(subset=[label_col])
        for _, row in df.iterrows():
            key = row['date'].strftime('%Y-%m-%d')
            if key in date_to_t:
                labels[date_to_t[key], r] = int(row[label_col])

    return labels


class DroughtDataset(Dataset):

    def __init__(self,
                 npz_dir: Path,
                 csv_dir: Path,
                 split: str,
                 channel_mean: Optional[np.ndarray] = None,
                 channel_std:  Optional[np.ndarray] = None,
                 label_col: str = 'drought_class_spi3'):
        assert split in SPLIT_DATES, f"split must be one of {list(SPLIT_DATES)}"
        self.split = split
        start, end = [pd.Timestamp(d) for d in SPLIT_DATES[split]]

        raw_images: list[np.ndarray] = []
        self._masks:  list[np.ndarray] = []
        self._labels: list[np.ndarray] = []

        for country in COUNTRIES:
            _npz_dir = COUNTRY_NPZ_DIRS.get(country, npz_dir)
            _csv_dir = COUNTRY_CSV_DIRS.get(country, csv_dir / country)

            npz_path = _npz_dir / f'{country}_smap_monthly.npz'
            if not npz_path.exists():
                print(f'WARNING: {npz_path.name} not found — skipping.')
                continue

            npz          = np.load(npz_path, allow_pickle=True)
            smap_cube    = npz['smap_cube']
            chirps_cube  = npz['chirps_cube']
            region_mask  = npz['region_mask']
            region_names = list(npz['region_names'])
            dates        = npz['dates']
            R = len(region_names)

            labels_TR = load_labels(_csv_dir, region_names, dates, label_col)

            for t, date_str in enumerate(dates):
                ts = pd.Timestamp(str(date_str))
                if not (start <= ts <= end):
                    continue

                image = np.concatenate(
                    [smap_cube[t], chirps_cube[t][np.newaxis]], axis=0
                ).astype(np.float32)


                masks_pad  = np.zeros((R_MAX, 64, 64), dtype=bool)
                labels_pad = np.full(R_MAX, -100, dtype=np.int64)
                masks_pad[:R]  = region_mask
                labels_pad[:R] = labels_TR[t]

                raw_images.append(image)
                self._masks.append(masks_pad)
                self._labels.append(labels_pad)

        if not raw_images:
            raise RuntimeError(f'No samples found for split="{split}".')

        raw_stack = np.stack(raw_images, axis=0)

        if channel_mean is None:
            self.channel_mean = np.nanmean(raw_stack, axis=(0, 2, 3)).astype(np.float32)
            self.channel_std  = np.nanstd( raw_stack, axis=(0, 2, 3)).astype(np.float32)
            self.channel_std  = np.where(self.channel_std < 1e-6,
                                         1.0, self.channel_std).astype(np.float32)
        else:
            self.channel_mean = channel_mean.astype(np.float32)
            self.channel_std  = channel_std.astype(np.float32)

        mean = self.channel_mean[:, None, None]
        std  = self.channel_std[:, None, None]

        self._images: list[np.ndarray] = []
        for img in raw_images:
            img = np.where(np.isnan(img), mean, img)
            img = (img - mean) / std
            self._images.append(img.astype(np.float32))

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self._images[idx]),
            torch.from_numpy(self._masks[idx]),
            torch.from_numpy(self._labels[idx]),
        )


def make_dataloaders(
    npz_dir: Path,
    csv_dir: Path,
    batch_size:  int = 16,
    num_workers: int = 0,
    label_col: str = 'drought_class_spi3',
):
    train_ds = DroughtDataset(npz_dir, csv_dir, 'train', label_col=label_col)
    val_ds   = DroughtDataset(npz_dir, csv_dir, 'val',
                              channel_mean=train_ds.channel_mean,
                              channel_std=train_ds.channel_std,
                              label_col=label_col)
    test_ds  = DroughtDataset(npz_dir, csv_dir, 'test',
                              channel_mean=train_ds.channel_mean,
                              channel_std=train_ds.channel_std,
                              label_col=label_col)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    return (train_loader, val_loader, test_loader,
            train_ds.channel_mean, train_ds.channel_std)
