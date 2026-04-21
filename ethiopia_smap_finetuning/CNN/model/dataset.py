"""
DroughtDataset — loads all country SMAP cubes and CHIRPS CSV labels.

Each sample is one (country, month, region) triple:
    image  (10, CROP_SIZE, CROP_SIZE)  float32 — bbox crop of 9-ch data +
                                                  binary region mask as ch 10
    label  scalar int64                — 0=normal, 1=drought
    month  scalar int64                — month-of-year 1..12

Temporal split (chronological, no shuffling across time):
    train : 2015-04-01 → 2022-12-01
    val   : 2023-01-01 → 2023-12-01
    test  : 2024-01-01 → 2026-03-01
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


CROP_SIZE = 64
CROP_MARGIN = 4

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
    'val': ('2023-01-01', '2023-12-01'),
    'test': ('2024-01-01', '2026-03-01'),
}


def load_labels(csv_dir: Path,
                region_names: list,
                dates: np.ndarray,
                label_col: str = 'drought_class_spi3') -> np.ndarray:
    """
    Align per-region CSV labels to the npz dates array.

    Returns (T, R) int64 array; -100 where label is missing / NaN.
    Labels are binarised: 0 = normal, 1 = drought (moderate or severe).
    """
    T, R = len(dates), len(region_names)
    labels = np.full((T, R), -100, dtype=np.int64)
    date_to_t = {d: i for i, d in enumerate(dates)}

    for r, slug in enumerate(region_names):
        csv = csv_dir / f'{slug}_chirps_monthly.csv'
        if not csv.exists():
            continue
        df = pd.read_csv(csv, parse_dates=['date'])
        df = df.dropna(subset=[label_col])
        df['_t'] = df['date'].dt.strftime('%Y-%m-%d').map(date_to_t)
        df = df.dropna(subset=['_t'])
        if df.empty:
            continue
        t_idx = df['_t'].astype(int).values
        vals = df[label_col].astype(int).values
        vals = np.where(vals >= 1, 1, 0)
        labels[t_idx, r] = vals

    return labels


def _bbox(mask: np.ndarray, margin: int, size: int = 64):
    """Bounding box of non-zero pixels in mask, padded by margin, clamped to [0, size)."""
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    r0 = max(0, int(rows[0]) - margin)
    r1 = min(size, int(rows[-1]) + 1 + margin)
    c0 = max(0, int(cols[0]) - margin)
    c1 = min(size, int(cols[-1]) + 1 + margin)
    return r0, r1, c0, c1


def _resize(arr: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize (C, H, W) float32 array to (C, size, size)."""
    t = torch.from_numpy(arr).unsqueeze(0).float()
    t = torch.nn.functional.interpolate(
        t, size=(size, size), mode='bilinear', align_corners=False
    )
    return t.squeeze(0).numpy()


def _resolve_date_range(split: Optional[str],
                        date_range: Optional[tuple[str, str]]) -> tuple[pd.Timestamp, pd.Timestamp]:
    if split is not None and date_range is not None:
        raise ValueError('Specify either split or date_range, not both.')
    if split is None and date_range is None:
        raise ValueError('Either split or date_range must be provided.')

    if split is not None:
        if split not in SPLIT_DATES:
            raise ValueError(f"split must be one of {list(SPLIT_DATES)}, got {split!r}")
        start, end = SPLIT_DATES[split]
    else:
        start, end = date_range  # type: ignore[misc]

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts > end_ts:
        raise ValueError(f'Invalid date range: {start_ts} > {end_ts}')
    return start_ts, end_ts


class DroughtDataset(Dataset):

    def __init__(self,
                 npz_dir: Path,
                 csv_dir: Path,
                 split: Optional[str] = None,
                 date_range: Optional[tuple[str, str]] = None,
                 augment: Optional[bool] = None,
                 channel_mean: Optional[np.ndarray] = None,
                 channel_std: Optional[np.ndarray] = None,
                 label_col: str = 'drought_class_spi3'):
        start, end = _resolve_date_range(split, date_range)
        self.split = split or 'custom'
        self.augment = (split == 'train') if augment is None else augment

        raw_full_images: list[np.ndarray] = []
        meta: list[tuple[int, np.ndarray, int, int]] = []

        for country in COUNTRIES:
            _npz_dir = COUNTRY_NPZ_DIRS.get(country, npz_dir)
            _csv_dir = COUNTRY_CSV_DIRS.get(country, csv_dir / country)

            npz_path = _npz_dir / f'{country}_smap_monthly.npz'
            if not npz_path.exists():
                print(f'WARNING: {npz_path.name} not found — skipping.')
                continue

            npz = np.load(npz_path, allow_pickle=True)
            smap_cube = npz['smap_cube']
            chirps_cube = npz['chirps_cube']
            region_mask = npz['region_mask']
            region_names = list(npz['region_names'])
            dates = npz['dates']
            R = len(region_names)

            labels_tr = load_labels(_csv_dir, region_names, dates, label_col)

            for t, date_str in enumerate(dates):
                ts = pd.Timestamp(str(date_str))
                if not (start <= ts <= end):
                    continue

                image = np.concatenate(
                    [smap_cube[t], chirps_cube[t][np.newaxis]], axis=0
                ).astype(np.float32)

                img_idx = len(raw_full_images)
                raw_full_images.append(image)

                for r in range(R):
                    label = int(labels_tr[t, r])
                    if label == -100:
                        continue
                    meta.append((img_idx, region_mask[r], label, ts.month))

        if not raw_full_images:
            raise RuntimeError(f'No samples found for range {start.date()}..{end.date()}.')
        if not meta:
            raise RuntimeError(f'No labelled region-months for range {start.date()}..{end.date()}.')

        raw_stack = np.stack(raw_full_images, axis=0)
        if channel_mean is None:
            self.channel_mean = np.nanmean(raw_stack, axis=(0, 2, 3)).astype(np.float32)
            self.channel_std = np.nanstd(raw_stack, axis=(0, 2, 3)).astype(np.float32)
            self.channel_std = np.where(self.channel_std < 1e-6, 1.0, self.channel_std).astype(np.float32)
        else:
            self.channel_mean = channel_mean.astype(np.float32)
            self.channel_std = channel_std.astype(np.float32)

        mean = self.channel_mean[:, None, None]
        std = self.channel_std[:, None, None]

        norm_images: list[np.ndarray] = []
        for img in raw_full_images:
            img = np.where(np.isnan(img), mean, img)
            img = (img - mean) / std
            norm_images.append(img.astype(np.float32))

        self._images: list[np.ndarray] = []
        self._labels: list[int] = []
        self._months: list[int] = []

        for img_idx, rmask, label, month in meta:
            bbox = _bbox(rmask, CROP_MARGIN)
            if bbox is None:
                continue
            r0, r1, c0, c1 = bbox
            img_crop = norm_images[img_idx][:, r0:r1, c0:c1]
            mask_crop = rmask[r0:r1, c0:c1].astype(np.float32)[np.newaxis]
            if img_crop.shape[1] != CROP_SIZE or img_crop.shape[2] != CROP_SIZE:
                img_crop = _resize(img_crop, CROP_SIZE)
                mask_crop = _resize(mask_crop, CROP_SIZE)
            self._images.append(np.concatenate([img_crop, mask_crop], axis=0))
            self._labels.append(label)
            self._months.append(month)

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int):
        img = self._images[idx].copy()

        if self.augment:
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=-1).copy()
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=-2).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                img = np.rot90(img, k=k, axes=(-2, -1)).copy()

        return (
            torch.from_numpy(img),
            torch.tensor(self._labels[idx], dtype=torch.int64),
            torch.tensor(self._months[idx], dtype=torch.int64),
        )


def _make_train_loader(train_ds: DroughtDataset,
                       batch_size: int,
                       num_workers: int,
                       sampler_power: float) -> DataLoader:
    labels_arr = np.array(train_ds._labels)
    n_drought = int((labels_arr == 1).sum())
    n_normal = int((labels_arr == 0).sum())
    raw_drought_share = n_drought / max(len(train_ds), 1)

    if sampler_power > 0:
        imbalance_ratio = n_normal / max(n_drought, 1)
        w_drought = imbalance_ratio ** sampler_power
        w_normal = 1.0
        sample_weights = torch.tensor([
            w_drought if lbl == 1 else w_normal
            for lbl in train_ds._labels
        ], dtype=torch.float32)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        effective_drought_share = (
            n_drought * w_drought /
            max(n_normal * w_normal + n_drought * w_drought, 1e-12)
        )
        print(f'Sampler — drought samples: {n_drought}  '
              f'normal-only samples: {n_normal}  '
              f'raw drought share: {raw_drought_share:.2%}  '
              f'drought oversample weight: {w_drought:.2f}x  '
              f'effective drought share: {effective_drought_share:.2%}')
        return DataLoader(train_ds, batch_size=batch_size,
                          sampler=sampler, num_workers=num_workers,
                          pin_memory=True)

    print(f'Sampler — disabled. drought samples: {n_drought}  '
          f'normal-only samples: {n_normal}  '
          f'raw drought share: {raw_drought_share:.2%}')
    return DataLoader(train_ds, batch_size=batch_size,
                      shuffle=True, num_workers=num_workers,
                      pin_memory=True)


def make_dataloaders_for_ranges(
    npz_dir: Path,
    csv_dir: Path,
    train_range: tuple[str, str],
    val_range: tuple[str, str],
    test_range: Optional[tuple[str, str]] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    label_col: str = 'drought_class_spi3',
    sampler_power: float = 1.0,
):
    train_ds = DroughtDataset(npz_dir, csv_dir,
                              date_range=train_range,
                              augment=True,
                              label_col=label_col)
    val_ds = DroughtDataset(npz_dir, csv_dir,
                            date_range=val_range,
                            augment=False,
                            channel_mean=train_ds.channel_mean,
                            channel_std=train_ds.channel_std,
                            label_col=label_col)
    test_ds = None
    if test_range is not None:
        test_ds = DroughtDataset(npz_dir, csv_dir,
                                 date_range=test_range,
                                 augment=False,
                                 channel_mean=train_ds.channel_mean,
                                 channel_std=train_ds.channel_std,
                                 label_col=label_col)

    train_loader = _make_train_loader(train_ds, batch_size, num_workers, sampler_power)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers,
                                 pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.channel_mean, train_ds.channel_std


def make_dataloaders(
    npz_dir: Path,
    csv_dir: Path,
    batch_size: int = 16,
    num_workers: int = 0,
    label_col: str = 'drought_class_spi3',
    sampler_power: float = 1.0,
):
    return make_dataloaders_for_ranges(
        npz_dir=npz_dir,
        csv_dir=csv_dir,
        train_range=SPLIT_DATES['train'],
        val_range=SPLIT_DATES['val'],
        test_range=SPLIT_DATES['test'],
        batch_size=batch_size,
        num_workers=num_workers,
        label_col=label_col,
        sampler_power=sampler_power,
    )