"""
Per-region masked SMAP finetuning dataset.

For each Ethiopian region (x6) and each of the 132 overlapping months --> produce one training sample:

    x  = 8-channel SMAP image (44×43) with pixels OUTSIDE the region set to 0
    y  = binary drought label (0 = no drought, 1 = drought) from SPI-3
        where drought = SPI-3 ≤ -1.0 (merges former moderate + severe classes)

Output: ethiopia_smap_finetuning_dataset.npz
    images          — (792, 8, 44, 43)  float32   NaN-free, region-masked
    labels_spi3     — (792,)            int8       binary drought from SPI-3
    labels_spi6     — (792,)            int8       binary drought from SPI-6
    spi3_values     — (792,)            float32    raw SPI-3 for regression
    spi6_values     — (792,)            float32    raw SPI-6 for regression
    dates           — (792,)            str        ISO dates
    regions         — (792,)            str        region slug
    region_masks    — (6, 44, 43)       bool       spatial masks (for reference)
    region_names    — (6,)              str        region ordering
    feature_names   — (8,)              str        channel names
    lat_grid        — (44, 43)          float32    EASE-Grid latitudes
    lon_grid        — (44, 43)          float32    EASE-Grid longitudes

Strategy:
  1. Rasterize each GADM boundary onto the 44×43 EASE-Grid via point-in-polygon 
  (https://en.wikipedia.org/wiki/Point_in_polygon)
  2. For each (x_{r,t} - region, month), mask SMAP cube, fill remaining NaN with 0
  3. Pair with corresponding drought label from the CHIRPS CSV
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
BOUNDARY_DIR = PROJECT_DIR / 'reference_data_preprocessing'
NPZ_PATH = SCRIPT_DIR / 'ethiopia_smap_monthly.npz'
OUT_PATH = SCRIPT_DIR / 'ethiopia_smap_finetuning_dataset.npz'

REGIONS = ['afar', 'amhara', 'oromia', 'snnpr', 'somali', 'west_arsi']


def build_region_mask(lat_grid, lon_grid, geojson_path):
    """Rasterize a region boundary onto the EASE-Grid via point-in-polygon.

    Returns bool array (H, W) — True where the grid-cell centre falls inside
    the region polygon.
    """
    gdf = gpd.read_file(geojson_path)
    geom = gdf.union_all()
    H, W = lat_grid.shape
    mask = np.zeros((H, W), dtype=bool)
    for i in range(H):
        for j in range(W):
            if not np.isnan(lat_grid[i, j]):
                mask[i, j] = geom.contains(Point(lon_grid[i, j], lat_grid[i, j]))
    return mask


def load_labels(region_slug, target_dates):
    """Load CHIRPS CSV and return labels aligned to target_dates."""
    csv_path = BOUNDARY_DIR / f'{region_slug}_chirps_monthly.csv'
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.set_index('date').loc[target_dates]
    return df


def main():
    print('=== Building per-region masked finetuning dataset ===\n')

    # Load SMAP cube
    data = np.load(NPZ_PATH)
    cube = data['cube']              # (132, 8, 44, 43)
    dates = pd.to_datetime(data['dates'])
    feature_names = data['feature_names']
    lat_grid = data['lat_grid']
    lon_grid = data['lon_grid']
    T, C, H, W = cube.shape
    print(f'SMAP cube: {cube.shape}  ({T} months, {C} channels, {H}×{W})')

    # ── Step 1: Build region masks ───────────────────────────────────────
    print('\nRasterizing region boundaries ...')
    region_masks = np.zeros((len(REGIONS), H, W), dtype=bool)
    for r, slug in enumerate(REGIONS):
        geojson = BOUNDARY_DIR / f'{slug}_boundary.geojson'
        region_masks[r] = build_region_mask(lat_grid, lon_grid, geojson)
        n_pixels = region_masks[r].sum()
        print(f'  {slug:12s}: {n_pixels:4d} pixels '
              f'({100 * n_pixels / (H * W):.1f}% of grid)')

    # ── Step 2: Assemble samples ─────────────────────────────────────────
    N = T * len(REGIONS)  # 132 × 6 = 792
    print(f'\nAssembling {N} samples ({T} months × {len(REGIONS)} regions) ...')

    images = np.zeros((N, C, H, W), dtype=np.float32)
    labels_spi3 = np.zeros(N, dtype=np.int8)
    labels_spi6 = np.zeros(N, dtype=np.int8)
    spi3_values = np.zeros(N, dtype=np.float32)
    spi6_values = np.zeros(N, dtype=np.float32)
    sample_dates = []
    sample_regions = []

    idx = 0
    for r, slug in enumerate(REGIONS):
        mask = region_masks[r]  # (H, W) bool
        df = load_labels(slug, dates)

        for t in range(T):
            # Apply spatial mask: zero out pixels outside region
            img = cube[t].copy()           # (C, H, W)
            img[:, ~mask] = 0.0            # outside-region → 0
            np.nan_to_num(img, copy=False)  # remaining NaN → 0 (within-region gaps)
            images[idx] = img

            row = df.iloc[t]
            # Binary: 0 = no drought (SPI > -1.0), 1 = drought (SPI ≤ -1.0)
            labels_spi3[idx] = int(row['drought_class_spi3'] >= 1)
            labels_spi6[idx] = int(row['drought_class_spi6'] >= 1)
            spi3_values[idx] = float(row['spi_3'])
            spi6_values[idx] = float(row['spi_6'])

            sample_dates.append(dates[t].strftime('%Y-%m-%d'))
            sample_regions.append(slug)
            idx += 1

        n_no  = (df['drought_class_spi3'] == 0).sum()
        n_yes = (df['drought_class_spi3'] >= 1).sum()
        print(f'  {slug:12s}: {T} samples, '
              f'no_drought/drought = {n_no}/{n_yes}')

    # ── Step 3: Summary stats ────────────────────────────────────────────
    print(f'\n--- Dataset summary ---')
    print(f'Images:  {images.shape}  dtype={images.dtype}')
    print(f'Labels:  {labels_spi3.shape}  classes={np.unique(labels_spi3)}')

    total = len(labels_spi3)

        n = (labels_spi3 == cls).sum()
        print(f'  class {cls} ({name:11s}): {n:4d}  ({100 * n / total:.1f}%)')

    # Per-channel stats (across all samples, in-region pixels only)
    all_masks = np.zeros((N, H, W), dtype=bool)
    for i in range(N):
        r_idx = i // T
        all_masks[i] = region_masks[r_idx]

    for ch in range(C):
        vals = images[:, ch][all_masks]
        print(f'  {feature_names[ch]:25s}: '
              f'mean={vals.mean():.4f}  std={vals.std():.4f}  '
              f'min={vals.min():.4f}  max={vals.max():.4f}')

    # ── Step 4: Save ─────────────────────────────────────────────────────
    np.savez_compressed(
        OUT_PATH,
        images=images,
        labels_spi3=labels_spi3,
        labels_spi6=labels_spi6,
        spi3_values=spi3_values,
        spi6_values=spi6_values,
        dates=np.array(sample_dates),
        regions=np.array(sample_regions),
        region_masks=region_masks,
        region_names=np.array(REGIONS),
        feature_names=feature_names,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
    )
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f'\nSaved {OUT_PATH.name} ({size_mb:.1f} MB)')

    # Quick verification
    v = np.load(OUT_PATH)
    print(f'Verified: images={v["images"].shape}, labels={v["labels_spi3"].shape}, '
          f'regions={np.unique(v["regions"])}, dates={v["dates"][0]}..{v["dates"][-1]}')


if __name__ == '__main__':
    main()
