import re
import random
from glob import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import rasterio
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent
tifs = sorted(glob(str(DATA_DIR / 'chirps-v3.0.*.tif')))

by_year = defaultdict(list)
for f in tifs:
    m = re.search(r'chirps-v3\.0\.(\d{4})\.(\d{2})\.tif', f)
    if m:
        by_year[int(m.group(1))].append((int(m.group(2)), f))

years = [1985, 2008, 2026]
# pick a month available in all three years
common_months = sorted(
    set.intersection(*(set(m for m, _ in by_year[y]) for y in years))
)
month = random.choice(common_months)

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

for ax, year in zip(axes, years):
    fpath = next(f for m, f in by_year[year] if m == month)
    with rasterio.open(fpath) as src:
        data = src.read(1)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    data = np.where(data < 0, np.nan, data)
    im = ax.imshow(data, extent=extent, cmap='YlGnBu', vmin=0, vmax=300)
    ax.set_title(f'{year}-{month:02d}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

fig.suptitle('CHIRPS Monthly Precipitation — Africa (random samples)', fontsize=14)
plt.tight_layout(rect=[0, 0, 0.88, 0.95])
fig.colorbar(im, ax=axes, label='Precipitation (mm)', shrink=0.6, pad=0.04)
plt.savefig(DATA_DIR / 'chirps_random_samples.png', dpi=150, bbox_inches='tight')
plt.show()
