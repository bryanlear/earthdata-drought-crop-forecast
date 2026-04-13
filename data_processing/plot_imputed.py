'''Plot imputed single-channel and multi-feature SMAP cubes'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

ARRAY_DIR = Path('/earthdata-drought-crop-forecast/3d_numpy_arrays/soil_moisture_array')
OUT_DIR = Path('/earthdata-drought-crop-forecast/data_processing/plots')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_dates(date_strings):
    return [datetime.strptime(str(d), '%Y-%m-%d') for d in date_strings]


def plot_single_channel():
    data = np.load(ARRAY_DIR / 'smap_sm_west_arsi_3day_imputed.npz')
    cube = data['sm_cube']          # (T, 64, 64)
    dates = parse_dates(data['dates'])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('SMAP Soil Moisture – West Arsi 64×64 (3-day composites, imputed)', fontsize=13)

    # --- Top: spatial mean time series ---
    spatial_mean = np.nanmean(cube, axis=(1, 2))
    ax = axes[0]
    ax.plot(dates, spatial_mean, linewidth=0.5, color='steelblue')
    ax.set_ylabel('Soil Moisture (cm³/cm³)')
    ax.set_title('Spatial-mean soil moisture over time')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(True, alpha=0.3)

    # --- Bottom: snapshot center time step ---
    mid = cube.shape[0] // 2
    im = axes[1].imshow(cube[mid], cmap='YlGnBu', origin='upper')
    axes[1].set_title(f'Spatial snapshot – {dates[mid].strftime("%Y-%m-%d")}')
    axes[1].set_xlabel('Lon pixel')
    axes[1].set_ylabel('Lat pixel')
    plt.colorbar(im, ax=axes[1], label='cm³/cm³', shrink=0.8)

    plt.tight_layout()
    out = OUT_DIR / 'sm_single_channel.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {out}')


# ── Multi plot ──────────────────────────────────────────────
def plot_multifeature():
    data = np.load(ARRAY_DIR / 'smap_multifeature_west_arsi_3day_imputed.npz')
    cube = data['cube']                     # (T, 8, 64, 64)
    dates = parse_dates(data['dates'])
    names = [str(n) for n in data['feature_names']]

    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    fig.suptitle('SMAP Multi-feature – West Arsi 64×64 (3-day composites, imputed)', fontsize=13, y=0.995)

    colours = ['#1f77b4', '#ff7f0e', '#d62728', '#e377c2',
               '#2ca02c', '#17becf', '#9467bd', '#8c564b']

    for i, ax in enumerate(axes.flat):
        ts = np.nanmean(cube[:, i, :, :], axis=(1, 2))
        ax.plot(dates, ts, linewidth=0.5, color=colours[i])
        ax.set_title(names[i], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    axes[-1, 0].set_xlabel('Date')
    axes[-1, 1].set_xlabel('Date')
    plt.tight_layout()
    out = OUT_DIR / 'sm_multifeature.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    plot_single_channel()
    plot_multifeature()
