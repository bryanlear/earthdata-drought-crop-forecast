"""
Extract area-weighted monthly mean precipitation for West Arsi zone
from CHIRPS v3.0 Africa monthly GeoTIFFs

Steps:
  1. Download the West Arsi (MirabArsi) admin-2 boundary from GADM 4.1
  2. For each CHIRPS .tif, mask/clip to the polygon
  3. Compute cos(lat)-weighted mean over valid pixels
  4. Save CSV: date, precip_mm

Output
------
  reference_data_preprocessing/west_arsi_chirps_monthly.csv
"""

import os
import re
import glob

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from scipy.stats import gamma as gamma_dist, norm

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
TIF_DIR = os.path.join(PROJECT_DIR, "reference_data")
OUT_CSV = os.path.join(SCRIPT_DIR, "west_arsi_chirps_monthly.csv")
BOUNDARY_CACHE = os.path.join(SCRIPT_DIR, "west_arsi_boundary.geojson")

NODATA = -9999.0
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ETH_2.json.zip"
ZONE_NAME = "MirabArsi"  # GADM NAME_2 for West Arsi

# ── helpers ──────────────────────────────────────────────────────────────────

def get_boundary():
    """Return West Arsi polygon as a GeoDataFrame. Cache locally as GeoJSON."""
    if os.path.exists(BOUNDARY_CACHE):
        print(f"Loading cached boundary: {BOUNDARY_CACHE}")
        return gpd.read_file(BOUNDARY_CACHE)

    print(f"Downloading GADM admin-2 for Ethiopia …")
    gdf = gpd.read_file(GADM_URL)
    wa = gdf[gdf["NAME_2"] == ZONE_NAME].copy()
    if wa.empty:
        raise ValueError(f"Zone '{ZONE_NAME}' not found in GADM data")
    wa.to_file(BOUNDARY_CACHE, driver="GeoJSON")
    print(f"Boundary cached → {BOUNDARY_CACHE}")
    return wa


def parse_date(filename):
    """Extract (year, month) from a CHIRPS filename and return a date string."""
    m = re.search(r"chirps-v3\.0\.(\d{4})\.(\d{2})\.tif", filename)
    if m is None:
        return None
    return f"{m.group(1)}-{m.group(2)}-01"


def cosine_weighted_mean(data, transform):
    """
    Compute the cos(lat)-weighted mean of a 2-D masked array.
    `transform` is the affine transform of the clipped raster.
    """
    nrows, ncols = data.shape
    # centre latitude of each row
    row_indices = np.arange(nrows)
    lats = transform.f + (row_indices + 0.5) * transform.e  # transform.e < 0

    weights = np.cos(np.deg2rad(lats))[:, np.newaxis]  # (nrows, 1)
    weights = np.broadcast_to(weights, data.shape)

    valid = ~data.mask if np.ma.is_masked(data) else np.ones_like(data, dtype=bool)
    if not valid.any():
        return np.nan

    w = weights[valid]
    v = data[valid]
    return float(np.sum(w * v) / np.sum(w))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    boundary = get_boundary()
    geom = boundary.geometry.values

    tifs = sorted(glob.glob(os.path.join(TIF_DIR, "chirps-v3.0.*.tif")))
    print(f"Found {len(tifs)} CHIRPS v3.0 tifs")

    records = []
    for i, tif_path in enumerate(tifs, 1):
        fname = os.path.basename(tif_path)
        date_str = parse_date(fname)
        if date_str is None:
            continue

        with rasterio.open(tif_path) as src:
            clipped, clipped_transform = rio_mask(
                src, geom, crop=True, nodata=NODATA, all_touched=True
            )

        arr = clipped[0]  # single band
        masked = np.ma.masked_equal(arr, NODATA)
        precip = cosine_weighted_mean(masked, clipped_transform)

        records.append({"date": date_str, "precip_mm": round(precip, 4)})

        if i % 50 == 0 or i == len(tifs):
            print(f"  [{i}/{len(tifs)}] {fname}  →  {precip:.2f} mm")

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ── compute SPI-3 and SPI-6 ─────────────────────────────────────────────
    for scale in (3, 6):
        col_acc = f"precip_{scale}m"
        col_spi = f"spi_{scale}"

        # rolling accumulation
        df[col_acc] = df["precip_mm"].rolling(window=scale, min_periods=scale).sum()

        # standardise per calendar month using gamma CDF → normal PPF
        df[col_spi] = np.nan
        df["_month"] = df["date"].dt.month
        for m in range(1, 13):
            mask = (df["_month"] == m) & df[col_acc].notna()
            vals = df.loc[mask, col_acc].values
            if len(vals) < 10:
                continue

            # fraction of zero-accumulation months
            q = np.sum(vals == 0) / len(vals)
            pos = vals[vals > 0]

            if len(pos) < 2:
                continue

            # fit gamma to positive values
            a, loc, scale_param = gamma_dist.fit(pos, floc=0)

            # CDF: probability of zero mass + gamma CDF for positives
            cdf = np.where(
                vals == 0,
                q,
                q + (1 - q) * gamma_dist.cdf(vals, a, loc=loc, scale=scale_param),
            )
            # clamp to avoid ±inf from ppf
            cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
            df.loc[mask, col_spi] = norm.ppf(cdf)

        df.drop(columns=[col_acc, "_month"], inplace=True)

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows → {OUT_CSV}")
    print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
