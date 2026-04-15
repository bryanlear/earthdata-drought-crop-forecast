"""
Extract area-weighted monthly mean precipitation for multiple regions in Ethiopia
from CHIRPS v3.0 Africa monthly GeoTIFFs

Regions extracted (GADM admin level 1, dissolved):
  - West Arsi  (admin-2 zone within Oromia)
  - Oromia
  - Afar
  - Somali
  - Amhara
  - SNNPR (SouthernNations,Nationalities)

Each region produces:
  <slug>_chirps_monthly.csv  with columns:
    date, precip_mm, spi_3, spi_6, drought_class_spi3, drought_class_spi6
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
GADM_L2 = os.path.join(SCRIPT_DIR, "gadm41_ETH_2.json")

NODATA = -9999.0
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ETH_2.json.zip"

# ── regions to extract ───────────────────────────────────────────────────────
# Each entry:  (slug, level, field_value)
#   level="NAME_1" → dissolve all zones in that region
#   level="NAME_2" → single admin-2 zone
REGIONS = [
    ("west_arsi",  "NAME_2", "MirabArsi"),
    ("oromia",     "NAME_1", "Oromia"),
    ("afar",       "NAME_1", "Afar"),
    ("somali",     "NAME_1", "Somali"),
    ("amhara",     "NAME_1", "Amhara"),
    ("snnpr",      "NAME_1", "SouthernNations,Nationalities"),
]

# ── helpers ──────────────────────────────────────────────────────────────────

def load_gadm():
    """Load GADM level-2 data (from local file or download)."""
    if os.path.exists(GADM_L2):
        return gpd.read_file(GADM_L2)
    print("Downloading GADM admin-2 for Ethiopia …")
    gdf = gpd.read_file(GADM_URL)
    gdf.to_file(GADM_L2, driver="GeoJSON")
    return gdf


def get_region_boundary(gadm, slug, field, value):
    """Extract and dissolve a region boundary. Cache as GeoJSON."""
    cache = os.path.join(SCRIPT_DIR, f"{slug}_boundary.geojson")
    if os.path.exists(cache):
        return gpd.read_file(cache)

    subset = gadm[gadm[field] == value].copy()
    if subset.empty:
        raise ValueError(f"No features found where {field} == '{value}'")

    # dissolve all sub-zones into one polygon
    dissolved = subset.dissolve()
    dissolved.to_file(cache, driver="GeoJSON")
    print(f"  Boundary cached → {cache}")
    return dissolved


def parse_date(filename):
    m = re.search(r"chirps-v3\.0\.(\d{4})\.(\d{2})\.tif", filename)
    if m is None:
        return None
    return f"{m.group(1)}-{m.group(2)}-01"


def cosine_weighted_mean(data, transform):
    """Cos(lat)-weighted mean of a 2-D masked array."""
    nrows, ncols = data.shape
    row_indices = np.arange(nrows)
    lats = transform.f + (row_indices + 0.5) * transform.e

    weights = np.cos(np.deg2rad(lats))[:, np.newaxis]
    weights = np.broadcast_to(weights, data.shape)

    valid = ~data.mask if np.ma.is_masked(data) else np.ones_like(data, dtype=bool)
    if not valid.any():
        return np.nan

    w = weights[valid]
    v = data[valid]
    return float(np.sum(w * v) / np.sum(w))


def compute_spi(df):
    """Add SPI-3, SPI-6, and drought class columns in-place."""
    for scale in (3, 6):
        col_acc = f"_acc_{scale}"
        col_spi = f"spi_{scale}"

        df[col_acc] = df["precip_mm"].rolling(window=scale, min_periods=scale).sum()
        df[col_spi] = np.nan
        month_col = df["date"].dt.month

        for m in range(1, 13):
            mask = (month_col == m) & df[col_acc].notna()
            vals = df.loc[mask, col_acc].values
            if len(vals) < 10:
                continue

            q = np.sum(vals == 0) / len(vals)
            pos = vals[vals > 0]
            if len(pos) < 2:
                continue

            a, loc, scale_param = gamma_dist.fit(pos, floc=0)
            cdf = np.where(
                vals == 0, q,
                q + (1 - q) * gamma_dist.cdf(vals, a, loc=loc, scale=scale_param),
            )
            cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
            df.loc[mask, col_spi] = norm.ppf(cdf)

        df.drop(columns=[col_acc], inplace=True)

        # drought classification (0=no drought, 1=moderate, 2=severe/extreme)
        cls_col = f"drought_class_spi{scale}"
        df[cls_col] = np.where(
            df[col_spi].isna(), np.nan,
            np.where(df[col_spi] <= -1.5, 2,         # severe + extreme
            np.where(df[col_spi] <= -1.0, 1, 0))     # moderate / no drought
        )
        df[cls_col] = df[cls_col].astype("Int64")

    return df


def extract_region(tifs, geom, label):
    """Clip all tifs to geom and return a DataFrame."""
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

        arr = clipped[0]
        masked = np.ma.masked_equal(arr, NODATA)
        precip = cosine_weighted_mean(masked, clipped_transform)
        records.append({"date": date_str, "precip_mm": round(precip, 4)})

        if i % 100 == 0 or i == len(tifs):
            print(f"    [{i}/{len(tifs)}] {precip:.2f} mm")

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return compute_spi(df)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    gadm = load_gadm()
    tifs = sorted(glob.glob(os.path.join(TIF_DIR, "chirps-v3.0.*.tif")))
    print(f"Found {len(tifs)} CHIRPS v3.0 tifs\n")

    for slug, field, value in REGIONS:
        print(f"═══ {value} ({slug}) ═══")
        boundary = get_region_boundary(gadm, slug, field, value)
        geom = boundary.geometry.values

        df = extract_region(tifs, geom, slug)

        out_csv = os.path.join(SCRIPT_DIR, f"{slug}_chirps_monthly.csv")
        df.to_csv(out_csv, index=False)
        print(f"  Saved {len(df)} rows → {out_csv}")

        # quick drought summary
        for scale in (3, 6):
            cls = f"drought_class_spi{scale}"
            valid = df[cls].dropna()
            drought = (valid >= 1).sum()
            print(f"  SPI-{scale}: {drought}/{len(valid)} drought months "
                  f"({100*drought/len(valid):.1f}%)")
        print()

    print("Done — all regions extracted.")


if __name__ == "__main__":
    main()
