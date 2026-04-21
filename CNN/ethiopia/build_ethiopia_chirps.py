"""
CHIRPS monthly precipitation CSVs for 5 Ethiopian regions.
Steps: load GADM level-1 / pre-built boundaries → clip CHIRPS TIFs
       → cosine-weighted mean → compute SPI-3 / SPI-6 → save CSV per region.
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


SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CNN_DIR     = os.path.dirname(SCRIPT_DIR)
ROOT_DIR    = os.path.dirname(CNN_DIR)

TIF_DIR             = os.path.join(ROOT_DIR, "reference_data")
GADM_ETH1           = os.path.join(CNN_DIR, "gadm", "ethiopia", "gadm41_ETH_1.json")
WEST_ARSI_BOUNDARY  = os.path.join(ROOT_DIR, "reference_data",
                                   "reference_data_preprocessing",
                                   "west_arsi_boundary.geojson")
NODATA = -9999.0

REGIONS = {
    "afar":      ["Afar"],
    "amhara":    ["Amhara"],
    "oromia":    ["Oromia"],
    "somali":    ["Somali"],
    "west_arsi": None,
}

def load_region_boundary(slug, name1_values):
    if name1_values is None:
        if not os.path.exists(WEST_ARSI_BOUNDARY):
            raise FileNotFoundError(
                f"West Arsi boundary not found: {WEST_ARSI_BOUNDARY}"
            )
        return gpd.read_file(WEST_ARSI_BOUNDARY)

    gadm = gpd.read_file(GADM_ETH1)
    subset = gadm[gadm["NAME_1"].isin(name1_values)].copy()
    if subset.empty:
        raise ValueError(
            f"No features found for NAME_1={name1_values}. "
            f"Available: {sorted(gadm['NAME_1'].unique())}"
        )
    return subset.dissolve()


def parse_date(filename):
    m = re.search(r"chirps-v3\.0\.(\d{4})\.(\d{2})\.tif", filename)
    if m is None:
        return None
    return f"{m.group(1)}-{m.group(2)}-01"


def cosine_weighted_mean(data, transform):
    nrows, _ = data.shape
    lats = transform.f + (np.arange(nrows) + 0.5) * transform.e
    weights = np.cos(np.deg2rad(lats))[:, np.newaxis]
    weights = np.broadcast_to(weights, data.shape)

    valid = ~data.mask if np.ma.is_masked(data) else np.ones_like(data, dtype=bool)
    if not valid.any():
        return np.nan

    w = weights[valid]
    v = np.asarray(data)[valid]
    return float(np.sum(w * v) / np.sum(w))


def compute_spi(df):
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

        cls_col = f"drought_class_spi{scale}"
        df[cls_col] = np.where(
            df[col_spi].isna(), np.nan,
            np.where(df[col_spi] <= -1.5, 2,          # severe + extreme
            np.where(df[col_spi] <= -1.0, 1, 0))
        )
        df[cls_col] = df[cls_col].astype("Int64")

    return df


def extract_region(tifs, geom, slug):
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



def main():
    tifs = sorted(glob.glob(os.path.join(TIF_DIR, "chirps-v3.0.*.tif")))
    print(f"Found {len(tifs)} CHIRPS v3.0 TIFs\n")

    os.makedirs(SCRIPT_DIR, exist_ok=True)

    for slug, name1_values in REGIONS.items():
        print(f"\n{'═' * 50}")
        print(f"  ETHIOPIA — {slug.upper()}")
        print(f"{'═' * 50}")

        boundary = load_region_boundary(slug, name1_values)
        geom = boundary.geometry.values

        df = extract_region(tifs, geom, slug)

        out_csv = os.path.join(SCRIPT_DIR, f"{slug}_chirps_monthly.csv")
        df.to_csv(out_csv, index=False)
        print(f"  Saved {len(df)} rows → {out_csv}")

        for scale in (3, 6):
            cls = f"drought_class_spi{scale}"
            valid = df[cls].dropna()
            drought = (valid >= 1).sum()
            print(f"  SPI-{scale}: {drought}/{len(valid)} drought months "
                  f"({100*drought/len(valid):.1f}%)")

    print("\nDone — all Ethiopia regions extracted.")


if __name__ == "__main__":
    main()
