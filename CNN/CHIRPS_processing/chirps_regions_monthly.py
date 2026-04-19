"""
Extract area-weighted monthly mean precipitation for agro-ecological regions
across the Horn of Africa from CHIRPS v3.0 Africa monthly GeoTIFFs

Countries: Somalia, Eritrea, Djibouti, Kenya, Sudan, South Sudan.
Regions are agro-ecological zones (CNN/regions.md)

- Each approximated by dissolving the corresponding GADM admin-1 boundaries.

GADM level-1 files are expected under  CNN/gadm/<country>/

Each region produces:
  CNN/CHIRPS_processing/<country>/<region_slug>_chirps_monthly.csv
  with columns:
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
CNN_DIR = os.path.dirname(SCRIPT_DIR)                        # CNN/
ROOT_DIR = os.path.dirname(CNN_DIR)                          # project root
TIF_DIR = os.path.join(ROOT_DIR, "reference_data")
GADM_DIR = os.path.join(CNN_DIR, "gadm")

NODATA = -9999.0

# ── country / region definitions ─────────────────────────────────────────────
# Each country maps to:
#   gadm_file : path relative to CNN/ for the GADM level-1 JSON
#   regions   : dict  { region_slug : [list of NAME_1 values to dissolve] }

COUNTRIES = {
    "somalia": {
        "gadm_file": os.path.join(GADM_DIR, "somalia", "gadm41_SOM_1.json"),
        "regions": {
            "northwest_agropastoral":        ["Awdal", "WoqooyiGalbeed", "Togdheer"],
            "northeast_dry_pastoral":        ["Sanaag", "Sool", "Bari", "Nugaal"],
            "central_pastoral":              ["Mudug", "Galguduud", "Hiiraan"],
            "juba_riverine":                 ["JubbadaDhexe", "JubbadaHoose", "Gedo"],
            "shabelle_riverine":             ["ShabeellahaDhexe", "ShabeellahaHoose", "Banaadir"],
            "southern_rainfed_agropastoral": ["Bay", "Bakool"],
        },
    },
    "eritrea": {
        "gadm_file": os.path.join(GADM_DIR, "eritrea", "gadm41_ERI_1.json"),
        "regions": {
            "central_highlands":    ["Maekel"],
            "western_lowlands":     ["GashBarka"],
            "eastern_lowlands":     ["SemenawiKeyihBahri", "DebubawiKeyihBahri"],
            "southwestern_lowlands":["Anseba"],
            "escarpment_green_belt":["Debub"],
        },
    },
    "djibouti": {
        "gadm_file": os.path.join(GADM_DIR, "djibouti", "gadm41_DJI_1.json"),
        "regions": {
            "coastal_arid":            ["Djiboutii", "Obock", "Tadjoura"],
            "inland_pastoral_drylands":["Dikhil", "AliSabieh"],
            "oasis_wadi_irrigated":    ["Arta"],
        },
    },
    "kenya": {
        "gadm_file": os.path.join(GADM_DIR, "kenya", "gadm41_KEN_1.json"),
        "regions": {
            "northern_arid_pastoral": [
                "Turkana", "Marsabit", "Mandera", "Wajir", "Samburu", "Isiolo",
            ],
            "eastern_semiarid": [
                "Garissa", "TanaRiver", "Kitui", "Makueni", "Machakos",
                "Embu", "Tharaka-Nithi", "Meru",
            ],
            "central_highlands": [
                "Nyeri", "Kirinyaga", "Murang'a", "Kiambu",
                "Nyandarua", "Nairobi", "Laikipia",
            ],
            "rift_valley_highlands": [
                "Nakuru", "Narok", "Kajiado", "Baringo", "Elgeyo-Marakwet",
                "UasinGishu", "Nandi", "Kericho", "Bomet",
                "TransNzoia", "WestPokot",
            ],
            "western_high_rainfall": [
                "Kakamega", "Bungoma", "Busia", "Vihiga", "Siaya",
                "Kisumu", "HomaBay", "Migori", "Kisii", "Nyamira",
            ],
            "coastal_lowlands": [
                "Mombasa", "Kilifi", "Kwale", "Lamu", "TaitaTaveta",
            ],
        },
    },
    "sudan": {
        "gadm_file": os.path.join(GADM_DIR, "sudan", "gadm41_SDN_1.json"),
        "regions": {
            "desert":                 ["Northern", "RiverNile"],
            "semi_desert":            ["RedSea", "Kassala", "Khartoum"],
            "low_rainfall_savanna":   ["NorthKurdufan", "NorthDarfur", "AlQadarif"],
            "high_rainfall_savanna":  [
                "SouthKurdufan", "SouthDarfur", "CentralDarfur",
                "EastDarfur", "WestDarfur", "WestKurdufan",
            ],
            "gezira_irrigated_nile":  ["AlJazirah", "WhiteNile", "Sennar"],
            "blue_nile_rainfed":      ["BlueNile"],
        },
    },
    "south_sudan": {
        "gadm_file": os.path.join(GADM_DIR, "south_sudan", "gadm41_SSD_1.json"),
        "regions": {
            "greenbelt":                ["WestEquatoria", "CentralEquatoria"],
            "ironstone_plateau":        ["WestBahr-al-Ghazal"],
            "flood_plains":             ["Jungoli", "Unity", "Lakes"],
            "nile_sobat_river":         ["UpperNile"],
            "eastern_pastoral_drylands":["EasternEquatoria"],
            "hills_and_mountains":      ["NorthBahr-al-Ghazal", "Warap"],
        },
    },
}

# ── helpers ──────────────────────────────────────────────────────────────────

def load_gadm(gadm_file):
    """Load a GADM level-1 GeoJSON."""
    if not os.path.exists(gadm_file):
        raise FileNotFoundError(f"GADM file not found: {gadm_file}")
    return gpd.read_file(gadm_file)


def get_region_boundary(gadm, slug, name1_values, cache_dir):
    """Dissolve NAME_1 regions into one boundary. Cache as GeoJSON."""
    cache = os.path.join(cache_dir, f"{slug}_boundary.geojson")
    if os.path.exists(cache):
        return gpd.read_file(cache)

    subset = gadm[gadm["NAME_1"].isin(name1_values)].copy()
    if subset.empty:
        raise ValueError(
            f"No features found for NAME_1 in {name1_values}. "
            f"Available: {sorted(gadm['NAME_1'].unique())}"
        )

    dissolved = subset.dissolve()
    os.makedirs(cache_dir, exist_ok=True)
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
    tifs = sorted(glob.glob(os.path.join(TIF_DIR, "chirps-v3.0.*.tif")))
    print(f"Found {len(tifs)} CHIRPS v3.0 tifs\n")

    for country, cfg in COUNTRIES.items():
        print(f"\n{'═' * 60}")
        print(f"  {country.upper()}")
        print(f"{'═' * 60}")

        gadm = load_gadm(cfg["gadm_file"])
        out_dir = os.path.join(SCRIPT_DIR, country)
        os.makedirs(out_dir, exist_ok=True)

        # cache boundaries next to the GADM source file
        cache_dir = os.path.dirname(cfg["gadm_file"])

        for slug, name1_values in cfg["regions"].items():
            print(f"\n─── {slug}  ← {name1_values} ───")
            boundary = get_region_boundary(gadm, slug, name1_values, cache_dir)
            geom = boundary.geometry.values

            df = extract_region(tifs, geom, slug)

            out_csv = os.path.join(out_dir, f"{slug}_chirps_monthly.csv")
            df.to_csv(out_csv, index=False)
            print(f"  Saved {len(df)} rows → {out_csv}")

            for scale in (3, 6):
                cls = f"drought_class_spi{scale}"
                valid = df[cls].dropna()
                drought = (valid >= 1).sum()
                print(f"  SPI-{scale}: {drought}/{len(valid)} drought months "
                      f"({100*drought/len(valid):.1f}%)")

    print("\nDone — all countries / regions extracted.")


if __name__ == "__main__":
    main()
