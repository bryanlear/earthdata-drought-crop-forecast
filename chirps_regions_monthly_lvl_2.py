"""
CHIRPS monthly precipitation CSVs for 5 Horn of Africa countries (GADM level-2).
Steps: dissolve GADM admin-2 district boundaries per region → clip CHIRPS TIFs
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


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(CNN_DIR)
TIF_DIR = os.path.join(ROOT_DIR, "reference_data")
GADM_DIR = os.path.join(CNN_DIR, "gadm")

NODATA = -9999.0


COUNTRIES = {
    "somalia": {
        "gadm_file": os.path.join(GADM_DIR, "somalia", "gadm41_SOM_2.json"),
        "regions": {
            "northwest_agropastoral": [
                "Baki", "Boorama", "Lughaya", "Zeylac",
                "Berbera", "Gabiley", "Hargeysa",
                "Burao", "Buuhoodle", "Oodweyne", "Sheekh",
            ],
            "northeast_dry_pastoral": [
                "Badhan", "Ceel-Afwein", "Ceerigaabo",
                "Caynabo", "Lascaanod", "Taleex", "Xudun",
                "Bander-Beyla", "Bosaaso", "Calawla",
                "Iskushuban", "Qandala", "Qardho",
                "Burtinle", "Eyl", "Garoowe",
            ],
            "central_pastoral": [
                "Gaalkacayo", "Goldogob", "Hobyo",
                "Jariiban", "Xarardheere",
                "Caabudwaaq", "Cadaado", "CeelBuur",
                "CeelDheer", "Dhuusamareeb",
                "BeledWeyn", "BuuloBurdo", "Jalalaqsi",
            ],
            "juba_riverine": [
                "Bu'aale", "Jilib", "Saakow",
                "Afmadow", "Badhaadhe", "Jamaame", "Kismaayo",
                "Baar-Dheere", "BeledXaawo", "CeelWaaq",
                "Dolow", "Garbahaaray", "Luuk",
            ],
            "shabelle_riverine": [
                "Aadan", "Balcad", "Cadale", "Jawhar",
                "Afgooye", "Baraawe", "Kuntuwaaray",
                "Marka", "Qoryooley", "Sablale", "WanlaWeyn",
                "Mogadisho",
            ],
            "southern_rainfed_agropastoral": [
                "Baydhabo", "BuurXakaba", "Diinsoor", "QansaxDheere",
                "CeelBarde", "RabDhuure", "Tiyeeglow",
                "Wajid", "Xudur",
            ],
        },
    },
    "eritrea": {
        "gadm_file": os.path.join(GADM_DIR, "eritrea", "gadm41_ERI_2.json"),
        "regions": {
            "central_highlands": [
                "AsmaraCity", "Berikh", "GhalaNefhi", "Serejeka",
            ],
            "western_lowlands": [
                "Akordat", "Barentu", "Dghe", "Forto", "Gogne",
                "Haykota", "La`ElayGash", "LogoAnseba", "Mansura",
                "Mogolo", "Omhajer", "Shemboko", "Teseneye",
            ],
            "eastern_lowlands": [
                "Afabet", "Dahlak", "Foro", "Ghelaelo'",
                "Ghida`e", "Karora", "Mitswa`eCity", "Nakfa", "Sheib",
                "Areta'", "CentralSouthernRedSea",
                "SouthSouthernRedSea",
            ],
            "southwestern_lowlands": [
                "AdiTeklezan", "Asmat", "Elabered", "Gheleb",
                "Habero", "Hagaz", "Halhal", "Keren",
                "KerkeBet", "Sel`a",
            ],
            "escarpment_green_belt": [
                "AdiKeyih", "AdiKwala", "Areza", "Dekemehare",
                "Dibarwa", "KudoBu`er", "MayMine", "Mendefera",
                "Segeneyiti", "Senafe", "Tsorena",
            ],
        },
    },
    "djibouti": {
        "gadm_file": os.path.join(GADM_DIR, "djibouti", "gadm41_DJI_2.json"),
        "regions": {
            "coastal_arid": [
                "Djiboutii",
                "Adailou", "Dadda'to", "Khôr'Angar",
                "Moulhoule", "Obock",
                "Balho", "Dorra", "Lac'Assal",
                "MousaAli", "Randa", "Tadjoura",
            ],
            "inland_pastoral_drylands": [
                "AsEyla", "Dikhil", "Galafi", "Yuboki",
                "AliAdde", "AliSabieh", "Holhol",
            ],
            "oasis_wadi_irrigated": [
                "Arta", "Yuboki",
            ],
        },
    },
    "kenya": {
        "gadm_file": os.path.join(GADM_DIR, "kenya", "gadm41_KEN_2.json"),
        "regions": {
            "northern_arid_pastoral": [
                "Loima", "TurkanaCentral", "TurkanaEast",
                "TurkanaNorth", "TurkanaSouth", "TurkanaWest", "unknown2",
                "Laisamis", "Moyale", "NorthHorr", "Saku",
                "Banissa", "Lafey", "ManderaEast", "ManderaNorth",
                "ManderaSouth", "ManderaWest", "unknown1",
                "Eldas", "Tarbaj", "WajirEast", "WajirNorth",
                "WajirSouth", "WajirWest",
                "SamburuEast", "SamburuNorth", "SamburuWest",
                "IsioloNorth", "IsioloSouth",
            ],
            "eastern_semiarid": [
                "Balambala", "Daadab", "Fafi", "GarissaTownship",
                "Ijara", "Lagdera",
                "Bura", "Galole", "Garsen",
                "KituiCentral", "KituiEast", "KituiRural", "KituiSouth",
                "KituiWest", "MwingiCentral", "MwingiNorth", "MwingiWest",
                "Kaiti", "KibweziEast", "KibweziWest", "Kilome",
                "Makueni", "Mbooni",
                "Kangundo", "Kathiani", "MachakosTown", "Masinga",
                "Matungulu", "Mavoko", "Mwala", "Yatta", "unknown7",
                "Manyatta", "MbeereNorth", "MbeereSouth", "Runyenjes",
                "Chuka/Igambang'Ombe", "IgembeSouth", "Maara", "Tharaka",
                "Buuri", "CentralImenti", "IgembeCentral", "IgembeNorth",
                "IgembeSouth", "NorthImenti", "SouthImenti",
                "TiganiaEast", "TiganiaWest", "unknown5",
            ],
            "central_highlands": [
                "Kieni", "Mathira", "Mukurweini", "NyeriTown", "Othaya", "Tetu",
                "Gichugu", "KirinyagaCentral", "Mwea", "Ndia",
                "Gatanga", "Kandara", "Kangema", "Kigumo",
                "Kiharu", "Maragwa", "Mathioya",
                "GatunduNorth", "GatunduSouth", "Githunguri", "Juja",
                "Kabete", "Kiambaa", "Kiambu", "Kikuyu",
                "Lari", "Limuru", "Ruiru", "ThikaTown",
                "Kinangop", "Kipipiri", "Ndaragwa", "OlJorok", "OlKalou",
                "DagorettiNorth", "DagorettiSouth", "EmbakasiCentral",
                "EmbakasiEast", "EmbakasiNorth", "EmbakasiSouth",
                "EmbakasiWest", "Kamukunji", "Kasarani", "Kibra",
                "Langata", "Makadara", "Mathare", "Roysambu",
                "Ruaraka", "Starehe", "Westlands",
                "LaikipiaEast", "LaikipiaNorth", "LaikipiaWest",
            ],
            "rift_valley_highlands": [
                "Bahati", "Gilgil", "KuresoiNorth", "KuresoiSouth",
                "Molo", "Naivasha", "NakuruTownEast", "NakuruTownWest",
                "Njoro", "Rongai", "Subukia",
                "EmuruaDikirr", "Kilgoris", "NarokEast", "NarokNorth",
                "NarokSouth", "NarokWest",
                "KajiadoCentral", "KajiadoEast", "KajiadoNorth",
                "KajiadoSouth", "KajiadoWest",
                "805", "BaringoCentral", "BaringoNorth", "BaringoSouth",
                "EldamaRavine", "Mogotio", "Tiaty",
                "KeiyoNorth", "KeiyoSouth", "MarakwetEast", "MarakwetWest",
                "Ainabkoi", "Kapseret", "Kesses", "Moiben", "Soy", "Turbo",
                "Aldai", "Chesumei", "Emgwen", "Mosop", "NandiHills", "Tinderet",
                "Ainamoi", "Belgut", "Bureti", "KipkelionEast",
                "KipkelionWest", "Sigowet/Soin",
                "BometCentral", "BometEast", "Chepalungu", "Konoin", "Sotik",
                "Cherangany", "Endebess", "Kiminini", "Kwanza",
                "Saboti", "unknown4",
                "Kacheliba", "Kapenguria", "PokotSouth", "Sigor", "unknown3",
            ],
            "western_high_rainfall": [
                "Butere", "Ikolomani", "Khwisero", "Lugari", "Lurambi",
                "Malava", "Matungu", "MumiasEast", "MumiasWest",
                "Navakholo", "Shinyalu",
                "Bumula", "Kabuchai", "Kanduyi", "Kimilili", "Likuyani",
                "Lugari", "Mt.Elgon", "Sirisia", "Tongaren",
                "WebuteWest", "WebuyeEast",
                "Budalangi", "Butula", "Funyula", "Matayos",
                "Nambale", "TesoNorth", "TesoSouth",
                "Emuhaya", "Hamisi", "Luanda", "Sabatia", "Vihiga",
                "AlegoUsonga", "Bondo", "Gem", "Rarieda", "Ugenya", "Ugunja",
                "KisumuCentral", "KisumuEast", "KisumuWest",
                "Muhoroni", "Nyakach", "Nyando", "Seme",
                "HomaBayTown", "KabondoKasipul", "Karachuonyo",
                "Kasipul", "Mbita", "Ndhiwa", "Rangwe", "Suba", "unknown6",
                "Awendo", "KuriaEast", "KuriaWest", "Nyatike",
                "Rongo", "SunaEast", "SunaWest", "Uriri",
                "Bobasi", "BomachogeBorabu", "BomachogeChache", "Bonchari",
                "KitutuChacheNorth", "KitutuChacheSouth",
                "NyaribariChache", "NyaribariMasaba", "SouthMugirango",
                "Borabu", "KitutuMasaba", "NorthMugirango", "WestMugirango",
            ],
            "coastal_lowlands": [
                "Changamwe", "Jomvu", "Kisauni", "Likoni", "Mvita", "Nyali",
                "Ganze", "Kaloleni", "KilifiNorth", "KilifiSouth",
                "Magarini", "Malindi", "Rabai",
                "Kinango", "Lungalunga", "Matuga", "Msambweni",
                "LamuEast", "LamuWest",
                "Mwatate", "Taveta", "Voi", "Wundanyi",
            ],
        },
    },
    "sudan": {
        "gadm_file": os.path.join(GADM_DIR, "sudan", "gadm41_SDN_2.json"),
        "regions": {
            "desert": [
                "Addabah", "Dongola", "Merawi", "WadiHalfa",
                "AbuHamad", "AdDamer", "AlMatammah",
                "Atbara", "Berber", "Shendi",
            ],
            "semi_desert": [
                "Halayeb", "PortSudan", "Sinkat", "Tokar",
                "AlGash", "Hamashkorieb", "Kassala",
                "NahrAtbara", "Seteet",
                "Karary", "Khartoum", "KhartoumBahri",
                "Omdurman", "ShargEnNile", "SouthKhartoum", "UmBadda",
            ],
            "low_rainfall_savanna": [
                "Bara", "JebratalSheikh", "Sheikan",
                "Sowdari", "UmRawaba",
                "AlFasher", "Kabkabiya", "Kutum",
                "Mellit", "UmKadada",
                "AlFaw", "AlFushqa", "AlGadaref",
                "AlGalabat", "AlRahd",
            ],
            "high_rainfall_savanna": [
                "AbuJubaiyah", "Dilling", "Kadugli",
                "Rashad", "Talodi",
                "Buram", "IdElGhanem", "Kas", "Nyala", "Tulus",
                "Mukjar", "Zallingi",
                "AlDeain",
                "AlGeneina",
                "Abyei", "AsSalam", "EnNuhud",
                "Ghebeish", "Lagawa",
            ],
            "gezira_irrigated_nile": [
                "AlKamlin", "AlMahagil", "EastalGazera",
                "NorthalGazera", "SharqalGazera",
                "SouthalGazera", "UmAlGura",
                "AdDouiem", "AlGutaina", "AlJabalian", "Kosti",
                "AdDinder", "Sennar", "Singa",
            ],
            "blue_nile_rainfed": [
                "AdDamazin", "AlKurumik", "AlRoseires",
                "Baw", "Geissan",
            ],
        },
    },
    "south_sudan": {
        "gadm_file": os.path.join(GADM_DIR, "south_sudan", "gadm41_SSD_2.json"),
        "regions": {
            "greenbelt": [
                "Meridi", "Mundri", "Tombura", "Yambio",
                "BahralJabal", "KajoKaii", "NahrYei", "Terkaka",
            ],
            "ironstone_plateau": [
                "Raja", "Wau",
            ],
            "flood_plains": [
                "Akobo", "Ayod", "Bor", "FamalZaraf",
                "NahrAtiem", "Pibor", "Wat",
                "AlLeiri", "AlMayom", "Faring", "Rabkona",
                "Aliab", "Rumbek", "Shobet", "Yerol",
            ],
            "nile_sobat_river": [
                "AlMabien", "AlRenk", "Baleit", "Fashooda",
                "Malut", "Mayot", "Sobat", "Tonga",
            ],
            "eastern_pastoral_drylands": [
                "Amatonge", "Kapoeta", "Magwi", "Shokodom",
            ],
            "hills_and_mountains": [
                "Aryat", "Aweil", "NahrLol", "Wanjuk",
                "Gogrial", "Malek", "Tonj", "Warab",
            ],
        },
    },
}


def load_gadm(gadm_file):
    if not os.path.exists(gadm_file):
        raise FileNotFoundError(f"GADM file not found: {gadm_file}")
    return gpd.read_file(gadm_file)


def get_region_boundary(gadm, slug, name2_values, cache_dir):
    cache = os.path.join(cache_dir, f"{slug}_boundary.geojson")
    if os.path.exists(cache):
        return gpd.read_file(cache)

    subset = gadm[gadm["NAME_2"].isin(name2_values)].copy()
    if subset.empty:
        raise ValueError(
            f"No features found for NAME_2 in {name2_values}. "
            f"Available: {sorted(gadm['NAME_2'].unique())}"
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
            np.where(df[col_spi] <= -1.5, 2,
            np.where(df[col_spi] <= -1.0, 1, 0))
        )
        df[cls_col] = df[cls_col].astype("Int64")

    return df


def extract_region(tifs, geom, label):
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
    print(f"Found {len(tifs)} CHIRPS v3.0 tifs\n")

    for country, cfg in COUNTRIES.items():
        print(f"\n{'═' * 60}")
        print(f"  {country.upper()}")
        print(f"{'═' * 60}")

        gadm = load_gadm(cfg["gadm_file"])
        out_dir = os.path.join(SCRIPT_DIR, country)
        os.makedirs(out_dir, exist_ok=True)

        cache_dir = os.path.dirname(cfg["gadm_file"])

        for slug, name2_values in cfg["regions"].items():
            print(f"\n─── {slug}  ← {len(name2_values)} districts ───")
            boundary = get_region_boundary(gadm, slug, name2_values, cache_dir)
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
