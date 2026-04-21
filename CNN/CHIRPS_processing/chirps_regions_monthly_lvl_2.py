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
CNN_DIR = os.path.dirname(SCRIPT_DIR)                        # CNN/
ROOT_DIR = os.path.dirname(CNN_DIR)                          # project root
TIF_DIR = os.path.join(ROOT_DIR, "reference_data")
GADM_DIR = os.path.join(CNN_DIR, "gadm")

NODATA = -9999.0


COUNTRIES = {
    "somalia": {
        "gadm_file": os.path.join(GADM_DIR, "somalia", "gadm41_SOM_2.json"),
        "regions": {
            # Northwest agro-pastoral belt  (Awdal + Woqooyi Galbeed + Togdheer)
            "northwest_agropastoral": [
                "Baki", "Boorama", "Lughaya", "Zeylac",            # Awdal
                "Berbera", "Gabiley", "Hargeysa",                   # WoqooyiGalbeed
                "Burao", "Buuhoodle", "Oodweyne", "Sheekh",         # Togdheer
            ],
            # Northeast dry pastoral belt  (Sanaag + Sool + Bari + Nugaal)
            "northeast_dry_pastoral": [
                "Badhan", "Ceel-Afwein", "Ceerigaabo",              # Sanaag
                "Caynabo", "Lascaanod", "Taleex", "Xudun",          # Sool
                "Bander-Beyla", "Bosaaso", "Calawla",                # Bari
                "Iskushuban", "Qandala", "Qardho",
                "Burtinle", "Eyl", "Garoowe",                       # Nugaal
            ],
            # Central pastoral belt  (Mudug + Galguduud + Hiiraan)
            "central_pastoral": [
                "Gaalkacayo", "Goldogob", "Hobyo",                   # Mudug
                "Jariiban", "Xarardheere",
                "Caabudwaaq", "Cadaado", "CeelBuur",                # Galguduud
                "CeelDheer", "Dhuusamareeb",
                "BeledWeyn", "BuuloBurdo", "Jalalaqsi",             # Hiiraan
            ],
            # Juba riverine belt  (Jubba Dhexe + Jubba Hoose + Gedo)
            "juba_riverine": [
                "Bu'aale", "Jilib", "Saakow",                       # JubbadaDhexe
                "Afmadow", "Badhaadhe", "Jamaame", "Kismaayo",      # JubbadaHoose
                "Baar-Dheere", "BeledXaawo", "CeelWaaq",            # Gedo
                "Dolow", "Garbahaaray", "Luuk",
            ],
            # Shabelle riverine belt  (Shabelle Dhexe + Shabelle Hoose + Banaadir)
            "shabelle_riverine": [
                "Aadan", "Balcad", "Cadale", "Jawhar",              # ShabeellahaDhexe
                "Afgooye", "Baraawe", "Kuntuwaaray",                 # ShabeellahaHoose
                "Marka", "Qoryooley", "Sablale", "WanlaWeyn",
                "Mogadisho",                                         # Banaadir
            ],
            # Southern rainfed agro-pastoral belt  (Bay + Bakool)
            "southern_rainfed_agropastoral": [
                "Baydhabo", "BuurXakaba", "Diinsoor", "QansaxDheere",  # Bay
                "CeelBarde", "RabDhuure", "Tiyeeglow",                # Bakool
                "Wajid", "Xudur",
            ],
        },
    },
    "eritrea": {
        "gadm_file": os.path.join(GADM_DIR, "eritrea", "gadm41_ERI_2.json"),
        "regions": {
            # Central highlands  (Maekel)
            "central_highlands": [
                "AsmaraCity", "Berikh", "GhalaNefhi", "Serejeka",
            ],
            # Western lowlands  (Gash Barka)
            "western_lowlands": [
                "Akordat", "Barentu", "Dghe", "Forto", "Gogne",
                "Haykota", "La`ElayGash", "LogoAnseba", "Mansura",
                "Mogolo", "Omhajer", "Shemboko", "Teseneye",
            ],
            # Eastern lowlands  (Semienawi Keyih Bahri + Debubawi Keyih Bahri)
            "eastern_lowlands": [
                "Afabet", "Dahlak", "Foro", "Ghelaelo'",            # SemenawiKeyihBahri
                "Ghida`e", "Karora", "Mitswa`eCity", "Nakfa", "Sheib",
                "Areta'", "CentralSouthernRedSea",                   # DebubawiKeyihBahri
                "SouthSouthernRedSea",
            ],
            # Southwestern lowlands  (Anseba)
            "southwestern_lowlands": [
                "AdiTeklezan", "Asmat", "Elabered", "Gheleb",
                "Habero", "Hagaz", "Halhal", "Keren",
                "KerkeBet", "Sel`a",
            ],
            # Escarpment / Green belt  (Debub)
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
            # Coastal arid belt  (Djibouti city + Obock + Tadjoura)
            "coastal_arid": [
                "Djiboutii",                                         # Djiboutii
                "Adailou", "Dadda'to", "Khôr'Angar",                # Obock
                "Moulhoule", "Obock",
                "Balho", "Dorra", "Lac'Assal",                      # Tadjoura
                "MousaAli", "Randa", "Tadjoura",
            ],
            # Inland pastoral drylands  (Dikhil + Ali Sabieh)
            "inland_pastoral_drylands": [
                "AsEyla", "Dikhil", "Galafi", "Yuboki",             # Dikhil
                "AliAdde", "AliSabieh", "Holhol",                   # AliSabieh
            ],
            # Oasis / Wadi irrigated pockets  (Arta)
            "oasis_wadi_irrigated": [
                "Arta", "Yuboki",                                    # Arta
            ],
        },
    },
    "kenya": {
        "gadm_file": os.path.join(GADM_DIR, "kenya", "gadm41_KEN_2.json"),
        "regions": {
            # Northern arid pastoral belt
            "northern_arid_pastoral": [
                # Turkana
                "Loima", "TurkanaCentral", "TurkanaEast",
                "TurkanaNorth", "TurkanaSouth", "TurkanaWest", "unknown2",
                # Marsabit
                "Laisamis", "Moyale", "NorthHorr", "Saku",
                # Mandera
                "Banissa", "Lafey", "ManderaEast", "ManderaNorth",
                "ManderaSouth", "ManderaWest", "unknown1",
                # Wajir
                "Eldas", "Tarbaj", "WajirEast", "WajirNorth",
                "WajirSouth", "WajirWest",
                # Samburu
                "SamburuEast", "SamburuNorth", "SamburuWest",
                # Isiolo
                "IsioloNorth", "IsioloSouth",
            ],
            # Eastern semi-arid belt
            "eastern_semiarid": [
                # Garissa
                "Balambala", "Daadab", "Fafi", "GarissaTownship",
                "Ijara", "Lagdera",
                # Tana River
                "Bura", "Galole", "Garsen",
                # Kitui
                "KituiCentral", "KituiEast", "KituiRural", "KituiSouth",
                "KituiWest", "MwingiCentral", "MwingiNorth", "MwingiWest",
                # Makueni
                "Kaiti", "KibweziEast", "KibweziWest", "Kilome",
                "Makueni", "Mbooni",
                # Machakos
                "Kangundo", "Kathiani", "MachakosTown", "Masinga",
                "Matungulu", "Mavoko", "Mwala", "Yatta", "unknown7",
                # Embu
                "Manyatta", "MbeereNorth", "MbeereSouth", "Runyenjes",
                # Tharaka-Nithi
                "Chuka/Igambang'Ombe", "IgembeSouth", "Maara", "Tharaka",
                # Meru
                "Buuri", "CentralImenti", "IgembeCentral", "IgembeNorth",
                "IgembeSouth", "NorthImenti", "SouthImenti",
                "TiganiaEast", "TiganiaWest", "unknown5",
            ],
            # Central highlands
            "central_highlands": [
                # Nyeri
                "Kieni", "Mathira", "Mukurweini", "NyeriTown", "Othaya", "Tetu",
                # Kirinyaga
                "Gichugu", "KirinyagaCentral", "Mwea", "Ndia",
                # Murang'a
                "Gatanga", "Kandara", "Kangema", "Kigumo",
                "Kiharu", "Maragwa", "Mathioya",
                # Kiambu
                "GatunduNorth", "GatunduSouth", "Githunguri", "Juja",
                "Kabete", "Kiambaa", "Kiambu", "Kikuyu",
                "Lari", "Limuru", "Ruiru", "ThikaTown",
                # Nyandarua
                "Kinangop", "Kipipiri", "Ndaragwa", "OlJorok", "OlKalou",
                # Nairobi
                "DagorettiNorth", "DagorettiSouth", "EmbakasiCentral",
                "EmbakasiEast", "EmbakasiNorth", "EmbakasiSouth",
                "EmbakasiWest", "Kamukunji", "Kasarani", "Kibra",
                "Langata", "Makadara", "Mathare", "Roysambu",
                "Ruaraka", "Starehe", "Westlands",
                # Laikipia
                "LaikipiaEast", "LaikipiaNorth", "LaikipiaWest",
            ],
            # Rift valley highlands
            "rift_valley_highlands": [
                # Nakuru
                "Bahati", "Gilgil", "KuresoiNorth", "KuresoiSouth",
                "Molo", "Naivasha", "NakuruTownEast", "NakuruTownWest",
                "Njoro", "Rongai", "Subukia",
                # Narok
                "EmuruaDikirr", "Kilgoris", "NarokEast", "NarokNorth",
                "NarokSouth", "NarokWest",
                # Kajiado
                "KajiadoCentral", "KajiadoEast", "KajiadoNorth",
                "KajiadoSouth", "KajiadoWest",
                # Baringo
                "805", "BaringoCentral", "BaringoNorth", "BaringoSouth",
                "EldamaRavine", "Mogotio", "Tiaty",
                # Elgeyo-Marakwet
                "KeiyoNorth", "KeiyoSouth", "MarakwetEast", "MarakwetWest",
                # Uasin Gishu
                "Ainabkoi", "Kapseret", "Kesses", "Moiben", "Soy", "Turbo",
                # Nandi
                "Aldai", "Chesumei", "Emgwen", "Mosop", "NandiHills", "Tinderet",
                # Kericho
                "Ainamoi", "Belgut", "Bureti", "KipkelionEast",
                "KipkelionWest", "Sigowet/Soin",
                # Bomet
                "BometCentral", "BometEast", "Chepalungu", "Konoin", "Sotik",
                # Trans Nzoia
                "Cherangany", "Endebess", "Kiminini", "Kwanza",
                "Saboti", "unknown4",
                # West Pokot
                "Kacheliba", "Kapenguria", "PokotSouth", "Sigor", "unknown3",
            ],
            # Western high-rainfall farming belt
            "western_high_rainfall": [
                # Kakamega
                "Butere", "Ikolomani", "Khwisero", "Lugari", "Lurambi",
                "Malava", "Matungu", "MumiasEast", "MumiasWest",
                "Navakholo", "Shinyalu",
                # Bungoma
                "Bumula", "Kabuchai", "Kanduyi", "Kimilili", "Likuyani",
                "Lugari", "Mt.Elgon", "Sirisia", "Tongaren",
                "WebuteWest", "WebuyeEast",
                # Busia
                "Budalangi", "Butula", "Funyula", "Matayos",
                "Nambale", "TesoNorth", "TesoSouth",
                # Vihiga
                "Emuhaya", "Hamisi", "Luanda", "Sabatia", "Vihiga",
                # Siaya
                "AlegoUsonga", "Bondo", "Gem", "Rarieda", "Ugenya", "Ugunja",
                # Kisumu
                "KisumuCentral", "KisumuEast", "KisumuWest",
                "Muhoroni", "Nyakach", "Nyando", "Seme",
                # Homa Bay
                "HomaBayTown", "KabondoKasipul", "Karachuonyo",
                "Kasipul", "Mbita", "Ndhiwa", "Rangwe", "Suba", "unknown6",
                # Migori
                "Awendo", "KuriaEast", "KuriaWest", "Nyatike",
                "Rongo", "SunaEast", "SunaWest", "Uriri",
                # Kisii
                "Bobasi", "BomachogeBorabu", "BomachogeChache", "Bonchari",
                "KitutuChacheNorth", "KitutuChacheSouth",
                "NyaribariChache", "NyaribariMasaba", "SouthMugirango",
                # Nyamira
                "Borabu", "KitutuMasaba", "NorthMugirango", "WestMugirango",
            ],
            # Coastal lowlands
            "coastal_lowlands": [
                # Mombasa
                "Changamwe", "Jomvu", "Kisauni", "Likoni", "Mvita", "Nyali",
                # Kilifi
                "Ganze", "Kaloleni", "KilifiNorth", "KilifiSouth",
                "Magarini", "Malindi", "Rabai",
                # Kwale
                "Kinango", "Lungalunga", "Matuga", "Msambweni",
                # Lamu
                "LamuEast", "LamuWest",
                # Taita Taveta
                "Mwatate", "Taveta", "Voi", "Wundanyi",
            ],
        },
    },
    "sudan": {
        "gadm_file": os.path.join(GADM_DIR, "sudan", "gadm41_SDN_2.json"),
        "regions": {
            # Desert belt  (Northern + River Nile)
            "desert": [
                "Addabah", "Dongola", "Merawi", "WadiHalfa",        # Northern
                "AbuHamad", "AdDamer", "AlMatammah",                 # RiverNile
                "Atbara", "Berber", "Shendi",
            ],
            # Semi-desert belt  (Red Sea + Kassala + Khartoum)
            "semi_desert": [
                "Halayeb", "PortSudan", "Sinkat", "Tokar",          # RedSea
                "AlGash", "Hamashkorieb", "Kassala",                 # Kassala
                "NahrAtbara", "Seteet",
                "Karary", "Khartoum", "KhartoumBahri",              # Khartoum
                "Omdurman", "ShargEnNile", "SouthKhartoum", "UmBadda",
            ],
            # Low-rainfall savanna  (North Kurdufan + North Darfur + Al Qadarif)
            "low_rainfall_savanna": [
                "Bara", "JebratalSheikh", "Sheikan",                # NorthKurdufan
                "Sowdari", "UmRawaba",
                "AlFasher", "Kabkabiya", "Kutum",                   # NorthDarfur
                "Mellit", "UmKadada",
                "AlFaw", "AlFushqa", "AlGadaref",                   # AlQadarif
                "AlGalabat", "AlRahd",
            ],
            # High-rainfall savanna  (South Kurdufan + South/Central/East/West Darfur + West Kurdufan)
            "high_rainfall_savanna": [
                "AbuJubaiyah", "Dilling", "Kadugli",                # SouthKurdufan
                "Rashad", "Talodi",
                "Buram", "IdElGhanem", "Kas", "Nyala", "Tulus",     # SouthDarfur
                "Mukjar", "Zallingi",                                # CentralDarfur
                "AlDeain",                                           # EastDarfur (Nyala shared)
                "AlGeneina",                                         # WestDarfur
                "Abyei", "AsSalam", "EnNuhud",                      # WestKurdufan
                "Ghebeish", "Lagawa",
            ],
            # Gezira / Irrigated Nile belt  (Al Jazirah + White Nile + Sennar)
            "gezira_irrigated_nile": [
                "AlKamlin", "AlMahagil", "EastalGazera",            # AlJazirah
                "NorthalGazera", "SharqalGazera",
                "SouthalGazera", "UmAlGura",
                "AdDouiem", "AlGutaina", "AlJabalian", "Kosti",     # WhiteNile
                "AdDinder", "Sennar", "Singa",                      # Sennar
            ],
            # Blue Nile / Rainfed mechanized belt  (Blue Nile)
            "blue_nile_rainfed": [
                "AdDamazin", "AlKurumik", "AlRoseires",
                "Baw", "Geissan",
            ],
        },
    },
    "south_sudan": {
        "gadm_file": os.path.join(GADM_DIR, "south_sudan", "gadm41_SSD_2.json"),
        "regions": {
            # Greenbelt  (Western Equatoria + Central Equatoria)
            "greenbelt": [
                "Meridi", "Mundri", "Tombura", "Yambio",            # WestEquatoria
                "BahralJabal", "KajoKaii", "NahrYei", "Terkaka",    # CentralEquatoria
            ],
            # Ironstone plateau  (Western Bahr al Ghazal)
            "ironstone_plateau": [
                "Raja", "Wau",
            ],
            # Flood plains  (Jungoli + Unity + Lakes)
            "flood_plains": [
                "Akobo", "Ayod", "Bor", "FamalZaraf",              # Jungoli
                "NahrAtiem", "Pibor", "Wat",
                "AlLeiri", "AlMayom", "Faring", "Rabkona",          # Unity
                "Aliab", "Rumbek", "Shobet", "Yerol",               # Lakes
            ],
            # Nile-Sobat river zone  (Upper Nile)
            "nile_sobat_river": [
                "AlMabien", "AlRenk", "Baleit", "Fashooda",
                "Malut", "Mayot", "Sobat", "Tonga",
            ],
            # Eastern pastoral drylands  (Eastern Equatoria)
            "eastern_pastoral_drylands": [
                "Amatonge", "Kapoeta", "Magwi", "Shokodom",
            ],
            # Hills and Mountains  (North Bahr al Ghazal + Warap)
            "hills_and_mountains": [
                "Aryat", "Aweil", "NahrLol", "Wanjuk",              # NorthBahr-al-Ghazal
                "Gogrial", "Malek", "Tonj", "Warab",                # Warap
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

        # cache boundaries next to the GADM source file
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
