import os
import re
import urllib.request
from datetime import date

BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/africa/tifs/"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
START_YEAR = 1981

def get_available_files():
    with urllib.request.urlopen(BASE_URL) as resp:
        html = resp.read().decode()
    pattern = r'href="(chirps-v3\.0\.\d{4}\.\d{2}\.tif)"'
    return sorted(set(re.findall(pattern, html)))

def parse_year_month(filename):
    m = re.search(r'chirps-v3\.0\.(\d{4})\.(\d{2})\.tif', filename)
    return int(m.group(1)), int(m.group(2))

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    today = date.today()
    files = get_available_files()

    targets = []
    for f in files:
        y, m = parse_year_month(f)
        if y >= START_YEAR and date(y, m, 1) <= today:
            targets.append(f)

    print(f"Found {len(targets)} files to download (from {START_YEAR} to latest)")

    for i, filename in enumerate(targets, 1):
        tif_path = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(tif_path):
            print(f"[{i}/{len(targets)}] Already exists: {filename}")
            continue

        url = BASE_URL + filename
        print(f"[{i}/{len(targets)}] Downloading: {filename}")

        try:
            urllib.request.urlretrieve(url, tif_path)
            print(f"  -> Saved: {filename}")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            if os.path.exists(tif_path):
                os.remove(tif_path)

    print("Done.")

if __name__ == "__main__":
    main()
