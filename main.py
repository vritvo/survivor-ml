"""Rebuild the web app's data: load the survivoR export, build the modeling
table, and write one JSON per season for the app to read.

This is what the scheduled refresh workflow runs.

Usage:
    uv run python main.py                # rebuild from the local data/json/
    uv run python main.py --fetch-data   # download the latest JSON tables first
"""

import argparse
import urllib.request

from src.load import DATA_DIR, TABLES_TO_LOAD, load_data
from src.features.build import build_modeling_table
from src.export import export_all_seasons

DATA_URL = "https://raw.githubusercontent.com/doehm/survivoR/master/dev/json"

def fetch_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename in TABLES_TO_LOAD.values():
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(f"{DATA_URL}/{filename}", DATA_DIR / filename)
    print("Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-data", action="store_true",
                        help="Download latest survivoR JSON tables before running")
    args = parser.parse_args()

    if args.fetch_data:
        fetch_data()

    data = load_data()
    df = build_modeling_table(data)
    export_all_seasons(df)

if __name__ == "__main__":
    main()
