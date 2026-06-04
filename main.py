import argparse
from pathlib import Path
import urllib.request

from src.load import load_data
from src.features.build import build_modeling_table
from src.export import export_all_seasons

DATA_URL = "https://github.com/doehm/survivoR/raw/refs/heads/master/dev/xlsx/survivoR.xlsx"
DATA_PATH = Path("data/survivoR.xlsx")

def fetch_data():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading survivoR.xlsx...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print("Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-data", action="store_true",
                        help="Download latest survivoR.xlsx before running")
    args = parser.parse_args()

    if args.fetch_data:
        fetch_data()

    data = load_data()
    df = build_modeling_table(data)
    export_all_seasons(df)

if __name__ == "__main__":
    main()