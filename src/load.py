"""Load and clean survivoR data from Excel export."""

import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "survivoR.xlsx"

# Sheets we'll actually use for modeling.
# Add more here as you need them.
SHEETS_TO_LOAD = [
    "Castaways",
    "Castaway Details",
    "Episodes",
    "Vote History",
    "Challenge Results",
    "Confessionals",
    "Boot Mapping",
    "Advantage Details",
    "Advantage Movement",
    "Tribe Mapping",
]


def load_raw(path: Path = DATA_PATH) -> dict[str, pd.DataFrame]:
    """Load raw sheets from Excel into a dict of DataFrames."""
    xls = pd.ExcelFile(path)
    return {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in SHEETS_TO_LOAD}


# --- Per-sheet cleaning functions ---
# Each takes a raw DataFrame and returns a cleaned one.
# The pattern: fix types, drop unnecessary columns, standardize names.


def clean_sheet(sheet: str) -> pd.DataFrame:
    return sheet 

# Maps sheet names to their cleaning functions.
CLEANERS = {
    
}


def load_data(path: Path = DATA_PATH) -> dict[str, pd.DataFrame]:
    """Load and clean all sheets. This is the main entry point.

    Usage:
        from src.load import load_data
        data = load_data()
        votes = data["Vote History"]
    """
    raw = load_raw(path)
    return {name: CLEANERS[name](df) for name, df in raw.items()}


if __name__ == "__main__":
    data = load_data()
    
    
    