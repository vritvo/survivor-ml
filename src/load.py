"""Load and clean survivoR data from Excel export."""

import pandas as pd
from pathlib import Path

# Sheets to load
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

DATA_PATH = Path(__file__).parent.parent / "data" / "survivoR.xlsx"


def load_raw(path: Path = DATA_PATH) -> dict[str, pd.DataFrame]:
    """Load raw sheets from Excel into a dict of DataFrames."""
    xls = pd.ExcelFile(path)
    return {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in SHEETS_TO_LOAD}


# --- Per-sheet cleaning functions ---
# Each takes a raw DataFrame and returns a cleaned one.
# The pattern: fix types, select only columns needed for modeling, drop junk.


def clean_castaways(df: pd.DataFrame) -> pd.DataFrame:
    """One row per castaway per season appearance.

    Has version column — used downstream to filter to US-only.
    """
    df = df.copy()
    return df[[
        "version", "version_season", "season",
        "castaway_id", "castaway", "full_name",
        "age", "city", "state",
        "episode", "day", "order", "result", "jury_status",
        "place", "original_tribe", "jury", "finalist", "winner",
    ]]


def clean_castaway_details(df: pd.DataFrame) -> pd.DataFrame:
    """One row per unique castaway (across all appearances). Demographics."""
    df = df.copy()
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])
    return df[[
        "castaway_id", "full_name", "castaway",
        "date_of_birth", "gender", "personality_type",
    ]]


def clean_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per episode. Dates needed for computing player age."""
    df = df.copy()
    df["episode_date"] = pd.to_datetime(df["episode_date"])
    return df[[
        "version", "version_season", "season",
        "episode", "episode_title", "episode_label", "episode_date",
    ]]


def clean_vote_history(df: pd.DataFrame) -> pd.DataFrame:
    """One row per castaway per vote event at tribal council."""
    df = df.copy()
    df = df.drop_duplicates()
    return df[[
        "version", "version_season", "season", "episode", "day",
        "tribe_status", "tribe", "castaway", "castaway_id",
        "immunity", "vote", "vote_id", "voted_out", "voted_out_id",
        "vote_order", "nullified", "tie",
    ]]


def clean_challenge_results(df: pd.DataFrame) -> pd.DataFrame:
    """One row per castaway per challenge. Dropped won_* flags (derivable from result + type)."""
    df = df.copy()
    return df[[
        "version", "version_season", "season", "episode",
        "castaway_id", "castaway", "tribe", "tribe_status",
        "challenge_type", "outcome_type", "result", "sit_out", "challenge_id",
    ]]


def clean_confessionals(df: pd.DataFrame) -> pd.DataFrame:
    """One row per castaway per episode."""
    df = df.copy()
    
    # Dropped confessional_time (mostly null).
    return df[[
        "version", "version_season", "season", "episode",
        "castaway_id", "castaway", "confessional_count",
    ]]


def clean_boot_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """One row per castaway per episode (all players still in the game).
    """
    df = df.copy()
    return df[[
        "version", "version_season", "season", "episode",
        "castaway_id", "castaway",
        "order", "n_boots", "final_n",
        "tribe", "tribe_status", "game_status",
    ]]


def clean_advantage_details(df: pd.DataFrame) -> pd.DataFrame:
    """One row per advantage per season."""
    df = df.copy()
    return df[[
        "version", "version_season", "season",
        "advantage_id", "advantage_type",
    ]]


def clean_advantage_movement(df: pd.DataFrame) -> pd.DataFrame:
    """One row per advantage event (found, played, transferred, etc.)."""
    df = df.copy()
    return df[[
        "version", "version_season", "season", "episode", "day",
        "castaway_id", "castaway", "advantage_id",
        "event", "played_for", "played_for_id", "success", "votes_nullified",
    ]]


def clean_tribe_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """One row per castaway per episode — which tribe they're on."""
    df = df.copy()
    return df[[
        "version", "version_season", "season", "episode", "day",
        "castaway_id", "castaway", "tribe", "tribe_status",
    ]]


# Maps sheet names to their cleaning functions.
CLEANERS = {
    "Castaways": clean_castaways,
    "Castaway Details": clean_castaway_details,
    "Episodes": clean_episodes,
    "Vote History": clean_vote_history,
    "Challenge Results": clean_challenge_results,
    "Confessionals": clean_confessionals,
    "Boot Mapping": clean_boot_mapping,
    "Advantage Details": clean_advantage_details,
    "Advantage Movement": clean_advantage_movement,
    "Tribe Mapping": clean_tribe_mapping,
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
    for name, df in data.items():
        print(f"{name:25s} {df.shape[0]:>6,} rows x {df.shape[1]:>2} cols")
