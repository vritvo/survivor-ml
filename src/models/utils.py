"""Shared helpers for preparing modeling tables and splitting them by season."""

from collections.abc import Iterable

import pandas as pd


def preprocess(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Drop rows with a missing value in any of feature_cols.

    sklearn estimators reject NaNs, so models run this before
    splitting into train and test.
    """
    df = df.copy()
    df = df.dropna(subset=feature_cols)

    return df


def split_by_season(
    df: pd.DataFrame,
    train_seasons: Iterable[int],
    test_seasons: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/test by season, so no season appears in both."""
    train = df[df["season"].isin(train_seasons)]
    test = df[df["season"].isin(test_seasons)]
    return train, test
