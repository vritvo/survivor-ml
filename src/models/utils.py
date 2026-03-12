""" 
Utility functions for modeling.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def preprocess(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Prepare the modeling table for sklearn.

    - Drop rows with missing features
    """
    df = df.copy()

    # Drop rows where any feature is missing 
    df = df.dropna(subset=feature_cols)

    return df



def split_by_season(df: pd.DataFrame, train_seasons: list[int], test_seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/test by season (temporal split, no leakage)."""
    train = df[df["season"].isin(train_seasons)]
    test = df[df["season"].isin(test_seasons)]
    return train, test

