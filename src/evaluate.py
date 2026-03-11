"""Shared evaluation utilities: temporal cross-validation over seasons."""

import pandas as pd
import numpy as np
from typing import Callable


def expanding_window_cv(
    df: pd.DataFrame,
    train_and_evaluate_fn: Callable[[pd.DataFrame, pd.DataFrame], dict],
    min_train_seasons: int = 10,
    test_window: int = 5,
) -> dict:
    """Expanding-window temporal cross-validation over seasons.

    Splits data into folds where training starts at season 1 and grows,
    while the test window slides forward. Each fold trains on all seasons
    before the test window.

    Example with min_train_seasons=10, test_window=5, 49 seasons:
        Fold 1: train 1-10,  test 11-15
        Fold 2: train 1-15,  test 16-20
        Fold 3: train 1-20,  test 21-25
        ...

    Returns:
        Dict with per-fold results and averaged metrics.
    """
    all_seasons = sorted(df["season"].unique())
    max_season = all_seasons[-1]

    fold_results = []
    test_start = min_train_seasons + 1

    # Loop through seasons, incrementing by test_window
    while test_start <= max_season:
        
        # Get the train and test seasons
        test_end = min(test_start + test_window - 1, max_season)
        train_seasons = [s for s in all_seasons if s < test_start]
        test_seasons = [s for s in all_seasons if test_start <= s <= test_end]

        if not train_seasons or not test_seasons:
            test_start += test_window
            continue

        train_df = df[df["season"].isin(train_seasons)]
        test_df = df[df["season"].isin(test_seasons)]

        # Run the model on the train and test data
        metrics = train_and_evaluate_fn(train_df, test_df)
        
        # Add the train and test seasons to the metrics
        metrics["fold_train_seasons"] = f"1-{max(train_seasons)}"
        metrics["fold_test_seasons"] = f"{min(test_seasons)}-{max(test_seasons)}"
        fold_results.append(metrics)

        test_start += test_window

    # Calculate the averages of the metrics
    numeric_keys = [k for k in fold_results[0] if isinstance(fold_results[0][k], (int, float))]
    averages = {k: np.mean([f[k] for f in fold_results]) for k in numeric_keys}

    return {
        "folds": fold_results,
        "mean": averages,
        "n_folds": len(fold_results),
    }
