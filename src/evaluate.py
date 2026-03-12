"""Shared evaluation utilities: temporal cross-validation and feature selection."""

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


def forward_selection(
    df: pd.DataFrame,
    candidate_features: list[str],
    make_cv_callback: Callable[[list[str]], Callable],
    metric: str = "episode_accuracy",
    higher_is_better: bool = True,
    **cv_kwargs,
) -> dict:
    """Greedy forward feature selection using expanding-window CV.

    Starts with no features, adds the one that improves the metric most,
    repeats until no feature improves performance.

    Args:
        df: Preprocessed modeling table.
        candidate_features: All features to consider.
        make_cv_callback: Factory that takes a list of feature names and returns
            a (train_df, test_df) -> dict callback for CV.
        metric: Key in the CV results dict to optimize (default: episode_accuracy).
        higher_is_better: If True, maximize the metric. If False, minimize it.
        **cv_kwargs: Passed to expanding_window_cv.

    Returns:
        Dict with selected features, history of each step, and best score.
    """
    selected: list[str] = []
    remaining = list(candidate_features)
    history: list[dict] = []
    best_score = -np.inf if higher_is_better else np.inf

    def _is_improvement(new: float, old: float) -> bool:
        return new > old if higher_is_better else new < old

    step = 0
    while remaining:
        step += 1
        step_results = []

        for feature in remaining:
            trial_features = selected + [feature]
            callback = make_cv_callback(trial_features)
            cv = expanding_window_cv(df, callback, **cv_kwargs)
            score = cv["mean"][metric]
            step_results.append({"feature": feature, "score": score})

        step_results.sort(key=lambda r: r["score"], reverse=higher_is_better)
        best_candidate = step_results[0]

        if not _is_improvement(best_candidate["score"], best_score):
            fmt = f"{best_candidate['score']:.4f}"
            best_fmt = f"{best_score:.4f}"
            print(f"\n  Step {step}: no improvement (best candidate "
                  f"{best_candidate['feature']} -> {fmt}, "
                  f"current best {best_fmt}). Stopping.")
            break

        selected.append(best_candidate["feature"])
        remaining.remove(best_candidate["feature"])
        best_score = best_candidate["score"]

        history.append({
            "step": step,
            "added": best_candidate["feature"],
            "score": best_score,
            "all_candidates": step_results,
        })

        print(f"  Step {step}: +{best_candidate['feature']:45s} -> {best_score:.4f}  "
              f"({len(remaining)} remaining)")

    return {
        "selected_features": selected,
        "best_score": best_score,
        "history": history,
    }
