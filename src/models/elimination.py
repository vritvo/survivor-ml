"""
Model(elimination): predict who gets voted out each episode.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
from src.load import load_data
from src.features.build import build_modeling_table

# --- Configuration ---

# Features the model uses.
FEATURE_COLS = [
    "episode",
    "age",
    "age_x_episode",
    "gender_Male",
    "gender_Non-binary",
    "personality_missing",
    # "mbti_extravert",
    # "mbti_intuitive",
    # "mbti_feeling",
    # "mbti_perceiving",
    "is_returnee",
    #"num_previous_seasons",
    "votes_against_cumulative_by_previous_ep",
    "votes_against_last_3_eps", 
    "correct_votes_cumulative_by_previous_ep",
    #"final_n",
    "tribe_status_Merged",
    "tribe_status_Original",
    "tribe_status_Swapped",
    "tribe_status_Swapped_2",
    "advantages_held",
    # "has_advantage",
    # "confessional_share_last_ep",
]

TARGET_COL = "eliminated_this_episode"

# Temporal split: train on seasons 1-40, test on 41-50
TRAIN_SEASONS = range(1, 41)
TEST_SEASONS = range(41, 51)


# --- Preprocessing ---

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the modeling table for sklearn.

    - Drop rows with missing features
    """
    df = df.copy()

    # Drop rows where any feature is missing 
    df = df.dropna(subset=FEATURE_COLS)

    return df


# --- Train/test split ---

def split_by_season(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/test by season (temporal split, no leakage)."""
    train = df[df["season"].isin(TRAIN_SEASONS)]
    test = df[df["season"].isin(TEST_SEASONS)]
    return train, test


# --- Training ---

def train_model(train: pd.DataFrame) -> tuple[LogisticRegression, StandardScaler]:
    """Train a logistic regression on the training set.

    Returns the fitted model and scaler (needed to transform test data the same way).
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLS])
    y_train = train[TARGET_COL]

    model = LogisticRegression(l1_ratio=.5, solver="saga", max_iter=5000, C=1)
    
    
    model.fit(X_train, y_train)

    return model, scaler


# --- Evaluation ---

def predict_and_evaluate(
    model: LogisticRegression,
    scaler: StandardScaler,
    test: pd.DataFrame,
) -> dict:
    """Generate predictions and compute metrics on the test set.

    Returns a dict with:
    - episode_accuracy: fraction of episodes where argmax prediction = actual boot
    - brier_score: calibration of predicted probabilities (lower = better)
    - predictions: DataFrame with per-player probabilities for inspection
    """
    X_test = scaler.transform(test[FEATURE_COLS])
    probs = model.predict_proba(X_test)[:, 1]  # P(eliminated)

    # Build predictions DataFrame for per-episode evaluation
    preds = test[["season", "episode", "castaway_id", "castaway", TARGET_COL]].copy()
    preds["prob_eliminated"] = probs

    # Normalize within each episode so probabilities sum to 1
    episode_sums = preds.groupby(["season", "episode"])["prob_eliminated"].transform("sum")
    preds["prob_eliminated"] = preds["prob_eliminated"] / episode_sums

    # --- Episode-level accuracy ---
    # For each episode, did the player with the highest predicted probability
    # actually get eliminated?
    # TODO: think about how to handle episodes with 2+ eliminations —
    # currently we count it as correct if our top pick was any of the eliminated players.
    correct = 0
    total = 0
    for (_season, _episode), group in preds.groupby(["season", "episode"]):
        if group[TARGET_COL].sum() == 0:
            continue  # skip no-elimination episodes
        top_pick = group.loc[group["prob_eliminated"].idxmax(), "castaway_id"]
        actually_eliminated = group.loc[group[TARGET_COL] == 1, "castaway_id"].values
        if top_pick in actually_eliminated:
            correct += 1
        total += 1

    episode_accuracy = correct / total if total > 0 else 0

    # --- Brier score ---
    brier = brier_score_loss(preds[TARGET_COL], preds["prob_eliminated"])

    # --- Naive baseline: random guess ---
    # If you picked a random player each episode, your accuracy = 1/N where N = cast size
    baseline_accuracy = (1 / preds.groupby(["season", "episode"]).size()).mean()
    # Uniform probabilities: each player gets 1/N in their episode
    baseline_probs = 1 / preds.groupby(["season", "episode"])["season"].transform("count")
    baseline_brier = brier_score_loss(preds[TARGET_COL], baseline_probs)

    return {
        "episode_accuracy": episode_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "brier_score": brier,
        "baseline_brier": baseline_brier,
        "n_test_episodes": total,
        "predictions": preds,
    }


# --- Full pipeline ---

def train_eval_pipeline(df: pd.DataFrame) -> dict:
    """Run the full pipeline: preprocess → split → train → evaluate."""
    df = preprocess(df)
    train, test = split_by_season(df)

    print(f"Train: {len(train):,} rows ({train['season'].nunique()} seasons)")
    print(f"Test:  {len(test):,} rows ({test['season'].nunique()} seasons)")
    print(f"Features: {FEATURE_COLS}")
    print()

    model, scaler = train_model(train)
    results = predict_and_evaluate(model, scaler, test)

    print(f"Episode accuracy: {results['episode_accuracy']:.1%} (model) | {results['baseline_accuracy']:.1%} (baseline)  ({results['n_test_episodes']} episodes)")
    print(f"Brier score:      {results['brier_score']:.4f} (model) | {results['baseline_brier']:.4f} (baseline)")

    # Feature coefficients (ranked by absolute value)
    coef_tuples = list(zip(FEATURE_COLS, model.coef_[0]))
    coef_tuples.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\nFeature coefficients (ordered by absolute value):")
    for name, coef in coef_tuples:
        print(f"  {name:45s} {coef:+.4f}")

    return results


if __name__ == "__main__":

    data = load_data()
    df = build_modeling_table(data)
    results = train_eval_pipeline(df)
