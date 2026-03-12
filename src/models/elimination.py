"""
Model(elimination): predict who gets voted out each episode.

Usage:
    python -m src.models.elimination                  # train & evaluate (single split + CV)
    python -m src.models.elimination --tune           # hyperparameter grid search
    python -m src.models.elimination --select         # forward feature selection
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
from src.load import load_data
from src.features.build import build_modeling_table
from src.evaluate import expanding_window_cv, forward_selection
from src.models.utils import preprocess, split_by_season

# --- Configuration ---

FEATURE_COLS = [
    # "episode",
    "age",
    # "age_x_episode",
    # "gender_Male",
    # "gender_Non-binary",
    # "personality_missing",
    # "mbti_extravert",
    # "mbti_intuitive",
    "mbti_feeling",
    # "mbti_perceiving",
    # "is_returnee",
    "num_previous_seasons",
    # "votes_against_cumulative_by_previous_ep",
    "votes_against_last_3_eps", 
    # "correct_votes_cumulative_by_previous_ep",
    # "times_in_danger",
    # "final_n",
    # "tribe_status_Merged",
    # "tribe_status_Original",
    # "tribe_status_Swapped",
    # "tribe_status_Swapped_2",
    "advantages_held",
    "individual_immunity_wins",
    # "has_advantage",
    # "confessional_share_last_ep",
]

TARGET_COL = "eliminated_this_episode"

TRAIN_SEASONS = range(1, 41)
TEST_SEASONS = range(41, 51)

# Hyperparameters (update after running --tune)
C = 5
L1_RATIO = .5


# --- Training ---

def train_model(
    train: pd.DataFrame,
    C: float = C,
    l1_ratio: float = L1_RATIO,
    feature_cols: list[str] | None = None,
) -> tuple[LogisticRegression, StandardScaler]:
    """Train elastic net logistic regression (saga solver)."""
    features = feature_cols or FEATURE_COLS
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    y_train = train[TARGET_COL]

    model = LogisticRegression(
        C=C, l1_ratio=l1_ratio, solver="saga", max_iter=5000,
    )
    model.fit(X_train, y_train)

    return model, scaler


# --- Evaluation ---

def predict_and_evaluate(
    model: LogisticRegression,
    scaler: StandardScaler,
    test: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> dict:
    """Generate predictions and compute metrics on the test set.

    Returns a dict with:
    - episode_accuracy: fraction of episodes where argmax prediction = actual boot
    - brier_score: calibration of predicted probabilities (lower = better)
    - predictions: DataFrame with per-player probabilities for inspection
    """
    features = feature_cols or FEATURE_COLS
    X_test = scaler.transform(test[features])
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
    # Break ties randomly to avoid data-ordering artifacts.
    # TODO: think about how to handle episodes with 2+ eliminations —
    # currently we count it as correct if our top pick was any of the eliminated players.
    rng = np.random.default_rng(42)
    correct = 0
    total = 0
    for (_season, _episode), group in preds.groupby(["season", "episode"]):
        if group[TARGET_COL].sum() == 0:
            continue  # skip no-elimination episodes
        probs_with_noise = group["prob_eliminated"] + rng.uniform(0, 1e-10, len(group))
        top_pick = group.loc[probs_with_noise.idxmax(), "castaway_id"]
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

def _make_train_and_evaluate(feature_cols: list[str] | None = None, **kwargs):
    """Create a train-and-evaluate callback. Passes kwargs to train_model."""
    def fn(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        model, scaler = train_model(train_df, feature_cols=feature_cols, **kwargs)
        return predict_and_evaluate(model, scaler, test_df, feature_cols=feature_cols)
    return fn


def train_eval_pipeline(df: pd.DataFrame) -> dict:
    """Run the full pipeline: preprocess -> split -> train -> evaluate."""
    df = preprocess(df, FEATURE_COLS)
    train, test = split_by_season(df, TRAIN_SEASONS, TEST_SEASONS)

    print(f"Train: {len(train):,} rows ({train['season'].nunique()} seasons)")
    print(f"Test:  {len(test):,} rows ({test['season'].nunique()} seasons)")
    print(f"Features: {FEATURE_COLS}")
    print()

    model, scaler = train_model(train)
    results = predict_and_evaluate(model, scaler, test)

    print(f"Episode accuracy: {results['episode_accuracy']:.1%} (model) | {results['baseline_accuracy']:.1%} (baseline)  ({results['n_test_episodes']} episodes)")
    print(f"Brier score:      {results['brier_score']:.4f} (model) | {results['baseline_brier']:.4f} (baseline)")

    coef_tuples = list(zip(FEATURE_COLS, model.coef_[0]))
    coef_tuples_sorted = sorted(coef_tuples, key=lambda x: abs(x[1]), reverse=True)
    print(f"\nFeature coefficients (ordered by absolute value):")
    for name, coef in coef_tuples_sorted:
        print(f"  {name:45s} {coef:+.4f}")

    return results


def cross_validate(df: pd.DataFrame) -> dict:
    """Run expanding-window cross-validation and print results."""
    df = preprocess(df, FEATURE_COLS)

    print(f"Features: {FEATURE_COLS}")
    print(f"Running expanding-window CV...\n")

    cv_results = expanding_window_cv(df, _make_train_and_evaluate(feature_cols=FEATURE_COLS))


    for fold in cv_results["folds"]:
        print(f"  Train {fold['fold_train_seasons']:>5s} | Test {fold['fold_test_seasons']:>5s} | "
              f"Accuracy: {fold['episode_accuracy']:.1%} (baseline {fold['baseline_accuracy']:.1%}) | "
              f"Episodes: {fold['n_test_episodes']}")

    mean = cv_results["mean"]
    print(f"\n  Mean accuracy: {mean['episode_accuracy']:.1%} (baseline {mean['baseline_accuracy']:.1%}) "
          f"over {cv_results['n_folds']} folds")
    print(f"  Mean Brier:    {mean['brier_score']:.4f} (baseline {mean['baseline_brier']:.4f})")

    return cv_results


def tune_hyperparameters(df: pd.DataFrame) -> tuple[float, float]:
    """Grid search over C and l1_ratio using expanding-window CV.

    Returns (best_C, best_l1_ratio).
    """
    df = preprocess(df, FEATURE_COLS)

    C_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    l1_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

    print(f"Tuning over {len(C_values)} x {len(l1_ratios)} = {len(C_values) * len(l1_ratios)} combos\n")

    results = []
    for C in C_values:
        for l1 in l1_ratios:
            cv = expanding_window_cv(df, _make_train_and_evaluate(feature_cols=FEATURE_COLS, C=C, l1_ratio=l1))
            mean_acc = cv["mean"]["episode_accuracy"]
            mean_brier = cv["mean"]["brier_score"]
            results.append({"C": C, "l1_ratio": l1, "accuracy": mean_acc, "brier": mean_brier})
            print(f"  C={C:<6} l1_ratio={l1:<5} → accuracy={mean_acc:.1%}  brier={mean_brier:.4f}")

    results_sorted = sorted(results, key=lambda r: r["accuracy"], reverse=True)
    best = results_sorted[0]
    print(f"\nBest: C={best['C']}, l1_ratio={best['l1_ratio']} "
          f"→ accuracy={best['accuracy']:.1%}, brier={best['brier']:.4f}")

    return best["C"], best["l1_ratio"]


def run_forward_selection(df: pd.DataFrame) -> dict:
    """Run forward feature selection and print results."""
    df = preprocess(df, FEATURE_COLS)

    def make_cv_callback(features: list[str]):
        return _make_train_and_evaluate(feature_cols=features)

    print(f"Candidates: {FEATURE_COLS}\n")
    results = forward_selection(df, FEATURE_COLS, make_cv_callback)

    print(f"\nSelected {len(results['selected_features'])} features "
          f"(from {len(FEATURE_COLS)} candidates):")
    for f in results["selected_features"]:
        print(f"  - {f}")
    print(f"\nBest CV accuracy: {results['best_score']:.1%}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tune", action="store_true", help="Run hyperparameter grid search")
    group.add_argument("--select", action="store_true", help="Run forward feature selection")
    args = parser.parse_args()

    data = load_data()
    df = build_modeling_table(data)

    if args.tune:
        print("=== Hyperparameter tuning ===\n")
        tune_hyperparameters(df)
    elif args.select:
        print("=== Forward feature selection ===\n")
        run_forward_selection(df)
    else:
        print("=== Single split ===\n")
        results = train_eval_pipeline(df)

        print("\n\n=== Cross-validation ===\n")
        cv_results = cross_validate(df)
