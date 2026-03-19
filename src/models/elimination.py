"""
Model(elimination): predict who gets voted out each episode.

Usage:
    python -m src.models.elimination                  # train & evaluate (single split + CV)
    python -m src.models.elimination --tune           # hyperparameter grid search
    python -m src.models.elimination --select         # forward feature selection

For predicting a specific season, use predict_season(df, target_season) from code/notebook.

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


# --- Prediction ---

def predict(
    model: LogisticRegression,
    scaler: StandardScaler,
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Generate per-player elimination probabilities, normalized within each episode.

    Works on any DataFrame with the right feature columns — train, test, or a new season.
    Includes the target column in the output if present in the input.
    """
    features = feature_cols or FEATURE_COLS
    X = scaler.transform(df[features])
    probs = model.predict_proba(X)[:, 1]  # P(eliminated)

    id_cols = ["season", "episode", "castaway_id", "castaway"]
    if TARGET_COL in df.columns:
        id_cols.append(TARGET_COL)
    preds = df[id_cols].copy()
    preds["prob_eliminated"] = probs

    # Raw logistic regression outputs don't sum to 1 across players in an episode,
    # so normalize to get a proper "who goes home" distribution.
    episode_sums = preds.groupby(["season", "episode"])["prob_eliminated"].transform("sum")
    preds["prob_eliminated"] = preds["prob_eliminated"] / episode_sums

    return preds


# --- Evaluation ---

def evaluate(predictions: pd.DataFrame) -> dict:
    """Compute episode accuracy and Brier score from a predictions DataFrame.

    Expects columns: prob_eliminated, eliminated_this_episode.
    """
    preds = predictions

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

    # --- Naive baseline: uniform 1/N probability per player ---
    baseline_accuracy = (1 / preds.groupby(["season", "episode"]).size()).mean()
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


def predict_and_evaluate(
    model: LogisticRegression,
    scaler: StandardScaler,
    test: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> dict:
    """Generate predictions and compute metrics on the test set."""
    preds = predict(model, scaler, test, feature_cols)
    return evaluate(preds)


# --- Full pipeline ---

def _make_train_and_evaluate(feature_cols: list[str] | None = None, **kwargs):
    """Create a train-and-evaluate callback. Passes kwargs to train_model."""
    def fn(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        model, scaler = train_model(train_df, feature_cols=feature_cols, **kwargs)
        return predict_and_evaluate(model, scaler, test_df, feature_cols=feature_cols)
    return fn


def train_eval_pipeline(df: pd.DataFrame) -> dict:
    """Run the full pipeline: preprocess -> split -> train -> evaluate.

    Single fixed split (TRAIN_SEASONS vs TEST_SEASONS). For a more robust
    estimate, use cross_validate() which averages over multiple splits.
    """
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

    # Show which features the model relies on most
    coef_tuples = list(zip(FEATURE_COLS, model.coef_[0]))
    coef_tuples_sorted = sorted(coef_tuples, key=lambda x: abs(x[1]), reverse=True)
    print(f"\nFeature coefficients (ordered by absolute value):")
    for name, coef in coef_tuples_sorted:
        print(f"  {name:45s} {coef:+.4f}")

    return results


def cross_validate(df: pd.DataFrame) -> dict:
    """Run expanding-window cross-validation and print results.

    Each fold trains on all seasons up to some cutoff, then tests on the next test_window seasons.
    """
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

    Returns (best_C, best_l1_ratio). Selects by highest episode accuracy.
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


def predict_season(df: pd.DataFrame, target_season: int) -> pd.DataFrame:
    """
    Train on all seasons before target_season, return predictions for that season. No evaluation.
    """
    df = preprocess(df, FEATURE_COLS)
    train = df[df["season"] < target_season]
    target = df[df["season"] == target_season]

    if target.empty:
        raise ValueError(f"No data found for season {target_season}")

    print(f"Training on seasons 1-{target_season - 1} ({len(train):,} rows), "
          f"predicting season {target_season} ({len(target):,} rows)")

    model, scaler = train_model(train)
    return predict(model, scaler, target)


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
