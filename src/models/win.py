"""
Model(win): predict each player's probability of winning the season.

Usage:
    python -m src.models.win                  # train & evaluate (single split + CV)
    python -m src.models.win --tune           # hyperparameter grid search
    python -m src.models.win --select         # forward feature selection
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
from src.models.elimination import (
    train_model as train_elim_model,
    FEATURE_COLS as ELIM_FEATURE_COLS,
)

# --- Configuration ---

FEATURE_COLS = [
    # "episode",
    "age", #
    #  "age_x_episode",
    # "gender_Male",
    # "gender_Non-binary",
    # "personality_missing",
    # "mbti_extravert",
    # "mbti_intuitive",
    # "mbti_feeling",
    # "mbti_perceiving",
    # "is_returnee",
    "num_previous_seasons", #
    # "votes_against_cumulative_by_previous_ep",
    "votes_against_last_3_eps", #
    # "correct_votes_cumulative_by_previous_ep",
    "times_in_danger", #
    "final_n", #
    # "tribe_status_Merged",
    # "tribe_status_Original",
    # "tribe_status_Swapped",
    # "tribe_status_Swapped_2",
    #  "advantages_held", # ----
    # "individual_immunity_wins",
    # "has_advantage",
    # "confessional_share_last_ep", 
    "confessional_share_rolling_3",
    # "confessional_share_cumulative",
    # "elim_risk",
]

TARGET_COL = "won_season"

TRAIN_SEASONS = range(1, 41)
TEST_SEASONS = range(41, 51)

C = 0.5

# All columns that must be non-null before we can generate elim_risk
_BASE_FEATURES = [f for f in FEATURE_COLS if f != "elim_risk"]
_ALL_NEEDED = list(set(_BASE_FEATURES) | set(ELIM_FEATURE_COLS))


# --- Elimination risk scores ---

def _normalize_elim_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize elim_risk within each episode so scores sum to 1."""
    ep_sums = df.groupby(["season", "episode"])["elim_risk"].transform("sum")
    df["elim_risk"] = df["elim_risk"] / ep_sums
    return df


def add_elim_risk_oof(train_df: pd.DataFrame) -> pd.DataFrame:
    """Generate out-of-fold elimination risk scores using an expanding window.

    For each season S in train_df, trains the elimination model on all
    seasons prior to S and predicts on S. Season 1 (no prior data) gets
    uniform 1/N scores. This mirrors real-world deployment and avoids
    data leakage from in-sample predictions.
    """
    result = train_df.copy()
    result["elim_risk"] = np.nan

    seasons = sorted(train_df["season"].unique())

    for i, season in enumerate(seasons):
        season_mask = result["season"] == season
        prior_seasons = seasons[:i]

        # No prior seasons to train on — assign uniform 1/N
        if not prior_seasons:
            n_per_ep = result.loc[season_mask].groupby(
                ["season", "episode"]
            )["season"].transform("count")
            result.loc[season_mask, "elim_risk"] = 1.0 / n_per_ep
            continue

        # Train elim model on seasons 1..S-1, predict on season S
        prior_data = train_df[train_df["season"].isin(prior_seasons)]
        elim_model, elim_scaler = train_elim_model(prior_data)

        X = elim_scaler.transform(result.loc[season_mask, ELIM_FEATURE_COLS])
        probs = elim_model.predict_proba(X)[:, 1]
        result.loc[season_mask, "elim_risk"] = probs

    return _normalize_elim_risk(result)


def add_elim_risk(train_df: pd.DataFrame, predict_df: pd.DataFrame) -> pd.DataFrame:
    """Train elimination model on train_df, add normalized P(eliminated) to predict_df."""
    elim_model, elim_scaler = train_elim_model(train_df)
    X = elim_scaler.transform(predict_df[ELIM_FEATURE_COLS])
    probs = elim_model.predict_proba(X)[:, 1]

    result = predict_df.copy()
    result["elim_risk"] = probs
    return _normalize_elim_risk(result)


# --- Training ---

def train_model(
    train: pd.DataFrame,
    C: float = C,
    feature_cols: list[str] | None = None,
) -> tuple[LogisticRegression, StandardScaler]:
    """Train logistic regression with balanced class weights (lbfgs solver)."""
    features = feature_cols or FEATURE_COLS
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    y_train = train[TARGET_COL]

    model = LogisticRegression(
        C=C, class_weight="balanced", solver="lbfgs", max_iter=5000,
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

    Returns overall metrics and a predictions DataFrame. 
    """
    features = feature_cols or FEATURE_COLS
    X_test = scaler.transform(test[features])
    probs = model.predict_proba(X_test)[:, 1]  # P(won_season)

    preds = test[["season", "episode", "castaway_id", "castaway", TARGET_COL]].copy()
    preds["prob_win"] = probs

    # Normalize within each episode so probabilities sum to 1
    episode_sums = preds.groupby(["season", "episode"])["prob_win"].transform("sum")
    preds["prob_win"] = preds["prob_win"] / episode_sums

    # --- Per-episode winner rank ---
    # For each episode, rank the actual winner among remaining players.
    # Rank 1 = model's top pick. Lower is better.
    rng = np.random.default_rng(42)
    ranks = []
    top1_correct = 0
    top3_correct = 0
    total = 0

    for (_season, _episode), group in preds.groupby(["season", "episode"]):
        if group[TARGET_COL].sum() == 0:
            continue
        # Rank by descending prob_win (break ties randomly)
        noisy_probs = group["prob_win"] + rng.uniform(0, 1e-10, len(group))
        group = group.assign(rank=noisy_probs.rank(ascending=False).astype(int))
        winner_rank = group.loc[group[TARGET_COL] == 1, "rank"].values[0]
        ranks.append(winner_rank)
        if winner_rank == 1:
            top1_correct += 1
        if winner_rank <= 3:
            top3_correct += 1
        total += 1

    mean_winner_rank = np.mean(ranks) if ranks else float("nan")
    top1_accuracy = top1_correct / total if total > 0 else 0
    top3_accuracy = top3_correct / total if total > 0 else 0

    # --- Brier score ---
    brier = brier_score_loss(preds[TARGET_COL], preds["prob_win"])

    # --- Baselines ---
    # Random guess: winner rank = (N+1)/2 on average
    episode_sizes = preds.groupby(["season", "episode"]).size()
    baseline_mean_rank = ((episode_sizes + 1) / 2).mean()
    baseline_top1 = (1 / episode_sizes).mean()
    baseline_probs = 1 / preds.groupby(["season", "episode"])["season"].transform("count")
    baseline_brier = brier_score_loss(preds[TARGET_COL], baseline_probs)

    return {
        "mean_winner_rank": mean_winner_rank,
        "top1_accuracy": top1_accuracy,
        "top3_accuracy": top3_accuracy,
        "baseline_mean_rank": baseline_mean_rank,
        "baseline_top1": baseline_top1,
        "brier_score": brier,
        "baseline_brier": baseline_brier,
        "n_test_episodes": total,
        "predictions": preds,
    }


def winner_rank_detail(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute the model's rank for the actual winner in each (season, episode).

    Returns one row per episode with winner_rank, baseline_rank, and
    rank_improvement (positive = model ranked winner higher than random).
    """
    rng = np.random.default_rng(42)

    rows = []
    for (season, episode), group in predictions.groupby(["season", "episode"]):
        if group[TARGET_COL].sum() == 0:
            continue
        # Tiny noise breaks ties randomly instead of by row order
        noisy_probs = group["prob_win"] + rng.uniform(0, 1e-10, len(group))
        ranked = group.assign(rank=noisy_probs.rank(ascending=False).astype(int))
        winner_rank = ranked.loc[ranked[TARGET_COL] == 1, "rank"].values[0]
        n_players = len(group)
        rows.append({
            "season": season,
            "episode": episode,
            "winner_rank": winner_rank,
            "n_players": n_players,
            "baseline_rank": (n_players + 1) / 2,  # expected rank under random
            "rank_improvement": (n_players + 1) / 2 - winner_rank,
            "top1_correct": int(winner_rank == 1),
            "top3_correct": int(winner_rank <= 3),
            "baseline_top1": 1 / n_players,
        })

    return pd.DataFrame(rows)


def metrics_by_episode_number(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate winner_rank_detail by episode number.

    Returns one row per episode number with mean accuracy metrics,
    standard errors, and how many seasons contributed data.
    """
    detail = winner_rank_detail(predictions)

    # Average across seasons at each episode number
    by_episode = detail.groupby("episode").agg(
        mean_winner_rank=("winner_rank", "mean"),
        se_winner_rank=("winner_rank", "sem"),
        mean_baseline_rank=("baseline_rank", "mean"),
        top1_accuracy=("top1_correct", "mean"),
        se_top1=("top1_correct", "sem"),
        top3_accuracy=("top3_correct", "mean"),
        baseline_top1=("baseline_top1", "mean"),
        mean_n_players=("n_players", "mean"),
        n_seasons=("season", "nunique"),
    ).reset_index()

    return by_episode


# --- Full pipeline ---

def _make_train_and_evaluate(feature_cols: list[str] | None = None, **kwargs):
    """Create a train-and-evaluate callback.

    Generates elimination risk scores fresh for each fold, then trains the win model.
    """
    def fn(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        train_df = add_elim_risk_oof(train_df)
        test_df = add_elim_risk(train_df, test_df)
        model, scaler = train_model(train_df, feature_cols=feature_cols, **kwargs)
        return predict_and_evaluate(model, scaler, test_df, feature_cols=feature_cols)
    return fn


def train_eval_pipeline(df: pd.DataFrame) -> dict:
    """Run the full pipeline: preprocess -> split -> generate risk scores -> train -> evaluate."""
    df = preprocess(df, _ALL_NEEDED)
    train, test = split_by_season(df, TRAIN_SEASONS, TEST_SEASONS)

    train = add_elim_risk_oof(train)
    test = add_elim_risk(train, test)

    print(f"Train: {len(train):,} rows ({train['season'].nunique()} seasons)")
    print(f"Test:  {len(test):,} rows ({test['season'].nunique()} seasons)")
    print(f"Features: {FEATURE_COLS}")
    print()

    model, scaler = train_model(train)
    results = predict_and_evaluate(model, scaler, test)

    print(f"Mean winner rank: {results['mean_winner_rank']:.2f} (model) | {results['baseline_mean_rank']:.2f} (baseline)")
    print(f"Top-1 accuracy:   {results['top1_accuracy']:.1%} (model) | {results['baseline_top1']:.1%} (baseline)")
    print(f"Top-3 accuracy:   {results['top3_accuracy']:.1%}")
    print(f"Brier score:      {results['brier_score']:.4f} (model) | {results['baseline_brier']:.4f} (baseline)")
    print(f"({results['n_test_episodes']} test episodes)")

    coef_tuples = list(zip(FEATURE_COLS, model.coef_[0]))
    coef_tuples_sorted = sorted(coef_tuples, key=lambda x: abs(x[1]), reverse=True)
    print(f"\nFeature coefficients (ordered by absolute value):")
    for name, coef in coef_tuples_sorted:
        print(f"  {name:45s} {coef:+.4f}")

    return results


def cross_validate(df: pd.DataFrame) -> dict:
    """Run expanding-window cross-validation and print results."""
    df = preprocess(df, _ALL_NEEDED)

    print(f"Features: {FEATURE_COLS}")
    print(f"Running expanding-window CV...\n")

    cv_results = expanding_window_cv(df, _make_train_and_evaluate(feature_cols=FEATURE_COLS))

    for fold in cv_results["folds"]:
        print(f"  Train {fold['fold_train_seasons']:>5s} | Test {fold['fold_test_seasons']:>5s} | "
              f"Winner rank: {fold['mean_winner_rank']:.2f} (baseline {fold['baseline_mean_rank']:.2f}) | "
              f"Top-1: {fold['top1_accuracy']:.1%} | "
              f"Episodes: {fold['n_test_episodes']}")

    mean = cv_results["mean"]
    print(f"\n  Mean winner rank: {mean['mean_winner_rank']:.2f} (baseline {mean['baseline_mean_rank']:.2f}) "
          f"over {cv_results['n_folds']} folds")
    print(f"  Mean Top-1: {mean['top1_accuracy']:.1%} (baseline {mean['baseline_top1']:.1%})")
    print(f"  Mean Brier: {mean['brier_score']:.4f} (baseline {mean['baseline_brier']:.4f})")

    return cv_results


def tune_hyperparameters(df: pd.DataFrame) -> float:
    """Grid search over C using expanding-window CV.

    Returns best_C.
    """
    df = preprocess(df, _ALL_NEEDED)

    C_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

    print(f"Tuning over {len(C_values)} C values\n")

    results = []
    for C in C_values:
        cv = expanding_window_cv(df, _make_train_and_evaluate(feature_cols=FEATURE_COLS, C=C))
        mean_rank = cv["mean"]["mean_winner_rank"]
        mean_brier = cv["mean"]["brier_score"]
        results.append({"C": C, "mean_winner_rank": mean_rank, "brier": mean_brier})
        print(f"  C={C:<6} -> rank={mean_rank:.2f}  brier={mean_brier:.4f}")

    results_sorted = sorted(results, key=lambda r: r["mean_winner_rank"])
    best = results_sorted[0]
    print(f"\nBest: C={best['C']} "
          f"-> rank={best['mean_winner_rank']:.2f}, brier={best['brier']:.4f}")

    return best["C"]


def run_forward_selection(df: pd.DataFrame) -> dict:
    """Run forward feature selection and print results."""
    df = preprocess(df, _ALL_NEEDED)

    def make_cv_callback(features: list[str]):
        return _make_train_and_evaluate(feature_cols=features)

    print(f"Candidates: {FEATURE_COLS}\n")
    results = forward_selection(
        df, FEATURE_COLS, make_cv_callback,
        metric="mean_winner_rank", higher_is_better=False,
    )

    print(f"\nSelected {len(results['selected_features'])} features "
          f"(from {len(FEATURE_COLS)} candidates):")
    for f in results["selected_features"]:
        print(f"  - {f}")
    print(f"\nBest CV winner rank: {results['best_score']:.2f}")

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
