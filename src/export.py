"""
App data export: combine the win and elimination models into the per-season JSON
the web app consumes. The orchestration layer above both model modules — the only
place that runs them together and knows the app's JSON format.

Usage:
    uv run python -m src.export              # export every season
    uv run python -m src.export --season 50  # export a single season
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.load import load_data
from src.features.build import build_modeling_table
from src.models.utils import preprocess
from src.models.win import (
    train_model as train_win_model,
    predict as win_predict,
    FEATURE_COLS as WIN_FEATURE_COLS,
)
from src.models.elimination import (
    train_model as train_elim_model,
    FEATURE_COLS as ELIM_FEATURE_COLS,
)

# Every exported row must have non-null inputs for BOTH models.
_ALL_NEEDED = list(set(WIN_FEATURE_COLS) | set(ELIM_FEATURE_COLS))


# --- Elimination risk scores ---

def _normalize_elim_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize elim_risk within each episode so scores sum to 1 (for the app's
    elimination panel)."""
    ep_sums = df.groupby(["season", "episode"])["elim_risk"].transform("sum")
    df["elim_risk"] = df["elim_risk"] / ep_sums
    return df


def add_elim_risk(train_df: pd.DataFrame, predict_df: pd.DataFrame) -> tuple[pd.DataFrame, object, object]:
    """Train elimination model on train_df, add normalized P(eliminated) to predict_df.

    Returns (predict_df_with_risk, elim_model, elim_scaler).
    """
    elim_model, elim_scaler = train_elim_model(train_df)
    X = elim_scaler.transform(predict_df[ELIM_FEATURE_COLS])
    probs = elim_model.predict_proba(X)[:, 1]

    result = predict_df.copy()
    result["elim_risk"] = probs
    return _normalize_elim_risk(result), elim_model, elim_scaler


# --- JSON export ---

def _export_season_json(
    preds: pd.DataFrame,
    target: pd.DataFrame,
    target_season: int,
    win_model,
    win_scaler,
    elim_model,
    elim_scaler,
) -> None:
    """Build and save the season JSON for the web app.

    Aggregates per-episode predictions into per-player records with feature
    values, and includes model coefficients/scaler info for both models.
    """
    elim_display_features = [f for f in ELIM_FEATURE_COLS if f not in ("final_n",)]
    all_export_features = list(WIN_FEATURE_COLS) + [f for f in elim_display_features if f not in WIN_FEATURE_COLS]

    # Merge feature values onto predictions
    feature_df = target[["season", "episode", "castaway_id"] + all_export_features].copy()
    preds = preds.merge(feature_df, on=["season", "episode", "castaway_id"], how="left")

    # Aggregate by castaway: one record per player with lists of per-episode values
    agg_cols = {
        'episode': lambda x: list(x),
        'prob_win': lambda x: list(x),
        'prob_eliminated': lambda x: list(x),
        'eliminated_this_episode': lambda x: list(x),
    }
    for feat in all_export_features:
        agg_cols[feat] = lambda x, f=feat: list(x)

    agg = preds.groupby(["season", "castaway_id", "castaway", "won_season"]).agg(agg_cols).reset_index()

    players = []
    for _, row in agg.iterrows():
        players.append({
            "season": int(row["season"]),
            "castaway_id": row["castaway_id"],
            "castaway": row["castaway"],
            "won_season": int(row["won_season"]),
            "episode": row["episode"],
            "prob_win": row["prob_win"],
            "prob_eliminated": row["prob_eliminated"],
            "eliminated_this_episode": row["eliminated_this_episode"],
            "win_features": {f: row[f] for f in WIN_FEATURE_COLS},
            "elim_features": {f: row[f] for f in elim_display_features},
        })

    def _model_info(model, scaler, features):
        return {
            "features": features,
            "coefficients": model.coef_[0].tolist(),
            "intercept": model.intercept_[0],
            "scaler_means": scaler.mean_.tolist(),
            "scaler_stds": scaler.scale_.tolist(),
        }

    # For the elimination model, only export the display subset of features
    elim_coef_indices = [ELIM_FEATURE_COLS.index(f) for f in elim_display_features]
    elim_info = {
        "features": elim_display_features,
        "coefficients": [elim_model.coef_[0][i] for i in elim_coef_indices],
        "intercept": elim_model.intercept_[0],
        "scaler_means": [elim_scaler.mean_[i] for i in elim_coef_indices],
        "scaler_stds": [elim_scaler.scale_[i] for i in elim_coef_indices],
    }

    season_data = {
        "win_model_info": _model_info(win_model, win_scaler, list(WIN_FEATURE_COLS)),
        "elim_model_info": elim_info,
        "players": players,
    }

    project_root = Path(__file__).parent.parent
    out_dir = project_root / "app" / "public" / "data" / "seasons"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"season_{target_season}.json", "w") as f:
        json.dump(season_data, f)

    # Update index of available seasons
    seasons = sorted(int(f.stem.split("_")[1]) for f in out_dir.glob("season_*.json"))
    with open(out_dir / "index.json", "w") as f:
        json.dump(seasons, f)


# --- Orchestration ---

def season_predictions(df: pd.DataFrame, target_season: int) -> tuple[pd.DataFrame, dict]:
    """Train both models on all prior seasons and return per-player predictions for
    `target_season` — win probability plus elimination probability — WITHOUT writing
    any files.

    Returns (preds, fit), where `fit` carries the trained models/scalers and the
    scored target rows that `_export_season_json` needs. For analysis (e.g. the
    notebook) you can ignore `fit`: `preds, _ = season_predictions(df, s)`.
    """
    df = preprocess(df, _ALL_NEEDED)
    train = df[df["season"] < target_season]
    target = df[df["season"] == target_season]

    if target.empty:
        raise ValueError(f"No data found for season {target_season}")

    # Score the target with the elimination model for the app's elimination panel.
    target, elim_model, elim_scaler = add_elim_risk(train, target)
    win_model, win_scaler = train_win_model(train)
    preds = win_predict(win_model, win_scaler, target)

    # win_predict returns win-side columns only; attach the elimination model's
    # per-player probability (already normalized within episode) here.
    preds = preds.merge(
        target[["season", "episode", "castaway_id", "elim_risk"]]
        .rename(columns={"elim_risk": "prob_eliminated"}),
        on=["season", "episode", "castaway_id"], how="left",
    )

    fit = {
        "target": target,
        "win_model": win_model,
        "win_scaler": win_scaler,
        "elim_model": elim_model,
        "elim_scaler": elim_scaler,
        "n_train": len(train),
    }
    return preds, fit


def export_season(df: pd.DataFrame, target_season: int) -> pd.DataFrame:
    """Train both models on all prior seasons, score `target_season`, and write the
    app JSON. Returns the win predictions.
    """
    preds, fit = season_predictions(df, target_season)

    print(f"Training on seasons 1-{target_season - 1} ({fit['n_train']:,} rows), "
          f"predicting season {target_season} ({len(fit['target']):,} rows)")

    _export_season_json(
        preds, fit["target"], target_season,
        fit["win_model"], fit["win_scaler"], fit["elim_model"], fit["elim_scaler"],
    )
    return preds


def export_all_seasons(df: pd.DataFrame) -> None:
    """Export the app JSON for every season from 2 to the latest available."""
    max_season = int(df["season"].max())
    for season in range(2, max_season + 1):
        print(f"\n=== Season {season}/{max_season} ===")
        export_season(df, season)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, metavar="SEASON",
                        help="Export a single season (default: all seasons)")
    args = parser.parse_args()

    data = load_data()
    df = build_modeling_table(data)

    if args.season:
        export_season(df, args.season)
    else:
        export_all_seasons(df)
