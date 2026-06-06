"""Shared evaluation utilities: cross-validation, bootstrap CIs, feature selection."""

import pandas as pd
import numpy as np
from typing import Callable


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    cluster_col: str,
    stat_fn: Callable[[pd.DataFrame], dict],
    n_boot: int = 2000,
    ci: float = 95,
    seed: int = 42,
) -> dict:
    """Cluster bootstrap CIs.

    Resamples whole clusters (e.g. seasons) with replacement and recomputes
    stat_fn each time, preserving within-cluster correlation for honest CIs on
    pooled metrics. stat_fn: DataFrame -> dict of scalar metrics. Returns
    {metric: {"point", "lo", "hi"}}.
    """
    rng = np.random.default_rng(seed)
    clusters = df[cluster_col].unique()
    groups = {c: g for c, g in df.groupby(cluster_col)}

    point = stat_fn(df)
    boot: dict[str, list] = {k: [] for k in point}

    for _ in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        resampled = pd.concat([groups[c] for c in sampled], ignore_index=True)
        res = stat_fn(resampled)
        for k, v in res.items():
            boot[k].append(v)

    alpha = (100 - ci) / 2
    out = {}
    for k in point:
        arr = np.asarray(boot[k], dtype=float)
        lo, hi = np.percentile(arr, [alpha, 100 - alpha])
        out[k] = {"point": point[k], "lo": lo, "hi": hi}

    return out


def _favorite_among_finalists(
    preds: pd.DataFrame,
    final_players: list,
    winner_id,
) -> tuple[float, int]:
    """Rank a season's final-episode players by the season-long favorite criterion
    (most episodes at #1; ties: lower mean rank, then higher finale prob).

    Returns (winner's rank, n_finalists); rank 1 = the model's pick, NaN if the
    winner isn't among the finalists.
    """
    p = preds[["episode", "castaway_id", "prob_win"]].copy()
    # Rank within each episode (1 = most likely to win that episode).
    p["rank"] = p.groupby("episode")["prob_win"].rank(ascending=False, method="min")
    p["is_top"] = (p["rank"] == 1.0).astype(int)

    frac1 = p.groupby("castaway_id")["is_top"].mean()
    mean_rank = p.groupby("castaway_id")["rank"].mean()
    max_ep = p["episode"].max()
    finale_prob = p[p["episode"] == max_ep].groupby("castaway_id")["prob_win"].first()

    candidates = [c for c in final_players if c in frac1.index]
    # Sort by frac1 desc, then mean_rank asc, then finale_prob desc.
    candidates.sort(
        key=lambda c: (-frac1.get(c, 0.0), mean_rank.get(c, np.inf), -finale_prob.get(c, 0.0))
    )

    n_finalists = len(candidates)
    if winner_id not in candidates:
        return float("nan"), n_finalists
    return float(candidates.index(winner_id) + 1), n_finalists


def oob_refit_bootstrap(
    df: pd.DataFrame,
    train_fn: Callable[[pd.DataFrame], tuple],
    predict_fn: Callable[..., pd.DataFrame],
    feature_cols: list[str],
    target_col: str = "won_season",
    n_boot: int = 1000,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Out-of-bag refit bootstrap for the descriptive winner analysis.

    Each iteration resamples seasons with replacement (in-bag), refits once, and
    scores the left-out (out-of-bag) seasons. An OOB season is never in its own
    training set, so each prediction is held-out; future seasons are allowed in
    training, making this descriptive, not predictive. ~n_boot fits total.

    train_fn: (train_df) -> (model, scaler); model must expose .coef_.
    predict_fn: (model, scaler, df) -> predictions with a normalized prob_win.

    Returns a dict with:
      - occurrences: per (OOB season, iteration) the winner's rank among
        finalists, n_finalists, and whether the winner was the pick (rank 1).
      - coefficients: standardized coefficient vector per refit.
      - n_boot, seasons.
    """
    rng = np.random.default_rng(seed)
    seasons = sorted(df["season"].unique())
    by_season = {s: g.copy() for s, g in df.groupby("season")}

    # Static per-season info: final-episode players and the winner.
    final_players = {}
    winners = {}
    for s, g in by_season.items():
        max_ep = g["episode"].max()
        final_players[s] = list(g.loc[g["episode"] == max_ep, "castaway_id"].unique())
        w = g.loc[g[target_col] == 1, "castaway_id"].unique()
        winners[s] = w[0] if len(w) else None

    coef_rows = []
    occ_rows = []
    step = max(1, n_boot // 10)

    for b in range(n_boot):
        sampled = rng.choice(seasons, size=len(seasons), replace=True)
        in_bag = set(sampled.tolist())
        oob = [s for s in seasons if s not in in_bag]

        train_df = pd.concat([by_season[s] for s in sampled], ignore_index=True)
        model, scaler = train_fn(train_df)
        coef_rows.append(np.asarray(model.coef_[0], dtype=float))

        for s in oob:
            if winners[s] is None:
                continue
            preds = predict_fn(model, scaler, by_season[s])
            rank, n_fin = _favorite_among_finalists(preds, final_players[s], winners[s])
            occ_rows.append((s, winners[s], rank, n_fin, int(rank == 1.0)))

        if verbose and (b + 1) % step == 0:
            print(f"  {b + 1}/{n_boot} refits")

    coefficients = pd.DataFrame(coef_rows, columns=list(feature_cols))
    occurrences = pd.DataFrame(
        occ_rows,
        columns=["season", "winner_id", "winner_rank", "n_finalists", "pick_is_winner"],
    )
    return {
        "coefficients": coefficients,
        "occurrences": occurrences,
        "n_boot": n_boot,
        "seasons": seasons,
    }


def summarize_winner_picks(
    result: dict,
    df: pd.DataFrame,
    target_col: str = "won_season",
) -> pd.DataFrame:
    """Aggregate `oob_refit_bootstrap` occurrences into a per-winner table, sorted
    by pick_rate (expected winners first, against-the-odds last).

    Columns: season, winner, pick_rate (fraction of OOB refits the winner was the
    favorite among finalists), mean_rank, mean_finalists, n_oob (OOB refit count =
    the pick_rate denominator), winner_id.
    """
    occ = result["occurrences"]
    names = df[df[target_col] == 1].groupby("season")["castaway"].first()

    agg = (
        occ.groupby(["season", "winner_id"])
        .agg(
            n_oob=("pick_is_winner", "size"),
            pick_rate=("pick_is_winner", "mean"),
            mean_rank=("winner_rank", "mean"),
            mean_finalists=("n_finalists", "mean"),
        )
        .reset_index()
    )
    agg["winner"] = agg["season"].map(names)
    cols = [
        "season", "winner", "pick_rate", "mean_rank",
        "mean_finalists", "n_oob", "winner_id",
    ]
    return agg[cols].sort_values("pick_rate", ascending=False).reset_index(drop=True)


def loso_winner_margins(
    df: pd.DataFrame,
    train_fn: Callable[[pd.DataFrame], tuple],
    predict_fn: Callable[..., pd.DataFrame],
    target_col: str = "won_season",
) -> pd.DataFrame:
    """Single leave-one-season-out fit per season.

    For each season, fit on all other seasons, score the held-out season, and
    measure the winner's season-long-favorite margin = the winner's fraction of
    episodes ranked #1 (over the whole field) minus the best other finalist's.
    """
    seasons = sorted(df["season"].unique())
    by_season = {s: g for s, g in df.groupby("season")}
    names = df[df[target_col] == 1].groupby("season")["castaway"].first()

    rows = []
    for s in seasons:
        model, scaler = train_fn(df[df["season"] != s])
        preds = predict_fn(model, scaler, by_season[s])

        max_ep = by_season[s]["episode"].max()
        finalists = list(
            by_season[s].loc[by_season[s]["episode"] == max_ep, "castaway_id"].unique()
        )
        w = by_season[s].loc[by_season[s][target_col] == 1, "castaway_id"].unique()
        if len(w) == 0:
            continue
        winner = w[0]

        # Fraction of episodes each player was the field's #1 pick.
        p = preds[["episode", "castaway_id", "prob_win"]].copy()
        p["rk"] = p.groupby("episode")["prob_win"].rank(ascending=False, method="min")
        frac1 = p.assign(top=(p["rk"] == 1).astype(int)).groupby("castaway_id")["top"].mean()
        fin = {c: float(frac1.get(c, 0.0)) for c in finalists}
        wf = fin.get(winner, 0.0)
        others = [v for c, v in fin.items() if c != winner]
        best_other = max(others) if others else 0.0

        rank, n_fin = _favorite_among_finalists(preds, finalists, winner)
        rows.append({
            "season": s,
            "winner": names.get(s),
            "winner_frac1": wf,
            "best_other_frac1": best_other,
            "margin": wf - best_other,
            "loso_rank": rank,
            "n_finalists": n_fin,
        })

    return pd.DataFrame(rows).sort_values("margin", ascending=False).reset_index(drop=True)


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
