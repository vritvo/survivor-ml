"""Shared evaluation utilities: cross-validation, bootstrap CIs, feature selection."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Callable

# Columns in build_modeling_table() that are not model inputs.
MODELING_TABLE_META_COLS = frozenset({
    "season",
    "episode",
    "castaway_id",
    "castaway",
    "tribe",
    "order",
    "eliminated_this_episode",
    "won_season",
    "num_votes_received",
    "votes_against_cumulative",
})

# Last-episode univariate snapshots where the feature mostly encodes "survived to
# finale" or endgame composition, not a substantive win signal on its own.
UNIVARIATE_MECHANICAL_EXCLUDE = frozenset({
    "final_n",
    "age_rank",
    "tribe_status_Merged",
    "tribe_status_Original",
    "tribe_status_Swapped",
    "tribe_status_Swapped_2",
})

# Tribe dummies only — still excluded for all-episode runs (structural, rarely interpretable).
UNIVARIATE_TRIBE_EXCLUDE = frozenset({
    "tribe_status_Merged",
    "tribe_status_Original",
    "tribe_status_Swapped",
    "tribe_status_Swapped_2",
})


def modeling_feature_cols(
    df: pd.DataFrame,
    exclude: set[str] | frozenset | None = None,
) -> list[str]:
    """Feature columns present in a modeling table (excludes IDs, targets, helpers)."""
    skip = MODELING_TABLE_META_COLS | set(exclude or [])
    return [
        c for c in df.columns
        if c not in skip and pd.api.types.is_numeric_dtype(df[c])
    ]


def player_season_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, player): their last in-game episode."""
    ordered = df.sort_values(["season", "castaway_id", "episode"])
    return ordered.groupby(["season", "castaway_id"], as_index=False).last()


def _univariate_logistic_coefs(
    sub: pd.DataFrame,
    features: list[str],
    target_col: str = "won_season",
) -> dict[str, float]:
    """Standardized univariate logistic coefficients (one feature at a time)."""
    return _partial_logistic_coefs(sub, features, control_cols=[], target_col=target_col)


def _partial_logistic_coefs(
    sub: pd.DataFrame,
    features: list[str],
    control_cols: list[str],
    target_col: str = "won_season",
) -> dict[str, float]:
    """Standardized partial logistic coefs: one feature at a time + shared controls."""
    out: dict[str, float] = {}
    for feat in features:
        cols = [feat, *control_cols]
        valid = sub[cols + [target_col]].dropna()
        if len(valid) < 20 or valid[feat].std() == 0 or valid[target_col].nunique() < 2:
            out[feat] = float("nan")
            continue
        X = StandardScaler().fit_transform(valid[cols])
        y = valid[target_col].values
        try:
            model = LogisticRegression(
                class_weight="balanced",
                solver="lbfgs",
                max_iter=2000,
                C=1.0,
            )
            model.fit(X, y)
            out[feat] = float(model.coef_[0][0])
        except (ValueError, np.linalg.LinAlgError):
            out[feat] = float("nan")
    return out


def summarize_univariate_win_associations(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "won_season",
    cluster_col: str = "season",
    unit: str = "all_episodes",
    exclude_mechanical: bool = True,
    n_boot: int = 2000,
    ci: float = 95,
    seed: int = 42,
) -> pd.DataFrame:
    """Marginal feature ↔ winning associations with season-cluster bootstrap CIs.

    Each feature gets its own standardized univariate logistic regression
    (positive coef → higher feature values associated with winning). CIs resample
    whole seasons.

    **unit** (default ``"all_episodes"``):
        - ``"all_episodes"`` — every player-episode row while in game (same rows
          as the win model trains on). Aligns with CV / forward selection.
        - ``"last_episode"`` — one row per player at their last in-game episode.
          Cumulative features at exit conflate survival length with the signal.

    By default excludes tribe-status dummies; ``last_episode`` also drops
    ``final_n`` and ``age_rank``.
    """
    if unit == "last_episode":
        analysis_df = player_season_snapshot(df)
        extra = UNIVARIATE_MECHANICAL_EXCLUDE if exclude_mechanical else frozenset()
    elif unit == "all_episodes":
        analysis_df = df
        extra = UNIVARIATE_TRIBE_EXCLUDE if exclude_mechanical else frozenset()
    else:
        raise ValueError(f"unit must be 'all_episodes' or 'last_episode', got {unit!r}")

    if feature_cols is None:
        features = modeling_feature_cols(df, exclude=extra)
    else:
        features = list(feature_cols)

    def stat_fn(sub: pd.DataFrame) -> dict[str, float]:
        return _univariate_logistic_coefs(sub, features, target_col=target_col)

    point = stat_fn(analysis_df)
    boot = cluster_bootstrap_ci(
        analysis_df,
        cluster_col,
        stat_fn,
        n_boot=n_boot,
        ci=ci,
        seed=seed,
    )

    n_winner_ps = (
        analysis_df.groupby(["season", "castaway_id"])[target_col].max().sum()
    )
    rows = []
    for feat in features:
        p = point[feat]
        b = boot[feat]
        rows.append({
            "feature": feat,
            "coef": float(p) if not np.isnan(p) else float("nan"),
            "ci_lo": b["lo"],
            "ci_hi": b["hi"],
            "ci_excludes_zero": (b["lo"] > 0) or (b["hi"] < 0),
            "n_rows": int(analysis_df[feat].notna().sum()),
            "n_winner_player_seasons": int(n_winner_ps),
            "unit": unit,
        })
    out = pd.DataFrame(rows)
    out["abs_coef"] = out["coef"].abs()
    return out.sort_values("abs_coef", ascending=False, na_position="last").reset_index(drop=True)


def summarize_stage_adjusted_win_associations(
    df: pd.DataFrame,
    control_cols: list[str] | None = None,
    feature_cols: list[str] | None = None,
    target_col: str = "won_season",
    cluster_col: str = "season",
    exclude_mechanical: bool = True,
    n_boot: int = 2000,
    ci: float = 95,
    seed: int = 42,
) -> pd.DataFrame:
    """Partial feature ↔ winning associations with fixed controls (default: ``final_n``).

    Each feature gets its own logistic regression with the control(s) included
    (positive coef → higher feature values associated with winning, holding controls
    fixed). CIs resample whole seasons. Excludes control columns from the feature list.
    """
    controls = list(control_cols or ["final_n"])
    extra = UNIVARIATE_TRIBE_EXCLUDE | frozenset(controls)
    if exclude_mechanical:
        extra = extra | UNIVARIATE_MECHANICAL_EXCLUDE

    if feature_cols is None:
        features = modeling_feature_cols(df, exclude=extra)
    else:
        features = [f for f in feature_cols if f not in controls]

    analysis_df = df

    def stat_fn(sub: pd.DataFrame) -> dict[str, float]:
        return _partial_logistic_coefs(
            sub, features, control_cols=controls, target_col=target_col,
        )

    point = stat_fn(analysis_df)
    boot = cluster_bootstrap_ci(
        analysis_df,
        cluster_col,
        stat_fn,
        n_boot=n_boot,
        ci=ci,
        seed=seed,
    )

    n_winner_ps = (
        analysis_df.groupby(["season", "castaway_id"])[target_col].max().sum()
    )
    rows = []
    for feat in features:
        p = point[feat]
        b = boot[feat]
        rows.append({
            "feature": feat,
            "coef": float(p) if not np.isnan(p) else float("nan"),
            "ci_lo": b["lo"],
            "ci_hi": b["hi"],
            "ci_excludes_zero": (b["lo"] > 0) or (b["hi"] < 0),
            "n_rows": int(analysis_df[feat].notna().sum()),
            "n_winner_player_seasons": int(n_winner_ps),
            "controls": "+".join(controls),
        })
    out = pd.DataFrame(rows)
    out["abs_coef"] = out["coef"].abs()
    return out.sort_values("abs_coef", ascending=False, na_position="last").reset_index(drop=True)


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


def _rank_finalists(preds: pd.DataFrame, final_players: list) -> dict:
    """Rank final-episode players by the season-long favorite criterion.

    Returns {castaway_id: rank} with rank 1 = the model's pick (most episodes at
    #1; ties: lower mean season rank, then higher finale prob).
    """
    p = preds[["episode", "castaway_id", "prob_win"]].copy()
    p["rank"] = p.groupby("episode")["prob_win"].rank(ascending=False, method="min")
    p["is_top"] = (p["rank"] == 1.0).astype(int)

    frac1 = p.groupby("castaway_id")["is_top"].mean()
    mean_rank = p.groupby("castaway_id")["rank"].mean()
    max_ep = p["episode"].max()
    finale_prob = p[p["episode"] == max_ep].groupby("castaway_id")["prob_win"].first()

    candidates = [c for c in final_players if c in frac1.index]
    candidates.sort(
        key=lambda c: (-frac1.get(c, 0.0), mean_rank.get(c, np.inf), -finale_prob.get(c, 0.0))
    )
    return {c: i + 1 for i, c in enumerate(candidates)}


def _favorite_among_finalists(
    preds: pd.DataFrame,
    final_players: list,
    winner_id,
) -> tuple[float, int]:
    """Return (winner's rank among finalists, n_finalists); NaN rank if not a finalist."""
    ranks = _rank_finalists(preds, final_players)
    if winner_id not in ranks:
        return float("nan"), len(ranks)
    return float(ranks[winner_id]), len(ranks)


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


def oob_coefficient_bootstrap(
    df: pd.DataFrame,
    train_fn: Callable[[pd.DataFrame], tuple],
    feature_cols: list[str],
    n_boot: int = 1000,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Season-resampled refit bootstrap: standardized coefficient vectors only.

    Each iteration resamples whole seasons with replacement, refits once, and stores
    the model's standardized coefficient vector. No OOB scoring — for coef stability
    only (e.g. elimination model where winner-pick metrics do not apply).
    """
    rng = np.random.default_rng(seed)
    seasons = sorted(df["season"].unique())
    by_season = {s: g.copy() for s, g in df.groupby("season")}

    coef_rows = []
    step = max(1, n_boot // 10)

    for b in range(n_boot):
        sampled = rng.choice(seasons, size=len(seasons), replace=True)
        train_df = pd.concat([by_season[s] for s in sampled], ignore_index=True)
        model, _ = train_fn(train_df)
        coef_rows.append(np.asarray(model.coef_[0], dtype=float))

        if verbose and (b + 1) % step == 0:
            print(f"  {b + 1}/{n_boot} refits")

    coefficients = pd.DataFrame(coef_rows, columns=list(feature_cols))
    return {
        "coefficients": coefficients,
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


def summarize_coefficient_stability(
    result: dict,
    ci: float = 95,
    reference_coefs: pd.Series | None = None,
) -> pd.DataFrame:
    """Aggregate bootstrap standardized coefficients into a stability table.

    Uses the ``coefficients`` DataFrame from ``oob_refit_bootstrap`` or
    ``oob_coefficient_bootstrap`` (one row per
    refit, columns = feature names). Per feature:

    - **median** — central standardized coefficient across refits
    - **ci_lo / ci_hi** — percentile interval (default 95%)
    - **sign_stability** — share of refits with the same sign as the median
    - **rank_stability** — share of refits where |coef| rank is within ±1 of its
      median rank (1 = most important)

    Optional **reference** column compares to a single-fit coefficient vector
    (e.g. the deployed model trained on all data). Sorted by |median| descending.
    """
    coefs = result["coefficients"]
    alpha = (100 - ci) / 2
    abs_ranks = coefs.abs().rank(axis=1, ascending=False, method="min")

    rows = []
    for feat in coefs.columns:
        c = coefs[feat]
        med = float(c.median())
        lo, hi = np.percentile(c, [alpha, 100 - alpha])
        sign = int(np.sign(med)) if med != 0 else 0
        if sign == 0:
            sign_stab = float((c == 0).mean())
        else:
            sign_stab = float((np.sign(c) == sign).mean())

        ranks = abs_ranks[feat]
        med_rank = float(ranks.median())
        rank_stab = float((np.abs(ranks - med_rank) <= 1).mean())

        row = {
            "feature": feat,
            "median": med,
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "sign_stability": sign_stab,
            "rank_stability": rank_stab,
            "n_refits": len(c),
        }
        if reference_coefs is not None and feat in reference_coefs.index:
            row["reference"] = float(reference_coefs[feat])
        rows.append(row)

    out = pd.DataFrame(rows)
    out["abs_median"] = out["median"].abs()
    out["ci_excludes_zero"] = (out["ci_lo"] > 0) | (out["ci_hi"] < 0)
    return out.sort_values("abs_median", ascending=False).reset_index(drop=True)


def loso_finalist_frac1(
    df: pd.DataFrame,
    train_fn: Callable[[pd.DataFrame], tuple],
    predict_fn: Callable[..., pd.DataFrame],
    target_col: str = "won_season",
) -> pd.DataFrame:
    """Per-finalist season-long #1 share from a single LOSO fit per season.

    One row per (season, finalist): season, castaway_id, castaway, is_winner,
    frac1, prob_win_finale, favorite_rank (among finalists), n_finalists. Base
    table for winner-margin, dominant-losing-finalist, and calibration views.
    """
    seasons = sorted(df["season"].unique())
    by_season = {s: g for s, g in df.groupby("season")}

    rows = []
    for s in seasons:
        model, scaler = train_fn(df[df["season"] != s])
        preds = predict_fn(model, scaler, by_season[s])

        max_ep = by_season[s]["episode"].max()
        fin = (by_season[s].loc[by_season[s]["episode"] == max_ep,
                                ["castaway_id", "castaway", target_col]]
               .drop_duplicates("castaway_id"))
        finalists = fin["castaway_id"].tolist()

        p = preds[["episode", "castaway_id", "prob_win"]].copy()
        p["rk"] = p.groupby("episode")["prob_win"].rank(ascending=False, method="min")
        frac1 = p.assign(top=(p["rk"] == 1).astype(int)).groupby("castaway_id")["top"].mean()
        fav_rank = _rank_finalists(preds, finalists)
        finale_prob = (
            preds[preds["episode"] == max_ep]
            .drop_duplicates("castaway_id")
            .set_index("castaway_id")["prob_win"]
        )

        for _, r in fin.iterrows():
            cid = r["castaway_id"]
            rows.append({
                "season": s,
                "castaway_id": cid,
                "castaway": r["castaway"],
                "is_winner": int(r[target_col] == 1),
                "frac1": float(frac1.get(cid, 0.0)),
                "prob_win_finale": float(finale_prob.get(cid, np.nan)),
                "favorite_rank": fav_rank.get(cid, np.nan),
                "n_finalists": len(fin),
            })

    return pd.DataFrame(rows)


def summarize_winner_margins(finalist_frac1: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a `loso_finalist_frac1` table into per-winner margins.

    margin = winner's frac1 − best other finalist's frac1. Sorted by margin
    descending (expected winners first).
    """
    winners = finalist_frac1[finalist_frac1["is_winner"] == 1].copy()
    best_other = (
        finalist_frac1[finalist_frac1["is_winner"] == 0]
        .groupby("season")["frac1"]
        .max()
        .rename("best_other_frac1")
    )
    out = winners.merge(best_other, on="season", how="left")
    out["best_other_frac1"] = out["best_other_frac1"].fillna(0.0)
    out = out.rename(columns={"castaway": "winner", "frac1": "winner_frac1"})
    out["margin"] = out["winner_frac1"] - out["best_other_frac1"]
    out["loso_rank"] = out["favorite_rank"]
    cols = [
        "season", "winner", "winner_frac1", "best_other_frac1",
        "margin", "loso_rank", "n_finalists",
    ]
    return out[cols].sort_values("margin", ascending=False).reset_index(drop=True)


def loso_winner_margins(
    df: pd.DataFrame,
    train_fn: Callable[[pd.DataFrame], tuple],
    predict_fn: Callable[..., pd.DataFrame],
    target_col: str = "won_season",
    finalist_frac1: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Single LOSO fit per season — winner margin view.

    Convenience wrapper: runs `loso_finalist_frac1` (unless `finalist_frac1` is
    passed) then `summarize_winner_margins`. Pass a precomputed `finalist_frac1`
    to avoid refitting when building both winner and loser plots.
    """
    if finalist_frac1 is None:
        finalist_frac1 = loso_finalist_frac1(
            df, train_fn, predict_fn, target_col=target_col,
        )
    return summarize_winner_margins(finalist_frac1)


def calibration_bins(
    df: pd.DataFrame,
    prob_col: str = "prob_win_finale",
    outcome_col: str = "is_winner",
    group_col: str | None = None,
    bin_edges: list[float] | None = None,
    min_n: int = 3,
) -> pd.DataFrame:
    """Binned calibration table: mean predicted prob vs actual win rate per bin.

    Each row is one bin (optionally per group). Columns include bin_lo, bin_hi,
    bin_label (e.g. \"15–25%\"), n, mean_predicted (x-axis on the plot), and
    observed_rate (y-axis). Rows with n < min_n are dropped.

    Default bin_edges: [0, 0.15, 0.25, 0.35, 0.5, 1.0] — fixed probability
    ranges, same for table and plot.
    """
    edges = bin_edges if bin_edges is not None else [0, 0.15, 0.25, 0.35, 0.5, 1.0]

    def _bins(sub: pd.DataFrame, label: str) -> list[dict]:
        sub = sub.copy()
        sub["_bin"] = pd.cut(sub[prob_col], bins=edges, include_lowest=True)
        rows = []
        for interval, grp in sub.groupby("_bin", observed=True):
            if len(grp) < min_n:
                continue
            lo, hi = interval.left, interval.right
            rows.append({
                "group": label,
                "bin_lo": lo,
                "bin_hi": hi,
                "bin_label": f"{lo:.0%}–{hi:.0%}",
                "n": len(grp),
                "mean_predicted": grp[prob_col].mean(),
                "observed_rate": grp[outcome_col].mean(),
            })
        return rows

    if group_col is None:
        out = _bins(df, "All")
    else:
        out = []
        for g, sub in df.groupby(group_col):
            out.extend(_bins(sub, g))
    return pd.DataFrame(out)


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
