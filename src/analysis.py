"""
Reusable analysis helpers (e.g. for notebook) 
"""

import pandas as pd
import numpy as np
from scipy.stats import binomtest

from src.evaluate import player_season_snapshot

# Sensible bin edges for binned outcome-rate tables (see summarize_binned_outcome_rates).
FEATURE_BIN_EDGES: dict[str, list[float]] = {
    "age": [18, 25, 30, 35, 40, 45, 80],
    "num_previous_seasons": [-0.5, 0.5, 1.5, 2.5, 10],
}

FEATURE_BIN_LABELS: dict[str, list[str]] = {
    "age": ["18–24", "25–29", "30–34", "35–39", "40–44", "45+"],
    "num_previous_seasons": ["0", "1", "2", "3+"],
}

# Backward-compatible aliases
FEATURE_WIN_RATE_BIN_EDGES = FEATURE_BIN_EDGES
FEATURE_WIN_RATE_BIN_LABELS = FEATURE_BIN_LABELS


def wilson_ci(
    n_success: int,
    n_total: int,
    *,
    conf: float = 0.95,
) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if n_total == 0:
        return float("nan"), float("nan")
    interval = binomtest(n_success, n_total).proportion_ci(
        confidence_level=conf,
        method="wilson",
    )
    return interval.low, interval.high


def _default_feature_bin_edges(series: pd.Series) -> tuple[list[float], list[str] | None]:
    """Pick bin edges for a feature when none are specified."""
    clean = series.dropna()
    if clean.empty:
        return [], None

    uniques = np.sort(clean.unique())
    if len(uniques) <= 6 and np.all(np.equal(np.mod(uniques, 1), 0)):
        edges = [
            uniques[0] - 0.5,
            *[(uniques[i] + uniques[i + 1]) / 2 for i in range(len(uniques) - 1)],
            uniques[-1] + 0.5,
        ]
        labels = [str(int(v)) for v in uniques]
        return edges, labels

    quintiles = np.unique(np.quantile(clean, [0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    if len(quintiles) < 3:
        return quintiles.tolist(), None
    return quintiles.tolist(), None


def summarize_binned_outcome_rates(
    df: pd.DataFrame,
    feature_col: str,
    *,
    bin_edges: list[float] | None = None,
    bin_labels: list[str] | None = None,
    target_col: str = "won_season",
    player_season: bool = True,
    ci: float = 95,
    min_n: int = 5,
) -> pd.DataFrame:
    """Empirical outcome rate by feature bin (linearity / shape check).

    Bins a feature and computes the mean of ``target_col`` per bin with Wilson CIs.
    Not season-clustered — fine for a first look at functional form.

    **player_season** (default True):
        One row per castaway per season (last in-game episode). Good for season
        outcomes like ``won_season``; static features like age.

    **player_season=False**:
        All modeling-table rows (player-episodes). Aligns with the elimination
        model, which predicts ``eliminated_this_episode`` each episode.

    Default bins use ``FEATURE_BIN_EDGES`` when defined; otherwise integer
    categories (≤6 unique values) or quintiles.
    """
    analysis_df = player_season_snapshot(df) if player_season else df.copy()
    if feature_col not in analysis_df.columns:
        raise KeyError(f"Column not found: {feature_col}")

    valid = analysis_df[[feature_col, target_col]].dropna()
    if valid.empty:
        return pd.DataFrame()

    edges = bin_edges
    labels = bin_labels
    if edges is None:
        if feature_col in FEATURE_BIN_EDGES:
            edges = FEATURE_BIN_EDGES[feature_col]
            labels = labels or FEATURE_BIN_LABELS.get(feature_col)
        else:
            edges, auto_labels = _default_feature_bin_edges(valid[feature_col])
            labels = labels or auto_labels

    conf = ci / 100
    baseline = float(valid[target_col].mean())

    binned = valid.assign(
        _bin=pd.cut(valid[feature_col], bins=edges, include_lowest=True, labels=labels),
    )
    rows = []
    for interval, grp in binned.groupby("_bin", observed=True):
        n = len(grp)
        if n < min_n:
            continue
        positives = int(grp[target_col].sum())
        rate = positives / n
        lo, hi = wilson_ci(positives, n, conf=conf)
        if isinstance(interval, str) or labels is not None:
            bin_label = str(interval)
            bin_lo = bin_hi = float("nan")
        else:
            bin_label = str(interval)
            bin_lo, bin_hi = float(interval.left), float(interval.right)
        rows.append({
            "feature": feature_col,
            "target_col": target_col,
            "bin_label": bin_label,
            "bin_lo": bin_lo,
            "bin_hi": bin_hi,
            "n": n,
            "n_positive": positives,
            "rate": rate,
            "ci_lo": lo,
            "ci_hi": hi,
            "baseline_rate": baseline,
        })

    return pd.DataFrame(rows)


def summarize_binned_win_rates(
    df: pd.DataFrame,
    feature_col: str,
    **kwargs,
) -> pd.DataFrame:
    """Alias for ``summarize_binned_outcome_rates`` with ``target_col='won_season'``."""
    out = summarize_binned_outcome_rates(
        df, feature_col, target_col="won_season", **kwargs,
    )
    if out.empty:
        return out
    return out.assign(
        n_winners=out["n_positive"],
        win_rate=out["rate"],
        baseline_win_rate=out["baseline_rate"],
    )
