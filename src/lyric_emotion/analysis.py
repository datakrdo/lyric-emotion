"""Aggregate per-song emotion/VAD scores into album- and year-level summaries
with bootstrap uncertainty (Phase 5). Consumes whatever scored DataFrame
`predict` eventually produces — no dependency on the trained model itself,
so this can be built and tested ahead of inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lyric_emotion.config import Aggregation, Statistics


def bootstrap_ci(
    values: np.ndarray, iterations: int, confidence_level: float, seed: int | None = None
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of `values` — one vectorized
    resample instead of a per-iteration Python loop.
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    means = rng.choice(values, size=(iterations, n), replace=True).mean(axis=1)
    alpha = 1 - confidence_level
    lower, upper = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lower), float(upper)


def mode_year(years: pd.Series) -> float:
    """Most common release year in a group. Mode, not mean: songs on one
    album often carry different per-song release dates (a bonus track, an
    earlier single) — averaging produces a fractional "year" that doesn't
    correspond to any real release.
    """
    modes = years.dropna().mode()
    return modes.iloc[0] if len(modes) else float("nan")


def aggregate_scores(
    df: pd.DataFrame,
    group_cols: list[str],
    score_cols: list[str],
    min_n: int,
    stats_config: Statistics,
    seed: int | None = None,
) -> pd.DataFrame:
    """Mean + bootstrap CI per score column, grouped by `group_cols`. Groups
    with fewer than `min_n` songs are dropped — never silently reported with
    an unreliable n.
    """
    rows = []
    for keys, group in df.groupby(group_cols):
        if len(group) < min_n:
            continue
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, key_values, strict=True))
        row["n"] = len(group)
        for col in score_cols:
            mean, lower, upper = bootstrap_ci(
                group[col].to_numpy(),
                stats_config.bootstrap_iterations,
                stats_config.confidence_level,
                seed,
            )
            row[f"{col}_mean"] = mean
            row[f"{col}_ci_low"] = lower
            row[f"{col}_ci_high"] = upper
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_by_album(
    df: pd.DataFrame,
    score_cols: list[str],
    config: Aggregation,
    stats_config: Statistics,
    seed: int | None = None,
) -> pd.DataFrame:
    group_cols = ["artist", "album"] + (["kind"] if "kind" in df.columns else [])
    return aggregate_scores(
        df, group_cols, score_cols, config.min_songs_per_album, stats_config, seed
    )


def aggregate_by_year(
    df: pd.DataFrame,
    score_cols: list[str],
    config: Aggregation,
    stats_config: Statistics,
    seed: int | None = None,
) -> pd.DataFrame:
    return aggregate_scores(
        df, ["artist", "year"], score_cols, config.min_songs_per_year, stats_config, seed
    )


def characteristic_emotions(probs: pd.DataFrame) -> pd.Series:
    """Which emotion stands out for each row, relative to the rest of the
    corpus — not which has the highest raw probability. GoEmotions labels
    like `neutral`/`approval`/`annoyance` dominate every raw-probability
    ranking (frequency bias in the training data), which flattens every
    song to the same 2-3 labels. Z-scoring each `prob_*` column first (so a
    song that's unusually high on a rare emotion outranks one that's merely
    average-high on a common one) gives a varied, legible palette instead.
    """
    z = (probs - probs.mean()) / probs.std()
    return z.idxmax(axis=1).str.removeprefix("prob_")


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Non-parametric effect size in [-1, 1]: fraction of pairs where x > y
    minus the fraction where x < y. Doesn't assume normality, unlike Cohen's d.
    """
    comparisons = x[:, None] - y[None, :]
    return float((np.sign(comparisons) > 0).mean() - (np.sign(comparisons) < 0).mean())


def compare_artists(
    df: pd.DataFrame,
    score_cols: list[str],
    artist_a: str,
    artist_b: str,
    stats_config: Statistics,
    seed: int | None = None,
    groups: pd.Series | None = None,
    permutations: int = 2000,
) -> pd.DataFrame:
    """Mean difference (a - b) with bootstrap CI, Cliff's delta effect size,
    and a Mann-Whitney p-value per score column, FDR-corrected across all of
    them since this runs many comparisons at once. With `groups` (album per
    row) it adds `p_album`/`p_adj_album`: the same delta tested by permuting
    whole albums, which is the p-value that respects song clustering.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for col in score_cols:
        a = df.loc[df["artist"] == artist_a, col].to_numpy()
        b = df.loc[df["artist"] == artist_b, col].to_numpy()
        boot_a = rng.choice(a, size=(stats_config.bootstrap_iterations, len(a)), replace=True)
        boot_b = rng.choice(b, size=(stats_config.bootstrap_iterations, len(b)), replace=True)
        diffs = boot_a.mean(axis=1) - boot_b.mean(axis=1)
        alpha = 1 - stats_config.confidence_level
        lower, upper = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        _, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append(
            {
                "score": col,
                "mean_diff": float(a.mean() - b.mean()),
                "ci_low": float(lower),
                "ci_high": float(upper),
                "cliffs_delta": cliffs_delta(a, b),
                "p_value": float(p_value),
                "n_a": len(a),
                "n_b": len(b),
            }
        )
    result = pd.DataFrame(rows)
    result["p_adj"], result["significant"] = _fdr_correct(result["p_value"], stats_config)
    if groups is not None:
        in_pair = df["artist"].isin([artist_a, artist_b])
        is_a = (df.loc[in_pair, "artist"] == artist_a).to_numpy()
        album_ids = groups[in_pair].to_numpy()
        result["p_album"] = [
            group_permutation_pvalue(
                df.loc[in_pair, col].to_numpy(), is_a, album_ids, permutations, seed
            )
            for col in score_cols
        ]
        result["p_adj_album"], _ = _fdr_correct(result["p_album"], stats_config)
    return result


def _fdr_correct(p_values: pd.Series, stats_config: Statistics) -> tuple[np.ndarray, np.ndarray]:
    methods = {"fdr_bh": "bh", "fdr_by": "by"}
    p_adj = stats.false_discovery_control(
        p_values.to_numpy(), method=methods[stats_config.multiple_testing_correction]
    )
    return p_adj, p_adj < stats_config.alpha


def temporal_regression(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Per-artist linear trend of `score_col` over `year`. Exploratory only —
    report as "estimated temporal association", never causally.
    """
    df = df.dropna(subset=["year", score_col])
    rows = []
    for artist, group in df.groupby("artist"):
        result = stats.linregress(group["year"], group[score_col])
        rows.append(
            {
                "artist": artist,
                "slope": result.slope,
                "intercept": result.intercept,
                "r2": result.rvalue**2,
                "p_value": result.pvalue,
                "n": len(group),
            }
        )
    return pd.DataFrame(rows)


# --- Cluster-aware inference -------------------------------------------------
# Songs on one album resemble each other, so 154 songs are really ~34 albums.
# Everything below permutes/resamples whole albums instead of songs.


def _group_index(groups: np.ndarray, is_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-song group id, and the band label of each group (groups nest in a band)."""
    _, inverse = np.unique(groups, return_inverse=True)
    first = np.unique(inverse, return_index=True)[1]
    return inverse, is_a[first]


def group_permutation_pvalue(
    values: np.ndarray,
    is_a: np.ndarray,
    groups: np.ndarray,
    iterations: int,
    seed: int | None = None,
) -> float:
    """Two-sided p-value for Cliff's delta between band a and b, permuting the
    band label across whole groups (albums) rather than across songs."""
    values, is_a = np.asarray(values, float), np.asarray(is_a, bool)
    inverse, group_label = _group_index(np.asarray(groups), is_a)
    observed = abs(cliffs_delta(values[is_a], values[~is_a]))
    rng = np.random.default_rng(seed)
    hits = used = 0
    for _ in range(iterations):
        label = rng.permutation(group_label)[inverse]
        if label.all() or not label.any():
            continue
        used += 1
        hits += abs(cliffs_delta(values[label], values[~label])) >= observed - 1e-12
    return (hits + 1) / (used + 1)


def cluster_bootstrap_delta_ci(
    values: np.ndarray,
    is_a: np.ndarray,
    groups: np.ndarray,
    iterations: int,
    confidence_level: float,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Cliff's delta (a vs b) with a percentile CI from resampling whole
    albums within each band."""
    values, is_a, groups = np.asarray(values, float), np.asarray(is_a, bool), np.asarray(groups)
    members = [
        [values[(groups == g) & (is_a == side)] for g in np.unique(groups[is_a == side])]
        for side in (True, False)
    ]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(iterations):
        a, b = (
            np.concatenate([side[i] for i in rng.integers(0, len(side), len(side))])
            for side in members
        )
        deltas.append(cliffs_delta(a, b))
    alpha = 1 - confidence_level
    low, high = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return cliffs_delta(values[is_a], values[~is_a]), float(low), float(high)


def _grouped_cv_auc(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, repeats: int, seed: int
) -> tuple[float, np.ndarray]:
    # NOTE: fixed C=0.5 logistic regression, no tuning (tuning on n~150 would overfit)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    scores = []
    for r in range(repeats):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + r)
        scores.append(
            cross_val_predict(model, X, y, cv=cv, groups=groups, method="decision_function")
        )
    oof = np.mean(scores, axis=0)
    return float(np.mean([roc_auc_score(y, s) for s in scores])), oof


def separability_cv(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, repeats: int, permutations: int, seed: int
) -> dict[str, Any]:
    """How well can the features tell the two bands apart on albums the model
    has not seen? Grouped (by album) cross-validated AUC, per-song
    out-of-fold scores, and a permutation p-value that shuffles the band label
    across albums."""
    X, y, groups = np.asarray(X, float), np.asarray(y, int), np.asarray(groups)
    auc, oof = _grouped_cv_auc(X, y, groups, repeats, seed)
    inverse, group_label = _group_index(groups, y.astype(bool))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(permutations):
        y_perm = rng.permutation(group_label.astype(int))[inverse]
        if y_perm.min() == y_perm.max():
            continue
        null.append(_grouped_cv_auc(X, y_perm, groups, repeats, int(rng.integers(1_000_000)))[0])
    p_value = (np.sum(np.array(null) >= auc) + 1) / (len(null) + 1) if null else float("nan")
    return {
        "auc": auc,
        "oof_score": oof,
        "p_value": float(p_value),
        "null_95": float(np.percentile(null, 95)) if null else float("nan"),
        "n_permutations": len(null),
    }


def family_scores(probs: pd.DataFrame, families: dict[str, list[str]]) -> pd.DataFrame:
    """One score per broad emotion family: the max of its member labels' `prob_*`."""
    return pd.DataFrame(
        {
            name: probs[[f"prob_{m}" for m in members]].max(axis=1)
            for name, members in families.items()
        }
    )
