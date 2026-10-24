"""Turn scored songs into the dashboard's data bundle (Phase 3).

The app never computes statistics: it renders `songs.parquet`, `albums.csv`
and `findings.json` written here, so every number and sentence it shows is
produced (and tested) in the pipeline. No lyrics ever enter the bundle.
"""

from __future__ import annotations

import hashlib
from importlib import metadata

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from lyric_emotion.analysis import (
    cliffs_delta,
    cluster_bootstrap_delta_ci,
    family_scores,
    group_permutation_pvalue,
    separability_cv,
)
from lyric_emotion.config import Config
from lyric_emotion.data import GOEMOTIONS_LABELS

# Labels that track literal questions in lyrics; dropped in a robustness check
# so the style finding can't be one emotion's artifact.
_QUESTION_LABELS = ("curiosity", "confusion", "disapproval")


# Smallest difference (Cliff's delta) worth reporting; a finding is "clear"
# only when its interval excludes this, not just zero.
SMALLEST_EFFECT = 0.147


def family_status(
    p_adj: float, ci_low: float, ci_high: float, alpha: float, smallest: float
) -> str:
    """ "clear": significant after correction and the interval stays beyond the
    smallest effect of interest; "suggestive": significant, or the interval
    excludes zero, but not both strongly enough; otherwise "none"."""
    nearest = min(abs(ci_low), abs(ci_high)) if ci_low * ci_high > 0 else 0.0
    if p_adj < alpha and nearest >= smallest:
        return "clear"
    return "suggestive" if p_adj < alpha or nearest > 0 else "none"


def verdict_text(compared: list[dict], band_a: str, band_b: str) -> str:
    """One-sentence answer for the landing page, built from the per-emotion
    comparison (`compared[0]` is always sadness, the question fans ask first)."""
    sad, rest = compared[0], [r for r in compared[1:] if r["significant"]]
    head = (
        "Equally sad, in different ways" if not sad["significant"] else f"{sad['higher']} is sadder"
    )
    parts = []
    for band in (band_a, band_b):
        names = [r["emotion"] for r in rest if r["higher"] == band]
        if names:
            joined = ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1]
            parts.append(f"{band} show more {joined}")
    return f"{head}: {'; '.join(parts)}." if parts else f"{head}: no other single emotion differs."


def _top_k_labels(probs: pd.DataFrame, k: int = 3) -> list[str]:
    """Per row, the k emotions that stand out most *relative to the corpus*
    (z-scored columns) — raw probabilities are dominated by the common labels."""
    z = ((probs - probs.mean()) / probs.std()).to_numpy()
    names = np.array([c.removeprefix("prob_") for c in probs.columns])
    return [", ".join(names[np.argsort(-row)[:k]]) for row in z]


def _fdr(p_values: list[float]) -> list[float]:
    return [float(p) for p in stats.false_discovery_control(np.array(p_values), method="bh")]


def question_line_share(lyrics: pd.Series) -> float:
    """Share of non-tag lyric lines that end with '?' (a verifiable fact the
    dashboard quotes next to the 'questioning' finding)."""
    lines = [
        line.strip()
        for text in lyrics
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("[")
    ]
    return float(np.mean([line.endswith("?") for line in lines])) if lines else float("nan")


def build_bundle(
    config: Config,
    scores: pd.DataFrame,
    window_keys: set[tuple[str, str]],
    by_album: pd.DataFrame,
    question_share: dict[str, float],
    domain_train_keys: set[tuple[str, str]],
    bands: tuple[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """`scores`: one row per studio song (prob_* columns, valence, ...).
    `bands`: (a, b); positive deltas mean a > b. Returns (songs, albums, findings)."""
    band_a, band_b = bands
    seed = config.project.random_seed
    prob_cols = [f"prob_{label}" for label in GOEMOTIONS_LABELS]
    scores = scores.reset_index(drop=True)

    in_window = np.array(
        [(a, t) in window_keys for a, t in zip(scores["artist"], scores["title"], strict=True)]
    )
    win = scores[in_window].reset_index(drop=True)
    is_a = (win["artist"] == band_a).to_numpy()
    albums = win["album_group"].to_numpy()
    st = config.statistics

    # --- darkness (valence) ---
    valence = win["valence"].to_numpy()
    delta, low, high = cluster_bootstrap_delta_ci(
        valence, is_a, albums, st.bootstrap_iterations, st.confidence_level, seed
    )
    valence_p = group_permutation_pvalue(valence, is_a, albums, st.permutations, seed)
    darkness = "none_detected" if low <= 0 <= high else "difference"

    # --- style: can the 28-emotion profile tell the bands apart on unseen albums? ---
    X = win[prob_cols].to_numpy()
    sep = separability_cv(X, is_a.astype(int), albums, st.cv_repeats, st.cv_permutations, seed)
    keep = [c for c in prob_cols if c.removeprefix("prob_") not in _QUESTION_LABELS]
    auc_wo_questions = separability_cv(
        win[keep].to_numpy(), is_a.astype(int), albums, st.cv_repeats, 0, seed
    )["auc"]
    style = (
        "clear"
        if sep["p_value"] < 0.05 and sep["auc"] >= 0.65
        else "suggestive"
        if sep["p_value"] < 0.10
        else "none"
    )

    # --- single emotion families, album-level inference ---
    fam = family_scores(win[prob_cols], config.emotion_families)
    fam_rows = []
    for name in fam.columns:
        v = fam[name].to_numpy()
        d = cliffs_delta(v[is_a], v[~is_a])
        fam_rows.append(
            {
                "family": name,
                "delta": d,
                "p_cure_higher": (1 + d) / 2,
                "p_album": group_permutation_pvalue(v, is_a, albums, st.permutations, seed),
            }
        )
    for row, p_adj in zip(fam_rows, _fdr([r["p_album"] for r in fam_rows]), strict=True):
        row["p_adj_album"] = p_adj
        _, row["ci_low"], row["ci_high"] = cluster_bootstrap_delta_ci(
            fam[row["family"]].to_numpy(),
            is_a,
            albums,
            st.bootstrap_iterations,
            st.confidence_level,
            seed,
        )
        row["status"] = family_status(
            p_adj, row["ci_low"], row["ci_high"], st.alpha, SMALLEST_EFFECT
        )
    fam_rows.sort(key=lambda r: r["p_album"])
    statuses = {r["status"] for r in fam_rows}
    emotions = (
        "clear" if "clear" in statuses else "suggestive" if "suggestive" in statuses else "none"
    )

    # --- every single emotion, album-level inference, sadness first ---
    emo_p = [
        group_permutation_pvalue(win[c].to_numpy(), is_a, albums, st.permutations, seed)
        for c in prob_cols
    ]
    compared = []
    for c, p, p_adj in zip(prob_cols, emo_p, _fdr(emo_p), strict=True):
        v = win[c].to_numpy()
        med_a, med_b = float(np.median(v[is_a])), float(np.median(v[~is_a]))
        compared.append(
            {
                "emotion": c.removeprefix("prob_"),
                "higher": band_a if med_a > med_b else band_b,
                "median_a": med_a,
                "median_b": med_b,
                "p_album": p,
                "p_adj": p_adj,
                "significant": p_adj < st.alpha,
                "pair_share_a": (1 + cliffs_delta(v[is_a], v[~is_a])) / 2,
            }
        )
    compared.sort(key=lambda r: (r["emotion"] != "sadness", r["p_album"]))
    compared = [r for i, r in enumerate(compared) if i == 0 or r["significant"]]

    # --- each band's fingerprint: its typical song, "neutral" (the model's catch-all) left out ---
    distinct = {(r["emotion"], r["higher"]) for r in compared if r["significant"]}
    profile: dict[str, dict] = {}
    for band, mask in ((band_a, is_a), (band_b, ~is_a)):
        med = win.loc[mask, prob_cols].median().rename(lambda c: c.removeprefix("prob_"))
        med = med.drop("neutral").sort_values(ascending=False)
        profile[band] = {
            "top": [
                {"emotion": e, "score": float(s), "distinct": (e, band) in distinct}
                for e, s in med.head(7).items()
            ],
            "sadness_rank": list(med.index).index("sadness") + 1,
        }

    # --- which labels sit at each pole of the style axis (rank-based, single label) ---
    label_auc = {c.removeprefix("prob_"): roc_auc_score(is_a, win[c]) for c in prob_cols}
    ranked = sorted(label_auc, key=label_auc.__getitem__)
    poles = {band_a: ranked[::-1][:3], band_b: ranked[:3]}

    # --- sensitivity: drop songs the stage-2 model was trained on ---
    clean = ~np.array(
        [(a, t) in domain_train_keys for a, t in zip(win["artist"], win["title"], strict=True)]
    )
    clean_delta = cliffs_delta(valence[is_a & clean], valence[~is_a & clean])
    clean_auc = separability_cv(
        X[clean], is_a[clean].astype(int), albums[clean], st.cv_repeats, 0, seed
    )["auc"]

    findings = {
        "bands": [band_a, band_b],
        "window": {
            "n_songs": int(len(win)),
            "n_a": int(is_a.sum()),
            "n_b": int((~is_a).sum()),
            "n_albums": int(len(set(albums))),
            "n_studio_albums": int(
                win.loc[
                    ~win["album_group"].str.contains("Singles & rarities"), "album_group"
                ].nunique()
            ),
            "start_year": int(win["year"].min()),
            "end_year": int(win["year"].max()),
        },
        "verdict": verdict_text(compared, band_a, band_b),
        "emotions_compared": compared,
        "profile": profile,
        "valence": {
            "status": darkness,
            "delta": delta,
            "ci_low": low,
            "ci_high": high,
            "p_album": valence_p,
            "smallest_effect_of_interest": SMALLEST_EFFECT,
        },
        "style": {
            "status": style,
            "auc": sep["auc"],
            "p_value": sep["p_value"],
            "null_95": sep["null_95"],
            "n_permutations": sep["n_permutations"],
            "auc_without_question_labels": auc_wo_questions,
        },
        "emotions": {"status": emotions, "families": fam_rows},
        "families": fam_rows,
        "poles": poles,
        "question_line_share": question_share,
        "sensitivity": {
            "excluding_stage2_training_songs": {
                "n": int(clean.sum()),
                "valence_delta": clean_delta,
                "style_auc": clean_auc,
            }
        },
        "meta": {
            "seed": seed,
            "config_sha256": hashlib.sha256(config.model_dump_json().encode()).hexdigest()[:16],
            "versions": {
                pkg: metadata.version(pkg) for pkg in ("numpy", "pandas", "scipy", "scikit-learn")
            },
        },
    }

    # --- per-song table (no lyrics) ---
    mean, std = scores["valence"].mean(), scores["valence"].std()
    songs = pd.DataFrame(
        {
            "artist": scores["artist"],
            "title": scores["title"],
            "album": scores["album"],
            "year": scores["year"],
            "in_window": in_window,
            "valence_z": (scores["valence"] - mean) / std,
            "valence_pct": scores["valence"].rank(pct=True) * 100,
            "style": np.nan,
            "stands_out": _top_k_labels(scores[prob_cols]),
        }
    )
    # per-song score for every emotion that shows up in a band's fingerprint (no lyrics)
    for emotion in sorted({r["emotion"] for p in profile.values() for r in p["top"]}):
        songs[emotion] = scores[f"prob_{emotion}"].round(3)
    oof = sep["oof_score"]
    songs.loc[in_window, "style"] = (oof - oof.mean()) / oof.std()

    # Album-level analogue of the per-song columns: scored against the other albums.
    per_album = scores.groupby(["artist", "album"])
    album_means = per_album[prob_cols].mean()
    album_traits = pd.DataFrame(
        {
            "stands_out": _top_k_labels(album_means),
            "valence_pct": per_album["valence"].mean().rank(pct=True) * 100,
        },
        index=album_means.index,
    ).reset_index()
    albums_out = by_album.merge(album_traits, on=["artist", "album"], how="left")
    for col in ("valence_mean", "valence_ci_low", "valence_ci_high"):
        albums_out[col.replace("valence_", "valence_z_")] = (albums_out[col] - mean) / std
    keep_cols = [
        c
        for c in (
            "artist",
            "album",
            "year",
            "n",
            "kind",
            "valence_z_mean",
            "valence_z_ci_low",
            "valence_z_ci_high",
            "valence_pct",
            "stands_out",
        )
        if c in albums_out.columns
    ]
    return songs, albums_out[keep_cols], findings
