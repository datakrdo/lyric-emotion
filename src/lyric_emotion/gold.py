"""Light gold set (Phase 4): sampling, the pre-registered validity rule and
the metrics that compare model scores with one human reader's ratings.

Thresholds mirror `docs/validation_protocol.md`, written before annotating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from lyric_emotion.data import _slugify

GOLD_COLUMNS = [
    "song_id",
    "artist",
    "title",
    "year",
    "tercile",
    "valence",
    "arousal",
    "families",
    "low_confidence",
    "notes",
    "annotation_date",
    "is_reannotation",
]
# NOTE: thresholds fixed in docs/validation_protocol.md before annotating; edit there first
VALIDATED_LOWER = 0.30


def sample_light_gold_set(window: pd.DataFrame, n_total: int, seed: int) -> pd.DataFrame:
    """Songs for annotation: allocated to artists in proportion to their size,
    spread evenly over terciles of the model's valence within each artist (so
    the sample isn't skewed to what the model finds easy). The model's score
    itself is dropped: the annotator must not see it."""
    window = window.copy()
    window["tercile"] = window.groupby("artist")["valence"].transform(
        lambda v: pd.qcut(v.rank(method="first"), 3, labels=False)
    )
    shares = window["artist"].value_counts(normalize=True) * n_total
    quota = shares.round().astype(int)
    quota.iloc[0] += n_total - quota.sum()
    parts = []
    for artist, group in window.groupby("artist"):
        per_tercile = [quota[artist] // 3 + (i < quota[artist] % 3) for i in range(3)]
        for t, k in enumerate(per_tercile):
            pool = group[group["tercile"] == t]
            parts.append(pool.sample(n=min(k, len(pool)), random_state=seed))
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    out["song_id"] = out["artist"].map(_slugify) + "__" + out["title"].map(_slugify)
    out = out[["song_id", "artist", "title", "year", "tercile"]]
    for col in GOLD_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out["is_reannotation"] = 0
    return out[GOLD_COLUMNS]


def spearman_ci(
    x: np.ndarray, y: np.ndarray, iterations: int, seed: int, level: float = 0.95
) -> tuple[float, float, float]:
    """Spearman rho with a percentile bootstrap CI over songs."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(iterations):
        idx = rng.integers(0, len(x), len(x))
        boots.append(stats.spearmanr(x[idx], y[idx])[0])
    alpha = 1 - level
    low, high = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stats.spearmanr(x, y)[0]), float(low), float(high)


def valence_verdict(rho: float, lower: float) -> str:
    """validated: CI lower bound >= 0.30; partial: point estimate >= 0.30 only."""
    if lower >= VALIDATED_LOWER:
        return "validated"
    return "partial" if rho >= VALIDATED_LOWER else "not_validated"


def family_auc_table(
    model_scores: pd.DataFrame, human_tags: pd.DataFrame, iterations: int, seed: int
) -> pd.DataFrame:
    """Per family: AUC of the model's family score against the human tag
    (0/1), with a bootstrap CI. Families the human never (or always) tagged
    are reported with NaN."""
    rng = np.random.default_rng(seed)
    rows = []
    for family in model_scores.columns:
        y, s = human_tags[family].to_numpy(), model_scores[family].to_numpy()
        row = {"family": family, "n_tagged": int(y.sum()), "auc": np.nan}
        if 0 < y.sum() < len(y):
            boots = []
            for _ in range(iterations):
                idx = rng.integers(0, len(y), len(y))
                if 0 < y[idx].sum() < len(y):
                    boots.append(roc_auc_score(y[idx], s[idx]))
            row |= {
                "auc": float(roc_auc_score(y, s)),
                "auc_low": float(np.percentile(boots, 2.5)),
                "auc_high": float(np.percentile(boots, 97.5)),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def calibrate_valence_cv(model_valence: np.ndarray, human_valence: np.ndarray, seed: int) -> dict:
    """Isotonic map model valence -> human 1-5 scale, checked out of fold
    (a map scored on its own fit would flatter itself). Returns the fitted
    map's breakpoints for the full sample and the out-of-fold MAE."""
    x, y = np.asarray(model_valence, float), np.asarray(human_valence, float)
    oof = np.zeros(len(x))
    for train, test in KFold(5, shuffle=True, random_state=seed).split(x):
        oof[test] = (
            IsotonicRegression(out_of_bounds="clip").fit(x[train], y[train]).predict(x[test])
        )
    final = IsotonicRegression(out_of_bounds="clip").fit(x, y)
    return {
        "oof_mae": float(np.mean(np.abs(oof - y))),
        "baseline_mae": float(np.mean(np.abs(y - np.median(y)))),
        "x": final.X_thresholds_.tolist(),
        "y": final.y_thresholds_.tolist(),
    }


def reannotation_agreement(gold: pd.DataFrame) -> dict:
    """Intra-annotator consistency on the re-annotated songs: ICC(2,1) for
    valence/arousal and exact agreement of the family tags."""
    from lyric_emotion.evaluate import _icc_2_1

    first = gold[gold["is_reannotation"] == 0].drop_duplicates("song_id").set_index("song_id")
    again = gold[gold["is_reannotation"] == 1].set_index("song_id")
    ids = first.index.intersection(again.index)
    if len(ids) < 3:
        return {"n": int(len(ids))}
    a, b = first.loc[ids], again.loc[ids]
    return {
        "n": int(len(ids)),
        "icc_valence": _icc_2_1(a["valence"].to_numpy(float), b["valence"].to_numpy(float)),
        "icc_arousal": _icc_2_1(a["arousal"].to_numpy(float), b["arousal"].to_numpy(float)),
        "family_exact_agreement": float(
            np.mean(
                [
                    set(str(x).split(";")) == set(str(y).split(";"))
                    for x, y in zip(a["families"], b["families"], strict=True)
                ]
            )
        ),
    }


def human_family_tags(gold: pd.DataFrame, families: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f: gold["families"].fillna("").map(lambda v, f=f: int(f in str(v).split(";")))
            for f in families
        }
    )


def evaluate_gold_set(gold: pd.DataFrame, scores: pd.DataFrame, families: dict, seed: int) -> dict:
    """Compare model scores (rows of `scores`, keyed by song_id) with the
    first-pass human ratings in `gold`. Returns a JSON-ready dict."""
    from lyric_emotion.analysis import family_scores

    done = gold[(gold["is_reannotation"] == 0)].dropna(subset=["valence"])
    merged = done.merge(scores, on="song_id", how="inner", suffixes=("", "_model"))
    if len(merged) < 20:
        raise RuntimeError(f"Only {len(merged)} annotated songs matched the scores; need >= 20.")
    rho, low, high = spearman_ci(merged["valence_model"], merged["valence"], 2000, seed)
    arousal_rho = float(stats.spearmanr(merged["arousal_model"], merged["arousal"])[0])
    fam_model = family_scores(
        merged[[c for c in merged.columns if c.startswith("prob_")]], families
    )
    fam_table = family_auc_table(fam_model, human_family_tags(merged, list(families)), 2000, seed)
    return {
        "n": int(len(merged)),
        "valence": {
            "rho": rho,
            "ci_low": low,
            "ci_high": high,
            "verdict": valence_verdict(rho, low),
        },
        "arousal_rho": arousal_rho,
        "families": fam_table.to_dict("records"),
        "calibration": calibrate_valence_cv(merged["valence_model"], merged["valence"], seed),
        "reliability": reannotation_agreement(gold),
    }


def validation_summary(result: dict) -> str:
    v = result["valence"]
    word = {
        "validated": "agrees with",
        "partial": "partly agrees with",
        "not_validated": "does not clearly agree with",
    }[v["verdict"]]
    return (
        f"On {result['n']} songs rated by a human reader, the brightness score {word} the "
        f"reader (rank correlation {v['rho']:.2f}, 95% range {v['ci_low']:.2f}–{v['ci_high']:.2f})."
    )
