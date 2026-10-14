"""Gold-set logic: sampling, the pre-registered validity rule, and metrics —
all on synthetic data."""

import numpy as np
import pandas as pd

from lyric_emotion.gold import (
    family_auc_table,
    sample_light_gold_set,
    spearman_ci,
    valence_verdict,
)


def _window(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "artist": ["The Cure"] * (n * 6 // 10) + ["The Smiths"] * (n - n * 6 // 10),
            "title": [f"t{i}" for i in range(n)],
            "year": 1985,
            "valence": rng.normal(size=n),
            "arousal": rng.normal(size=n),
            "prob_joy": rng.uniform(size=n),
        }
    )


def test_sample_light_gold_set_is_stratified_exact_size_and_reproducible():
    window = _window()
    a = sample_light_gold_set(window, n_total=30, seed=1)
    b = sample_light_gold_set(window, n_total=30, seed=1)
    assert len(a) == 30 and a["song_id"].is_unique
    pd.testing.assert_frame_equal(a, b)
    assert a["artist"].value_counts().to_dict() == {"The Cure": 18, "The Smiths": 12}
    # every valence tercile is represented for each artist
    assert (a.groupby("artist")["tercile"].nunique() == 3).all()
    # the human columns start empty: no model score may leak in
    assert a[["valence", "arousal"]].isna().all().all()
    assert not any(c.startswith("prob_") for c in a.columns)


def test_spearman_ci_brackets_a_known_correlation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=80)
    y = x + rng.normal(scale=1.0, size=80)
    rho, low, high = spearman_ci(x, y, iterations=500, seed=1)
    assert low < rho < high and 0.3 < rho < 0.9


def test_valence_verdict_follows_the_preregistered_thresholds():
    assert valence_verdict(0.6, 0.45) == "validated"
    assert valence_verdict(0.5, 0.2) == "partial"
    assert valence_verdict(0.2, 0.0) == "not_validated"


def test_family_auc_table_scores_model_against_human_tags():
    rng = np.random.default_rng(0)
    tag = rng.integers(0, 2, 80)
    model = pd.DataFrame(
        {"Joy & fun": tag + rng.normal(0, 0.5, 80), "Questioning": rng.normal(size=80)}
    )
    human = pd.DataFrame({"Joy & fun": tag, "Questioning": rng.integers(0, 2, 80)})
    out = family_auc_table(model, human, iterations=300, seed=1).set_index("family")
    assert out.loc["Joy & fun", "auc"] > 0.85 and out.loc["Joy & fun", "auc_low"] > 0.5
    assert out.loc["Questioning", "auc_low"] < 0.5 < out.loc["Questioning", "auc_high"]
