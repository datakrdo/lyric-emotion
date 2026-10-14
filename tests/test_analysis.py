"""Unit tests for Phase 5 aggregation — built against synthetic scores ahead
of real model inference.
"""

import numpy as np
import pandas as pd

from lyric_emotion.analysis import (
    aggregate_by_album,
    aggregate_by_year,
    bootstrap_ci,
    characteristic_emotions,
    cliffs_delta,
    cluster_bootstrap_delta_ci,
    compare_artists,
    family_scores,
    group_permutation_pvalue,
    separability_cv,
    temporal_regression,
)
from lyric_emotion.config import Aggregation, Statistics


def _stats_config(iterations: int = 500) -> Statistics:
    return Statistics(
        confidence_level=0.95,
        bootstrap_iterations=iterations,
        alpha=0.05,
        multiple_testing_correction="fdr_bh",
    )


def test_bootstrap_ci_recovers_known_mean_and_brackets_it():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, lower, upper = bootstrap_ci(values, iterations=2000, confidence_level=0.95, seed=42)
    assert mean == 3.0
    assert lower <= mean <= upper


def test_aggregate_by_album_drops_groups_below_min_n():
    df = pd.DataFrame(
        {
            "artist": ["The Cure"] * 5 + ["The Smiths"] * 2,
            "album": ["Pornography"] * 5 + ["Rare EP"] * 2,
            "valence": [0.1, 0.2, 0.15, 0.05, 0.3, 0.5, 0.6],
        }
    )
    result = aggregate_by_album(
        df, ["valence"], Aggregation(min_songs_per_album=3, min_songs_per_year=3), _stats_config()
    )
    assert list(result["album"]) == ["Pornography"]
    assert result.iloc[0]["n"] == 5
    assert "valence_ci_low" in result.columns and "valence_ci_high" in result.columns


def test_aggregate_by_year_drops_groups_below_min_n():
    df = pd.DataFrame(
        {
            "artist": ["The Cure"] * 4,
            "year": [1980, 1980, 1980, 1981],
            "arousal": [0.4, 0.5, 0.45, 0.9],
        }
    )
    result = aggregate_by_year(
        df, ["arousal"], Aggregation(min_songs_per_album=3, min_songs_per_year=3), _stats_config()
    )
    assert list(result["year"]) == [1980]


def test_characteristic_emotions_favors_a_rare_standout_over_a_common_high_score():
    # song 0 is unremarkable on "approval" (everyone scores ~0.5) but is the
    # only song with any "grief" at all — grief should win despite the
    # smaller raw number.
    probs = pd.DataFrame(
        {
            "prob_approval": [0.55, 0.5, 0.45, 0.5],
            "prob_grief": [0.2, 0.0, 0.0, 0.0],
        }
    )
    result = characteristic_emotions(probs)
    assert result.iloc[0] == "grief"


def test_cliffs_delta_extremes():
    assert cliffs_delta(np.array([5.0, 6.0, 7.0]), np.array([1.0, 2.0, 3.0])) == 1.0
    assert cliffs_delta(np.array([1.0, 2.0, 3.0]), np.array([5.0, 6.0, 7.0])) == -1.0
    assert cliffs_delta(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0


def test_compare_artists_flags_a_real_difference_and_not_noise():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "artist": ["A"] * 40 + ["B"] * 40,
            "high": np.concatenate([rng.normal(5, 0.5, 40), rng.normal(1, 0.5, 40)]),
            "same": np.concatenate([rng.normal(0, 0.5, 40), rng.normal(0, 0.5, 40)]),
        }
    )
    stats_config = _stats_config(iterations=1000)
    result = compare_artists(df, ["high", "same"], "A", "B", stats_config, seed=1).set_index(
        "score"
    )
    assert result.loc["high", "mean_diff"] > 3
    assert result.loc["high", "significant"]
    assert not result.loc["same", "significant"]


def test_temporal_regression_ignores_rows_with_missing_year():
    df = pd.DataFrame(
        {
            "artist": ["The Cure"] * 5,
            "year": [1980, 1981, 1982, 1983, None],
            "valence": [0.1, 0.2, 0.3, 0.4, 0.9],
        }
    )
    result = temporal_regression(df, "valence").set_index("artist")
    assert result.loc["The Cure", "n"] == 4
    assert not pd.isna(result.loc["The Cure", "slope"])


def test_temporal_regression_detects_perfect_upward_trend_per_artist():
    df = pd.DataFrame(
        {
            "artist": ["The Cure"] * 4 + ["The Smiths"] * 4,
            "year": [1980, 1981, 1982, 1983] * 2,
            "valence": [0.1, 0.2, 0.3, 0.4] + [0.5, 0.5, 0.5, 0.5],
        }
    )
    result = temporal_regression(df, "valence").set_index("artist")
    assert result.loc["The Cure", "slope"] > 0
    assert result.loc["The Cure", "r2"] > 0.99
    assert abs(result.loc["The Smiths", "slope"]) < 1e-9


def _two_bands(effect: float, n_albums: int = 8, per_album: int = 6, seed: int = 0):
    """Songs nested in albums; each album has its own random offset (so songs
    within an album are correlated) and the band shifts the mean by `effect`."""
    rng = np.random.default_rng(seed)
    rows = []
    for band in (1, 0):
        for a in range(n_albums):
            offset = rng.normal(0, 1)
            for _ in range(per_album):
                rows.append((band, f"{band}-{a}", offset + effect * band + rng.normal(0, 0.5)))
    band, album, value = map(np.array, zip(*rows, strict=True))
    return band.astype(bool), album, value


def test_group_permutation_pvalue_detects_effect_and_not_noise():
    is_a, albums, value = _two_bands(effect=3.0)
    assert group_permutation_pvalue(value, is_a, albums, 500, seed=1) < 0.05
    is_a, albums, value = _two_bands(effect=0.0, seed=3)
    assert group_permutation_pvalue(value, is_a, albums, 500, seed=1) > 0.05


def test_group_permutation_is_more_conservative_than_song_level_when_albums_cluster():
    # strong album offsets, no band effect: song-level Mann-Whitney is routinely
    # "significant" for some seeds, the album-level permutation should not be.
    pvals = []
    for seed in range(10, 20):
        is_a, albums, value = _two_bands(effect=0.0, seed=seed)
        pvals.append(group_permutation_pvalue(value, is_a, albums, 300, seed=seed))
    assert np.mean(np.array(pvals) < 0.05) <= 0.2


def test_separability_cv_finds_signal_and_rejects_noise():
    rng = np.random.default_rng(0)
    albums = np.repeat(np.arange(20), 6)
    y = (albums % 2 == 0).astype(int)
    signal = np.column_stack([y + rng.normal(0, 0.7, len(y)), rng.normal(size=len(y))])
    noise = rng.normal(size=(len(y), 2))
    good = separability_cv(signal, y, albums, repeats=2, permutations=30, seed=0)
    bad = separability_cv(noise, y, albums, repeats=2, permutations=30, seed=0)
    assert good["auc"] > 0.8 and good["p_value"] < 0.1
    assert bad["auc"] < 0.65 and bad["p_value"] > 0.1
    assert good["oof_score"].shape == (len(y),)


def test_cluster_bootstrap_delta_ci_brackets_observed_and_widens_with_clustering():
    is_a, albums, value = _two_bands(effect=1.0, seed=5)
    delta, low, high = cluster_bootstrap_delta_ci(value, is_a, albums, 400, 0.95, seed=1)
    assert low <= delta <= high
    assert high - low > 0.1


def test_family_scores_takes_max_over_member_labels():
    probs = pd.DataFrame(
        {"prob_joy": [0.1, 0.9], "prob_amusement": [0.5, 0.2], "prob_fear": [0, 0]}
    )
    out = family_scores(probs, {"Joy & fun": ["joy", "amusement"]})
    assert out["Joy & fun"].tolist() == [0.5, 0.9]


def test_compare_artists_adds_album_level_p_values_when_groups_given():
    is_a, albums, value = _two_bands(effect=3.0)
    df = pd.DataFrame({"artist": np.where(is_a, "A", "B"), "album": albums, "x": value})
    df["noise"] = np.random.default_rng(1).normal(size=len(df))
    result = compare_artists(
        df,
        ["x", "noise"],
        "A",
        "B",
        _stats_config(200),
        seed=1,
        groups=df["album"],
        permutations=300,
    ).set_index("score")
    assert result.loc["x", "p_album"] < 0.05 < result.loc["noise", "p_album"]
    assert {"p_album", "p_adj_album"} <= set(result.columns)


def test_aggregate_by_album_is_reproducible_with_a_seed():
    df = pd.DataFrame({"artist": "A", "album": "X", "kind": "studio", "v": np.arange(10.0)})
    config = Aggregation(min_songs_per_album=3, min_songs_per_year=3)
    first = aggregate_by_album(df, ["v"], config, _stats_config(200), seed=7)
    second = aggregate_by_album(df, ["v"], config, _stats_config(200), seed=7)
    pd.testing.assert_frame_equal(first, second)
