"""The dashboard bundle: pure transformation of scored songs into the three
files the app reads. Synthetic data, no models."""

import json

import numpy as np
import pandas as pd

from lyric_emotion.config import load_config
from lyric_emotion.data import GOEMOTIONS_LABELS
from lyric_emotion.export import build_bundle, verdict_text


def _scores(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for artist in ("The Cure", "The Smiths"):
        for a in range(6):
            offset = rng.normal(0, 0.05)
            for s in range(5):
                probs = {
                    f"prob_{k}": float(np.clip(rng.uniform(0, 0.4) + offset, 0, 1))
                    for k in GOEMOTIONS_LABELS
                }
                if artist == "The Cure":
                    probs["prob_joy"] += 0.5
                rows.append(
                    {
                        "artist": artist,
                        "title": f"{artist}-{a}-{s}",
                        "album": f"{artist}-album-{a}",
                        "album_group": f"{artist}|{artist}-album-{a}",
                        "year": 1980 + a,
                        "valence": rng.normal(0, 0.2),
                        "arousal": rng.normal(0, 0.1),
                        "dominance": rng.normal(0, 0.05),
                        **probs,
                    }
                )
    return pd.DataFrame(rows)


def test_build_bundle_reports_equal_darkness_but_clear_style_and_leaks_no_lyrics():
    config = load_config()
    config.statistics.permutations = 1500  # 28 corrected tests need a fine p-value grid
    config.statistics.cv_repeats = 1
    config.statistics.cv_permutations = 40
    scores = _scores()
    keys = set(zip(scores["artist"], scores["title"], strict=True))
    by_album = pd.DataFrame(
        {
            "artist": ["The Cure"],
            "album": ["x"],
            "year": [1980],
            "n": [5],
            "kind": ["studio"],
            "valence_mean": [0.0],
            "valence_ci_low": [-0.1],
            "valence_ci_high": [0.1],
        }
    )
    songs, albums, findings = build_bundle(
        config,
        scores,
        keys,
        by_album,
        {"The Cure": 0.016, "The Smiths": 0.045},
        set(),
        ("The Cure", "The Smiths"),
    )
    compared = findings["emotions_compared"]
    assert compared[0]["emotion"] == "sadness" and not compared[0]["significant"]
    assert any(r["emotion"] == "joy" and r["higher"] == "The Cure" for r in compared)
    cure = findings["profile"]["The Cure"]
    assert "neutral" not in {r["emotion"] for r in cure["top"]} and len(cure["top"]) == 7
    assert any(r["emotion"] == "joy" and r["distinct"] for r in cure["top"])
    assert "{" not in findings["verdict"]
    assert findings["valence"]["status"] == "none_detected"
    assert findings["style"]["auc"] > 0.7
    assert findings["families"][0]["family"] == "Joy & fun"
    assert findings["families"][0]["p_cure_higher"] > 0.8
    assert songs["stands_out"].str.count(",").eq(2).all()
    assert "joy" in songs.columns  # per-song score of the fingerprint emotions
    assert "lyrics" not in songs.columns and "text" not in songs.columns
    assert songs["style"].notna().all() and len(songs) == len(scores)
    assert {"valence_z_mean", "valence_z_ci_low"} <= set(albums.columns)
    assert {"stands_out", "valence_pct"} <= set(albums.columns)
    json.dumps(findings)  # must be serializable as is


def test_verdict_text_names_each_bands_emotions_and_never_invents_a_sadder_band():
    def row(emotion, higher, significant):
        return {"emotion": emotion, "higher": higher, "significant": significant}

    tie = [row("sadness", "B", False), row("joy", "A", True), row("confusion", "B", True)]
    text = verdict_text(tie, "A", "B")
    assert text.startswith("Equally sad") and "A show more joy" in text and "B show more" in text
    assert "sadder" in verdict_text([row("sadness", "B", True)], "A", "B")
    assert "no other single emotion" in verdict_text([row("sadness", "B", False)], "A", "B")


def test_family_status_is_clear_only_when_the_interval_clears_the_smallest_effect():
    from lyric_emotion.export import family_status

    assert family_status(0.01, 0.25, 0.67, 0.05, 0.147) == "clear"
    assert family_status(0.01, -0.57, -0.14, 0.05, 0.147) == "suggestive"  # excludes 0 only
    assert family_status(0.04, -0.2, 0.1, 0.05, 0.147) == "suggestive"  # interval spans 0
    assert family_status(0.30, -0.1, 0.5, 0.05, 0.147) == "none"  # not significant
