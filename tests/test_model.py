"""Unit tests for the pure logic in model.py — no GPU/network needed.
End-to-end training itself is exercised via `lyric-emotion train --smoke`,
not here (that's a live GPU run, not a unit test).
"""

import numpy as np
import pytest

from lyric_emotion.model import (
    GOEMOTIONS_LABELS,
    _ccc,
    _mean_by_song,
    _tune_thresholds,
    score_multilabel,
)


def test_ccc_is_1_for_identical_arrays():
    x = np.array([0.1, 0.5, -0.3, 0.8])
    assert _ccc(x, x) == pytest.approx(1.0)


def test_ccc_penalizes_constant_offset_unlike_pearson():
    x = np.array([0.1, 0.5, -0.3, 0.8])
    y = x + 1.0  # perfectly correlated but shifted
    assert _ccc(x, y) < 0.5  # Pearson r would be 1.0 here


def test_tune_thresholds_finds_perfect_separation():
    # class 0: perfectly separable at 0.5; class 1: never positive
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.1], [0.2, 0.1]])
    labels = np.array([[1, 0], [1, 0], [0, 0], [0, 0]])
    thresholds, macro_f1, per_class_f1 = _tune_thresholds(probs, labels)
    assert per_class_f1[0] == pytest.approx(1.0)
    assert per_class_f1[1] == 0.0
    assert len(thresholds) == 2


def test_score_multilabel_perfect_predictions_score_1():
    # every class matches its (all-zero-but-two) labels exactly
    n = len(GOEMOTIONS_LABELS)
    labels = np.zeros((4, n), dtype=int)
    labels[0, 0] = 1
    labels[1, 0] = 1
    labels[2, 1] = 1
    probs = labels.astype(float)
    thresholds = [0.5] * n
    result = score_multilabel(probs, labels, thresholds)
    # micro-F1 pools TP/FP/FN across all classes, so it's insensitive to the
    # 26 always-negative classes; macro-F1/mAP average per-class and count
    # those as 0 (no positives to predict), so they stay far below 1.
    assert result["micro_f1"] == pytest.approx(1.0)
    assert result["per_class_f1"][GOEMOTIONS_LABELS[0]] == pytest.approx(1.0)
    assert result["per_class_f1"][GOEMOTIONS_LABELS[1]] == pytest.approx(1.0)
    assert result["per_class_average_precision"][GOEMOTIONS_LABELS[0]] == pytest.approx(1.0)


def test_mean_by_song_averages_sections_back_to_one_row_per_song():
    # song 0 has 2 sections, song 1 has 1 section
    section_scores = np.array([[0.0, 1.0], [1.0, 1.0], [0.4, 0.6]])
    song_index = np.array([0, 0, 1])
    result = _mean_by_song(section_scores, song_index, n_songs=2)
    assert result[0] == pytest.approx([0.5, 1.0])
    assert result[1] == pytest.approx([0.4, 0.6])


def test_score_vad_full_scores_whole_text_and_sections_average_back(monkeypatch):
    import numpy as np

    from lyric_emotion import model

    seen = []

    def fake_score(_m, _t, texts, *_a):
        seen.append(list(texts))
        return np.array([[float(len(t)), 0.0, 0.0] for t in texts], dtype=np.float32)

    monkeypatch.setattr(model, "_score_batches", fake_score)
    lyric = "[Verse 1]\nab\n\n[Chorus]\nabcd"
    full = model.score_vad(None, None, [lyric], "cpu", 1024, 8, "full")
    sections = model.score_vad(None, None, [lyric], "cpu", 1024, 8, "sections")
    assert seen[0] == [lyric] and full[0, 0] == len(lyric)
    assert seen[1] == ["ab", "abcd"] and sections[0, 0] == 3.0
