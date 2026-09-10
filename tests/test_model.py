"""Unit tests for the pure logic in model.py — no GPU/network needed.
End-to-end training itself is exercised via `lyric-emotion train --smoke`,
not here (that's a live GPU run, not a unit test).
"""

import numpy as np
import pytest

from lyric_emotion.model import _ccc, _tune_thresholds


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
