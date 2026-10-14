"""Unit tests for the pure logic in evaluate.py — no network/GPU needed.
The baseline models themselves (cirimus, NRC-VAD download) are exercised
via `lyric-emotion evaluate-baselines`, not here.
"""

import numpy as np
import pandas as pd
import pytest

from lyric_emotion.data import _scale_emobank
from lyric_emotion.evaluate import _icc_2_1, _lexicon_vad_scores


def test_lexicon_vad_scores_averages_known_words():
    lexicon = pd.DataFrame(
        {"valence": [0.5, -0.5], "arousal": [0.2, 0.8], "dominance": [0.1, -0.1]},
        index=["happy", "angry"],
    )
    texts = pd.Series(["I feel happy and angry today", "unknownword"])
    scores = _lexicon_vad_scores(texts, lexicon)
    assert scores[0] == pytest.approx([0.0, 0.5, 0.0])
    assert scores[1] == pytest.approx([0.0, 0.0, 0.0])


def test_scale_emobank_maps_1_to_5_scale_onto_minus_1_to_1():
    scaled = _scale_emobank(pd.Series([1, 3, 5]))
    assert list(scaled) == pytest.approx([-1.0, 0.0, 1.0])


def test_icc_2_1_is_1_for_identical_raters():
    ratings = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _icc_2_1(ratings, ratings) == pytest.approx(1.0)


def test_icc_2_1_is_low_for_unrelated_raters():
    rater_a = np.array([1.0, 1.0, 1.0, 5.0, 5.0])
    rater_b = np.array([5.0, 1.0, 3.0, 1.0, 5.0])
    assert _icc_2_1(rater_a, rater_b) < 0.5
