"""Unit tests for the pure logic in evaluate.py — no network/GPU needed.
The baseline models themselves (cirimus, NRC-VAD download) are exercised
via `lyric-emotion evaluate-baselines`, not here.
"""

import pandas as pd
import pytest

from lyric_emotion.evaluate import _lexicon_vad_scores


def test_lexicon_vad_scores_averages_known_words():
    lexicon = pd.DataFrame(
        {"valence": [0.5, -0.5], "arousal": [0.2, 0.8], "dominance": [0.1, -0.1]},
        index=["happy", "angry"],
    )
    texts = pd.Series(["I feel happy and angry today", "unknownword"])
    scores = _lexicon_vad_scores(texts, lexicon)
    assert scores[0] == pytest.approx([0.0, 0.5, 0.0])
    assert scores[1] == pytest.approx([0.0, 0.0, 0.0])
