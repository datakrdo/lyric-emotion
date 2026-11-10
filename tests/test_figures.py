"""Each gallery figure writes a PNG in both themes (synthetic data, no models)."""

import numpy as np
import pandas as pd
import pytest

from lyric_emotion.config import Visualization
from lyric_emotion.data import GOEMOTIONS_LABELS
from lyric_emotion.figures import (
    emotion_distribution_figure,
    emotion_heatmap_figure,
    valence_timeseries_figure,
)

VIZ = Visualization(dpi=50, compare_periods=[])
ARTISTS = ("The Cure", "The Smiths")


def _songs() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = [
        {"artist": a, **{f"prob_{k}": rng.uniform(0, 0.5) for k in GOEMOTIONS_LABELS}}
        for a in ARTISTS
        for _ in range(5)
    ]
    return pd.DataFrame(rows)


def _by_album() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = [
        {
            "artist": a,
            "album": f"Album {i}",
            **{f"prob_{k}_mean": rng.uniform(0, 0.5) for k in GOEMOTIONS_LABELS},
        }
        for a in ARTISTS
        for i in range(3)
    ]
    return pd.DataFrame(rows)


def _by_year() -> pd.DataFrame:
    rows = [
        {
            "artist": a,
            "year": 1983 + i,
            "n": 5 + i,
            "valence_mean": 0.1 * i,
            "valence_ci_low": 0.1 * i - 0.1,
            "valence_ci_high": 0.1 * i + 0.1,
        }
        for a in ARTISTS
        for i in range(4)
    ]
    return pd.DataFrame(rows)


CASES = [
    (emotion_distribution_figure, _songs),
    (emotion_heatmap_figure, _by_album),
    (valence_timeseries_figure, _by_year),
]


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("fn,data", CASES, ids=lambda x: getattr(x, "__name__", ""))
def test_figure_writes_png(fn, data, theme, tmp_path):
    out = fn(data(), VIZ, tmp_path / f"{fn.__name__}_{theme}.png", theme=theme)
    assert out.read_bytes().startswith(b"\x89PNG")
