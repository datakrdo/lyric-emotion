"""Phase 1 tests: real network calls to Hugging Face Hub / GitHub.

Marked so they can be skipped offline (`pytest -m "not network"`), but they
run by default since the whole point of Phase 1 is verifying the actual
downloads and splits.
"""

import pytest

from lyric_emotion.data import GOEMOTIONS_LABELS, build_emotions_dataset, build_vad_dataset

pytestmark = pytest.mark.network


def test_emotions_dataset_has_28_binary_label_columns():
    df = build_emotions_dataset()
    assert len(GOEMOTIONS_LABELS) == 28
    for label in GOEMOTIONS_LABELS:
        assert set(df[label].unique()) <= {0, 1}
    assert set(df["split"].unique()) == {"train", "validation", "test"}


def test_emotions_splits_do_not_overlap():
    df = build_emotions_dataset()
    ids_by_split = {s: set(g["id"]) for s, g in df.groupby("split")}
    assert ids_by_split["train"].isdisjoint(ids_by_split["validation"])
    assert ids_by_split["train"].isdisjoint(ids_by_split["test"])
    assert ids_by_split["validation"].isdisjoint(ids_by_split["test"])


def test_vad_scores_are_scaled_to_minus_one_one():
    df = build_vad_dataset()
    for col in ("valence", "arousal", "dominance"):
        assert df[col].min() >= -1.0
        assert df[col].max() <= 1.0


def test_vad_splits_do_not_overlap():
    df = build_vad_dataset()
    ids_by_split = {s: set(g["id"]) for s, g in df.groupby("split")}
    train_ids = ids_by_split.get("train", set())
    for split_name, ids in ids_by_split.items():
        if split_name == "train":
            continue
        assert train_ids.isdisjoint(ids)
