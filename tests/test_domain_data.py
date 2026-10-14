"""Phase 4 domain-adaptation dataset assembly — pure logic, no network."""

import pandas as pd

from lyric_emotion.data import _MOOD_TO_VA, fetch_domain_lyrics


def test_mood_quadrants_cover_all_four_va_signs():
    assert set(_MOOD_TO_VA) == {"happy", "angry", "sad", "relaxed"}
    signs = {mood: (v > 0, a > 0) for mood, (v, a) in _MOOD_TO_VA.items()}
    assert len(set(signs.values())) == 4  # each mood lands in a distinct quadrant


def test_fetch_domain_lyrics_caches_and_reuses(tmp_path, monkeypatch):
    monkeypatch.setenv("GENIUS_ACCESS_TOKEN", "fake")

    calls = []

    class FakeSong:
        lyrics = "la la la"

    class FakeGenius:
        def __init__(self, *a, **k):
            pass

        def search_song(self, title, artist):
            calls.append((artist, title))
            return FakeSong()

    import lyricsgenius

    monkeypatch.setattr(lyricsgenius, "Genius", FakeGenius)

    pairs = pd.DataFrame({"artist": ["A", "A"], "title": ["Song", "Song"]})
    first = fetch_domain_lyrics(pairs, tmp_path)
    second = fetch_domain_lyrics(pairs, tmp_path)

    assert first[("A", "Song")] == "la la la"
    assert second[("A", "Song")] == "la la la"
    assert len(calls) == 1  # second call served from cache, not re-fetched


def test_drop_split_leaks_removes_test_and_validation_texts_seen_in_train():
    import pandas as pd

    from lyric_emotion.data import drop_split_leaks

    df = pd.DataFrame(
        {
            "text": ["a", "a", "b", "b", "c", "d"],
            "split": ["train", "test", "train", "validation", "test", "validation"],
        }
    )
    out = drop_split_leaks(df)
    assert sorted(zip(out["text"], out["split"], strict=True)) == [
        ("a", "train"),
        ("b", "train"),
        ("c", "test"),
        ("d", "validation"),
    ]


def test_clean_domain_set_drops_case_study_artists_leaks_and_train_repeats():
    from lyric_emotion.data import clean_domain_set

    df = pd.DataFrame(
        {
            "artist": ["The Cure", "Morrissey", "Other", "Other", "Other", "Other"],
            "text": ["x", "y", "a", "a", "b", "a"],
            "split": ["train", "train", "train", "train", "train", "test"],
        }
    )
    out = clean_domain_set(df, ["The Cure", "The Smiths", "Morrissey"])
    assert sorted(zip(out["text"], out["split"], strict=True)) == [("a", "train"), ("b", "train")]
