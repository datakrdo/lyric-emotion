"""Unit tests for the pure logic behind Genius acquisition — no network
needed. The live fetch itself is exercised via `lyric-emotion fetch`.
"""

from pathlib import Path

from lyric_emotion.data import (
    _normalized_title,
    _slugify,
    fetch_artist_songs,
    write_fetch_coverage_report,
)


def test_normalized_title_strips_parenthetical_suffixes():
    assert _normalized_title("Accept Yourself") == "accept yourself"
    assert _normalized_title("Accept Yourself (Live Session)") == "accept yourself"
    assert _normalized_title("Ask [Remastered]") == "ask"


def test_slugify_is_filesystem_safe():
    assert _slugify("The Cure") == "the-cure"
    assert _slugify("The Smiths") == "the-smiths"


def test_fetch_artist_songs_reuses_cache(tmp_path: Path):
    from lyric_emotion.config import Artist

    cache_dir = tmp_path / "genius"
    cache_dir.mkdir()
    (cache_dir / "the-cure.json").write_text(
        '{"artist": "The Cure", "requested": 1, "fetched_at": "x", "songs": []}'
    )

    raw = fetch_artist_songs(None, Artist(name="The Cure", max_songs=1), cache_dir)
    assert raw["songs"] == []


def test_write_fetch_coverage_report_counts_duplicates(tmp_path: Path):
    raw = {
        "artist": "The Cure",
        "requested": 3,
        "songs": [
            {"title": "Song A", "lyrics": "text", "album": {"name": "X"}, "release_date": "1980"},
            {
                "title": "Song A (Live)",
                "lyrics": "text",
                "album": {"name": "X"},
                "release_date": "1980",
            },
            {"title": "Song B", "lyrics": "", "album": None, "release_date": None},
        ],
    }
    report_path = write_fetch_coverage_report([raw], docs_dir=tmp_path)
    content = report_path.read_text()
    assert "| The Cure | 3 | 3 | 2 | 2 | 2 | 1 | 1 |" in content
