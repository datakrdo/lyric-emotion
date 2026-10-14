"""Unit tests for the pure logic behind Genius acquisition — no network
needed. The live fetch itself is exercised via `lyric-emotion fetch`.
"""

from pathlib import Path

import pandas as pd

from lyric_emotion.data import (
    _normalized_title,
    _slugify,
    build_song_album_map,
    canonical_studio_album,
    dedupe_studio_songs,
    download_dmdd,
    download_moodylyrics,
    fetch_artist_songs,
    filter_by_year_range,
    flag_song_versions,
    is_studio_take,
    label_studio_albums,
    release_date,
    sample_gold_set_candidates,
    save_studio_corpus,
    split_sections,
    write_dedup_report,
    write_fetch_coverage_report,
)


def test_split_sections_strips_tags_and_dedupes_repeated_chorus():
    lyrics = (
        "[Verse 1]\nfirst line\n\n"
        "[Chorus]\nrepeat me\n\n"
        "[Verse 2]\nsecond line\n\n"
        "[Chorus]\nrepeat me"
    )
    sections = split_sections(lyrics)
    assert sections == ["first line", "repeat me", "second line"]


def test_split_sections_falls_back_to_blank_line_paragraphs_without_tags():
    lyrics = "first paragraph\nsame paragraph\n\nsecond paragraph"
    assert split_sections(lyrics) == ["first paragraph\nsame paragraph", "second paragraph"]


def test_split_sections_never_returns_empty_for_nonempty_input():
    assert split_sections("just one line, no tags, no blank lines") == [
        "just one line, no tags, no blank lines"
    ]


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


def test_fetch_artist_songs_only_scrapes_lyrics_for_studio_takes(tmp_path: Path):
    from lyric_emotion.config import Artist

    class FakeGenius:
        excluded_terms = ["tracklist"]
        lyrics_calls: list[str] = []

        def search_artists(self, name):
            return {"sections": [{"type": "artist", "hits": [{"result": {"id": 1, "name": name}}]}]}

        def artist_songs(self, artist_id, per_page, page):
            if page > 1:
                return {"songs": [], "next_page": None}
            return {
                "songs": [
                    {"id": 10, "lyrics_state": "complete", "title": "Song A"},
                    {"id": 11, "lyrics_state": "complete", "title": "Song A (Live)"},
                    {"id": 12, "lyrics_state": "incomplete", "title": "Song B"},
                ],
                "next_page": None,
            }

        def song(self, song_id):
            titles = {10: "Song A", 11: "Song A (Live)"}
            return {"song": {"id": song_id, "title": titles[song_id], "url": f"u{song_id}"}}

        def lyrics(self, song_url):
            self.lyrics_calls.append(song_url)
            return "the lyrics"

    genius = FakeGenius()
    raw = fetch_artist_songs(genius, Artist(name="X", max_songs=10), tmp_path / "genius")

    assert genius.lyrics_calls == ["u10"]  # not the live version
    by_title = {s["title"]: s["lyrics"] for s in raw["songs"]}
    assert by_title["Song A"] == "the lyrics"
    assert by_title["Song A (Live)"] == ""


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


def test_flag_song_versions_detects_each_flag():
    flags = flag_song_versions(
        [
            {"title": "Song A (Live at Royal Festival Hall)"},
            {"title": "Song A (Chestnut Studio Demo)"},
            {"title": "Song A (Radio Edit)"},
            {"title": "Song A [2010 Remaster]"},
            {"title": "Song A"},
        ]
    )
    assert [f["is_live"] for f in flags] == [True, False, False, False, False]
    assert [f["is_demo"] for f in flags] == [False, True, False, False, False]
    assert [f["is_remix"] for f in flags] == [False, False, True, False, False]
    assert [f["is_reissue"] for f in flags] == [False, False, False, True, False]


def test_flag_song_versions_catches_live_album_from_name_and_config_exception():
    songs = [
        {"title": "Some Girls Are Bigger Than Others", "album": {"name": "The Queen Is Dead"}},
        {"title": "Meat Is Murder", "album": {"name": "Live at the BBC"}},
        {"title": "Heaven Knows I'm Miserable Now", "album": {"name": "Rank"}},
    ]
    flags = flag_song_versions(songs, non_studio_albums={"Rank"})
    assert [f["is_live"] for f in flags] == [False, True, True]


def test_dedupe_studio_songs_prefers_studio_take_and_merges_fuzzy_titles():
    songs = [
        {"title": "10:15 Saturday Night", "lyrics": "full studio lyrics here"},
        {"title": "10:15, Saturday Night (John Peel Session)", "lyrics": "short live take"},
        {"title": "10:15 Saturday Night (Chestnut Studio Demo)", "lyrics": "demo take"},
        {"title": "Another Song", "lyrics": "unique"},
    ]
    canonical = dedupe_studio_songs(songs)
    assert len(canonical) == 2
    titles = {s["title"] for s in canonical}
    assert "10:15 Saturday Night" in titles
    assert "Another Song" in titles


def test_dedupe_studio_songs_falls_back_to_longest_when_no_studio_take():
    songs = [
        {"title": "Song A (Live)", "lyrics": "short"},
        {"title": "Song A (Demo)", "lyrics": "a much longer demo lyric body"},
    ]
    canonical = dedupe_studio_songs(songs)
    assert len(canonical) == 1
    assert canonical[0]["lyrics"] == "a much longer demo lyric body"


def test_write_dedup_report_studio_count_excludes_songs_with_no_lyrics(tmp_path: Path):
    raw_artists = [
        {
            "artist": "The Cure",
            "songs": [
                {"title": "Song A", "lyrics": "full lyrics", "album": {"name": "X"}},
                {"title": "Song B", "lyrics": "", "album": {"name": "X"}},  # never fetched
                {"title": "Song C", "lyrics": "orphan page"},  # no album
            ],
        }
    ]
    report_path = write_dedup_report(raw_artists, docs_dir=tmp_path)
    content = report_path.read_text()
    assert "| The Cure | 3 | 3 | 0 | 0 | 0 | 0 | 1 |" in content


def test_filter_by_year_range_keeps_inclusive_window_and_drops_undated():
    df = pd.DataFrame(
        {
            "title": ["before", "start edge", "middle", "end edge", "after", "undated"],
            "release_date": [
                "1979-01-01",
                "1980-01-01",
                "1984-06-01",
                "1989-12-31",
                "1990-01-01",
                None,
            ],
        }
    )
    filtered = filter_by_year_range(df, 1980, 1989)
    assert list(filtered["title"]) == ["start edge", "middle", "end edge"]


def test_sample_gold_set_candidates_hits_exact_total_and_drops_lyrics():
    df = pd.DataFrame(
        {
            "artist": ["The Cure"] * 30 + ["The Smiths"] * 10,
            "title": (
                [f"cure song {i}" for i in range(30)] + [f"smiths song {i}" for i in range(10)]
            ),
            "lyrics": ["some lyrics text"] * 40,
            "release_date": ["1982-01-01"] * 20 + ["1992-01-01"] * 10 + ["1984-01-01"] * 10,
        }
    )
    sampled = sample_gold_set_candidates(df, n_total=20, seed=1)
    assert len(sampled) == 20
    assert "lyrics" not in sampled.columns
    assert set(sampled["decade"].unique()) <= {1980, 1990}
    assert len(sampled) == len(sampled.drop_duplicates(subset=["artist", "title"]))


def test_download_moodylyrics_reuses_cache_and_normalizes_columns(tmp_path: Path):
    cache_dir = tmp_path / "moodylyrics"
    cache_dir.mkdir()
    (cache_dir / "moodylyrics4q.csv").write_text(
        '"Index","Artist","Title","Mood"\n"ML1","George Michael","I Want Your Sex","happy"\n'
    )
    df = download_moodylyrics(cache_dir)
    assert list(df.columns) == ["artist", "title", "mood"]
    assert df.iloc[0]["mood"] == "happy"


def test_download_dmdd_reuses_cache_and_normalizes_columns(tmp_path: Path):
    cache_dir = tmp_path / "dmdd"
    cache_dir.mkdir()
    header = "dzr_sng_id,MSD_sng_id,MSD_track_id,valence,arousal,artist_name,track_name\n"
    row = "1,MSD1,TR1,-0.5,0.3,Dark Tranquillity,Insanity's Crescendo\n"
    for split in ["train", "validation", "test"]:
        (cache_dir / f"dmdd_{split}.csv").write_text(header + row)
    df = download_dmdd(cache_dir)
    assert list(df.columns) == ["artist", "title", "valence", "arousal", "split"]
    assert set(df["split"]) == {"train", "validation", "test"}
    assert len(df) == 3


def test_build_song_album_map_keeps_every_studio_album_a_song_appears_on():
    raw_artists = [
        {
            "artist": "The Smiths",
            "songs": [
                {
                    "title": "Heaven Knows I'm Miserable Now",
                    "lyrics": "full studio lyrics",
                    "album": {"name": "Hatful of Hollow"},
                    "release_date": "1984-01-01",
                },
                {
                    "title": "Heaven Knows I'm Miserable Now",
                    "lyrics": "full studio lyrics",
                    "album": {"name": "Louder Than Bombs"},
                    "release_date": "1987-01-01",
                },
                {
                    "title": "Heaven Knows I'm Miserable Now (Live in Boston)",
                    "lyrics": "live take",
                    "album": {"name": "Rank"},
                    "release_date": "1988-01-01",
                },
            ],
        }
    ]
    song_albums = build_song_album_map(raw_artists)
    assert len(song_albums) == 2
    assert set(song_albums["album"]) == {"Hatful of Hollow", "Louder Than Bombs"}
    assert (song_albums["title"] == "Heaven Knows I'm Miserable Now").all()


def _song(title, primary="The Cure", cover_of=None, language="en", **extra):
    rels = [{"relationship_type": "cover_of", "songs": [{"primary_artist": {"name": cover_of}}]}]
    return {
        "title": title,
        "primary_artist": {"name": primary},
        "song_relationships": rels if cover_of else [],
        "language": language,
        **extra,
    }


def test_flag_song_versions_flags_other_artists_covers_and_non_english():
    flags = flag_song_versions(
        [
            _song("Own"),
            _song("Cover Of Hendrix", cover_of="The Jimi Hendrix Experience"),
            _song("Alt Version", cover_of="  the CURE "),
            _song("By Someone Else", primary="Coheed and Cambria"),
            _song("Traduzione", language="it"),
            {"title": "No Metadata"},
        ],
        artist="The Cure",
    )
    assert [f["is_other_artist"] for f in flags] == [False, False, False, True, False, False]
    assert [f["is_cover"] for f in flags] == [False, True, False, False, False, False]
    assert [f["is_non_english"] for f in flags] == [False, False, False, False, True, False]
    assert [is_studio_take(f) for f in flags] == [True, False, True, False, False, True]


def test_dedupe_prefers_bands_own_take_over_longer_other_artist_version():
    songs = [
        _song("A Forest", lyrics="short original"),
        _song("A Forest", primary="Ramin Djawadi", lyrics="a much much longer arrangement text"),
    ]
    canonical = dedupe_studio_songs(songs, artist="The Cure")
    assert [s["lyrics"] for s in canonical] == ["short original"]


def test_canonical_studio_album_maps_editions_and_rejects_compilations():
    studio = ["Disintegration", "The Queen Is Dead", "Boys Don't Cry (x)"]
    assert canonical_studio_album("Disintegration (Deluxe Edition)", studio) == "Disintegration"
    assert canonical_studio_album("The Queen is Dead", studio) == "The Queen Is Dead"
    assert canonical_studio_album("Japanese Whispers", studio) is None


def test_release_date_falls_back_to_display_year_of_song_then_album():
    assert release_date({"release_date": "1984-02-20"}) == "1984-02-20"
    assert release_date({"release_date_for_display": "December 1988"}) == "1988"
    assert release_date({"album": {"release_date_for_display": "May 8, 1979"}}) == "1979"
    assert release_date({"album": {"name": "x"}}) is None


def test_label_studio_albums_merges_editions_and_marks_other_releases():
    from lyric_emotion.config import Artist

    album_map = pd.DataFrame(
        {
            "artist": ["The Cure"] * 3 + ["The Smiths"],
            "title": ["Closedown", "Closedown", "Lovesong", "Panic"],
            "album": ["Disintegration", "Disintegration (Deluxe Edition)", "Lovesong", "Panic"],
            "release_date": ["1989"] * 4,
        }
    )
    artists = [
        Artist(name="The Cure", max_songs=1, studio_albums=["Disintegration"]),
        Artist(name="The Smiths", max_songs=1),
    ]
    out = label_studio_albums(album_map, artists)
    rows = {(r.title, r.album): r.kind for r in out.itertuples()}
    assert rows == {
        ("Closedown", "Disintegration"): "studio",
        ("Lovesong", "Lovesong"): "other",
        ("Panic", "Panic"): "studio",  # no list configured -> everything counts as studio
    }


def test_assign_album_groups_counts_studio_songs_once_and_groups_rarities_by_year():
    import pandas as pd

    from lyric_emotion.data import assign_album_groups

    songs = pd.DataFrame(
        {
            "artist": ["B", "B", "B", "B"],
            "title": ["Hit", "Deep", "B-side", "Rare"],
            "album": ["Hits (Compilation)", "Album One", "Odds", "Odds"],
            "year": [1984, 1984, 1985, 1985],
        }
    )
    album_map = pd.DataFrame(
        {
            "artist": ["B", "B", "B", "B", "B"],
            "title": ["Hit", "Hit", "Deep", "B-side", "Rare"],
            "album": ["Album One", "Hits (Compilation)", "Album One", "Odds", "Odds"],
            "kind": ["studio", "other", "studio", "other", "other"],
        }
    )
    out = assign_album_groups(songs, album_map)
    assert len(out) == 4
    assert out["album_group"].tolist() == [
        "B|Album One",
        "B|Album One",
        "B|Singles & rarities 1985",
        "B|Singles & rarities 1985",
    ]
    assert out["album"].tolist()[0] == "Album One"


def test_dedupe_prefers_original_title_and_album_over_longer_variant():
    songs = [
        {
            "title": "This Charming Man (New York Vocal)",
            "album": {"name": 'The 12" Mixes'},
            "lyrics": "a much longer lyric with extra ad libs",
        },
        {"title": "This Charming Man", "album": {"name": "The Smiths"}, "lyrics": "the lyric"},
    ]
    assert dedupe_studio_songs(songs)[0]["title"] == "This Charming Man"

    songs = [
        {"title": "Song B", "lyrics": "a longer lyric on an orphan page"},
        {"title": "Song B", "album": {"name": "Album"}, "lyrics": "lyric"},
    ]
    assert dedupe_studio_songs(songs)[0]["album"] == {"name": "Album"}


def test_studio_corpus_leaves_out_songs_without_an_album(tmp_path: Path):
    raw = [
        {
            "artist": "A",
            "songs": [
                {
                    "title": "Real",
                    "album": {"name": "X"},
                    "lyrics": "words",
                    "release_date": "1984",
                },
                {"title": "Orphan", "lyrics": "words", "release_date": "1982-01-01"},
            ],
        }
    ]
    save_studio_corpus(raw, tmp_path)
    assert pd.read_parquet(tmp_path / "studio_corpus.parquet")["title"].tolist() == ["Real"]
