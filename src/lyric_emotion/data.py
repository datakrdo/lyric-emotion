"""Build redistributable training data for the two heads.

- `emotions_train.parquet`: text + a 28-wide binary label vector, from
  GoEmotions (Apache-2.0, official train/validation/test split).
- `vad_train.parquet`: text + valence/arousal/dominance scaled to [-1, 1],
  from EmoBank (CC BY-SA 4.0, official split added in 2019).

Both datasets ship their own split column; we use it as-is rather than
inventing one, per the project's reproducibility rule.
"""

import io
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from datasets import load_dataset
from rapidfuzz import fuzz

from lyric_emotion.config import Acquisition, Artist, Comparison, Config, comparison_window

EMOBANK_URL = "https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/emobank.csv"
GOEMOTIONS_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]
# EmoBank scores are on a 1-5 Likert scale; rescale to [-1, 1] to match the
# sentiment_score convention used later in the pipeline.
EMOBANK_MIN, EMOBANK_MAX = 1.0, 5.0

# Weak labels for Phase 4 domain adaptation — song IDs + mood/VAD tags only,
# no lyrics text (per THIRD_PARTY_NOTICES.md, neither dataset grants
# redistribution rights). Mirrored via a third-party GitHub repo since the
# original host (softeng.polito.it) has no stable direct-download URL.
MOODYLYRICS_URL = (
    "https://raw.githubusercontent.com/sgiammy/emotion-patterns-in-music-playlists/"
    "master/src/datasets/MoodyLyrics4Q.csv"
)
DMDD_BASE_URL = "https://raw.githubusercontent.com/deezer/deezer_mood_detection_dataset/master"
DMDD_SPLITS = ["train", "validation", "test"]


def _scale_emobank(series: pd.Series) -> pd.Series:
    return (series - EMOBANK_MIN) / (EMOBANK_MAX - EMOBANK_MIN) * 2 - 1


def download_moodylyrics(cache_dir: Path) -> pd.DataFrame:
    """MoodyLyrics4Q: 2000 songs tagged with one of Russell's 4 quadrants
    (happy/angry/sad/relaxed) from Last.fm tags. Cached locally, never
    committed.
    """
    dest = cache_dir / "moodylyrics4q.csv"
    if not dest.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        response = requests.get(MOODYLYRICS_URL, timeout=60)
        response.raise_for_status()
        dest.write_bytes(response.content)
    df = pd.read_csv(dest)
    return df.rename(columns={"Artist": "artist", "Title": "title", "Mood": "mood"})[
        ["artist", "title", "mood"]
    ]


def download_dmdd(cache_dir: Path) -> pd.DataFrame:
    """Deezer Mood Detection Dataset: ~18.6k tracks with continuous
    valence/arousal derived from Last.fm tags, official train/validation/test
    split. Cached locally, never committed.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for split in DMDD_SPLITS:
        dest = cache_dir / f"dmdd_{split}.csv"
        if not dest.exists():
            response = requests.get(f"{DMDD_BASE_URL}/{split}.csv", timeout=60)
            response.raise_for_status()
            dest.write_bytes(response.content)
        split_df = pd.read_csv(dest)
        split_df["split"] = split
        frames.append(split_df)
    combined = pd.concat(frames, ignore_index=True)
    return combined.rename(columns={"artist_name": "artist", "track_name": "title"})[
        ["artist", "title", "valence", "arousal", "split"]
    ]


# Russell's four quadrants as weak VAD anchors (arousal-only axis, since
# MoodyLyrics has no dominance label) — a coarse categorical stand-in, not a
# measured score.
_MOOD_TO_VA = {
    "happy": (0.7, 0.7),
    "angry": (-0.7, 0.7),
    "sad": (-0.7, -0.7),
    "relaxed": (0.7, -0.7),
}


# DMDD scores are z-scores (observed range about -2.3..2.8); a fixed linear
# rescale keeps their shape, where the old clip to +-1 flattened 46% of valence.
DMDD_Z_SCALE = 2.0


def fetch_domain_lyrics(
    pairs: pd.DataFrame, cache_dir: Path, acquisition: "Acquisition | None" = None
) -> dict[tuple[str, str], "str | None"]:
    """Look up lyrics on Genius for each unique (artist, title) pair, caching
    one JSON file per song so an interrupted run can resume without
    re-fetching. Returns None for songs Genius has no match for.
    """
    import lyricsgenius

    token = os.environ.get("GENIUS_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("GENIUS_ACCESS_TOKEN not set — check your .env")
    genius = lyricsgenius.Genius(
        token,
        timeout=acquisition.timeout_seconds if acquisition else 30,
        sleep_time=acquisition.request_sleep_seconds if acquisition else 1.0,
        retries=acquisition.max_retries if acquisition else 3,
        skip_non_songs=True,
    )
    genius.verbose = False

    cache_dir.mkdir(parents=True, exist_ok=True)
    lyrics_by_pair: dict[tuple[str, str], str | None] = {}
    for artist, title in pairs[["artist", "title"]].drop_duplicates().itertuples(index=False):
        cache_path = cache_dir / f"{_slugify(artist)}__{_slugify(title)}.json"
        if cache_path.exists():
            lyrics_by_pair[(artist, title)] = _load_json(cache_path).get("lyrics")
            continue
        song = genius.search_song(title, artist)
        lyrics = song.lyrics if song else None
        cache_path.write_text(_dumps({"artist": artist, "title": title, "lyrics": lyrics}))
        lyrics_by_pair[(artist, title)] = lyrics
    return lyrics_by_pair


def build_domain_dataset(config: Config) -> pd.DataFrame:
    """Weak-labeled domain-adaptation set for Phase 4: MoodyLyrics (full,
    mood quadrant -> valence/arousal) plus a sample of DMDD (native
    valence/arousal, z-scores divided by a fixed `DMDD_Z_SCALE`), joined
    with lyrics fetched from Genius. Songs Genius has no match for are
    dropped, as are case-study artists and split leaks.
    """
    cache_dir = Path(config.paths.cache_dir)
    moody = download_moodylyrics(cache_dir)
    moody["valence"] = moody["mood"].map(lambda m: _MOOD_TO_VA[m][0])
    moody["arousal"] = moody["mood"].map(lambda m: _MOOD_TO_VA[m][1])
    moody["split"] = "train"
    moody["source"] = "moodylyrics"

    dmdd = download_dmdd(cache_dir)
    dmdd = dmdd.sample(
        n=min(config.domain_adaptation.dmdd_sample_size, len(dmdd)),
        random_state=config.project.random_seed,
    ).copy()
    dmdd["valence"] = dmdd["valence"] / DMDD_Z_SCALE
    dmdd["arousal"] = dmdd["arousal"] / DMDD_Z_SCALE
    dmdd["source"] = "dmdd"

    combined = pd.concat(
        [moody[["artist", "title", "valence", "arousal", "split", "source"]], dmdd],
        ignore_index=True,
    )

    lyrics_by_pair = fetch_domain_lyrics(
        combined, Path(config.paths.raw_dir) / "domain_lyrics", config.acquisition
    )
    combined["text"] = combined.apply(
        lambda row: lyrics_by_pair.get((row["artist"], row["title"])), axis=1
    )
    return clean_domain_set(
        combined.dropna(subset=["text"]), config.domain_adaptation.exclude_artists
    )


def clean_domain_set(df: pd.DataFrame, exclude_artists: list[str]) -> pd.DataFrame:
    """Drop case-study artists, split leaks and repeated training texts."""
    banned = {_norm_name(a) for a in exclude_artists}
    keep = [not any(b in _norm_name(a) for b in banned) for a in df["artist"]]
    df = drop_split_leaks(df[keep])
    repeated = (df["split"] == "train") & df.duplicated(subset="text")
    return df[~repeated].reset_index(drop=True)


def drop_split_leaks(df: pd.DataFrame) -> pd.DataFrame:
    """Drop test rows whose text also sits in train/validation, and validation
    rows whose text also sits in train (the same lyric under two Genius
    entries) — an evaluation split must not contain training text."""
    train = set(df.loc[df["split"] == "train", "text"])
    validation = set(df.loc[df["split"] == "validation", "text"])
    leak = ((df["split"] == "test") & df["text"].isin(train | validation)) | (
        (df["split"] == "validation") & df["text"].isin(train)
    )
    return df[~leak].reset_index(drop=True)


def write_domain_data_report(df: pd.DataFrame, docs_dir: Path = Path("docs")) -> Path:
    """Coverage + class balance for the domain-adaptation set, per the
    project's rule that acquisition gets reviewed before training on it.
    """
    lines = ["# Domain adaptation data report", ""]
    lines.append("| source | rows | valence mean | arousal mean |")
    lines.append("|---|---|---|---|")
    for source, group in df.groupby("source"):
        lines.append(
            f"| {source} | {len(group)} | {group['valence'].mean():.3f} "
            f"| {group['arousal'].mean():.3f} |"
        )
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_path = docs_dir / "domain_data.md"
    report_path.write_text("\n".join(lines))
    return report_path


def build_emotions_dataset() -> pd.DataFrame:
    """Download GoEmotions and return a flat text + 28-column binary frame."""
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    frames = []
    for split_name, split in ds.items():
        df = split.to_pandas()
        labels = pd.DataFrame(
            [[1 if i in row else 0 for i in range(len(GOEMOTIONS_LABELS))] for row in df["labels"]],
            columns=GOEMOTIONS_LABELS,
        )
        frame = pd.concat([df[["id", "text"]], labels], axis=1)
        frame["split"] = split_name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_vad_dataset() -> pd.DataFrame:
    """Download EmoBank and return text + scaled valence/arousal/dominance."""
    response = requests.get(EMOBANK_URL, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df = df.rename(columns={"V": "valence", "A": "arousal", "D": "dominance"})
    for col in ("valence", "arousal", "dominance"):
        df[col] = _scale_emobank(df[col])
    return df[["id", "split", "text", "valence", "arousal", "dominance"]]


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


_SECTION_TAG = re.compile(r"^\s*\[[^\]]+\]\s*$", re.M)


def split_sections(lyrics: str) -> list[str]:
    """Split a Genius lyric sheet into its labeled sections ([Verse], [Chorus],
    ...), dropping the tag lines themselves and any section whose normalized
    text repeats one already kept — a repeated chorus should count once, per
    `annotation/GUIDELINES.md`. Falls back to blank-line-separated paragraphs
    for the rare lyric sheet with no section tags at all.
    """
    if _SECTION_TAG.search(lyrics):
        chunks = _SECTION_TAG.split(lyrics)
    else:
        chunks = re.split(r"\n\s*\n", lyrics)

    sections: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        text = chunk.strip()
        if not text:
            continue
        key = re.sub(r"\s+", " ", text.lower())
        if key in seen:
            continue
        seen.add(key)
        sections.append(text)
    return sections or [lyrics.strip()]


def _normalized_title(title: str) -> str:
    """Strip parenthetical/bracketed suffixes (live, remix, remaster year,
    feat. ...) so near-duplicate releases collapse to the same key for the
    coverage report. Real dedup with fuzzy matching happens later; this is
    just a diagnostic count.
    """
    return re.sub(r"[\[(].*?[\])]", "", title).strip().lower()


# Keyword patterns matched against a song's parenthetical/bracketed suffix
# (e.g. "10:15 Saturday Night (Chestnut Studio Demo)") to flag non-studio
# versions. "version"/"mix" are broad on purpose: Genius uses them for
# everything from alternate arrangements to anniversary reissues, and any
# of those should lose to a plain studio take when deduping.
_VERSION_PATTERNS = {
    "is_live": re.compile(r"\blive\b|\bsession\b|\brehearsal\b", re.I),
    "is_demo": re.compile(r"\bdemo\b", re.I),
    "is_remix": re.compile(r"\bremix\b|\bmix\b|\bedit\b|\bextended\b", re.I),
    "is_reissue": re.compile(r"\bremaster\w*\b|\breissue\b|\banniversary\b|\bversion\b", re.I),
}


def _version_flags(title: str, album_name: str) -> dict[str, bool]:
    """Match against the title's suffix AND the album name — a whole live
    album (e.g. a compilation of concert recordings) won't necessarily
    suffix every track title with "(Live ...)", so the album name is a
    second, independent signal.
    """
    suffix = " ".join(re.findall(r"[\[(](.*?)[\])]", title))
    haystack = f"{suffix} {album_name}"
    return {flag: bool(pattern.search(haystack)) for flag, pattern in _VERSION_PATTERNS.items()}


def _norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def _is_cover_of_other(song: dict, artist: str) -> bool:
    """Genius `cover_of` points at the original recording; it's a foreign cover
    only if that original's primary artist isn't the band itself (an alternate
    take of the band's own song also carries `cover_of`).
    """
    for rel in song.get("song_relationships") or []:
        if rel.get("relationship_type") != "cover_of":
            continue
        for target in rel.get("songs") or []:
            target_artist = (target.get("primary_artist") or {}).get("name") or ""
            if _norm_name(target_artist) != _norm_name(artist):
                return True
    return False


def flag_song_versions(
    songs: list[dict], non_studio_albums: "set[str] | None" = None, artist: "str | None" = None
) -> list[dict]:
    """Tag each song with is_live/is_demo/is_remix/is_reissue based on its
    title suffix and album name, plus its normalized (suffix-stripped)
    title for grouping. `non_studio_albums` is a case-insensitive set of
    album names (e.g. "Rank") known to be live/session recordings whose
    name carries no detectable keyword — force-flags is_live for those.
    With `artist`, also flags provenance: `is_other_artist` (Genius lists
    songs where the band is only credited, e.g. covers by others),
    `is_cover` (the band covering someone else) and `is_non_english`.
    """
    non_studio_albums = {a.lower() for a in (non_studio_albums or set())}
    flagged = []
    for song in songs:
        song = dict(song)
        album_name = (song.get("album") or {}).get("name") or ""
        song.update(_version_flags(song.get("title", ""), album_name))
        if album_name.lower() in non_studio_albums:
            song["is_live"] = True
        primary = (song.get("primary_artist") or {}).get("name")
        song["is_other_artist"] = bool(
            artist and primary and _norm_name(primary) != _norm_name(artist)
        )
        song["is_cover"] = bool(artist and _is_cover_of_other(song, artist))
        song["is_non_english"] = (song.get("language") or "en") != "en"
        song["normalized_title"] = _normalized_title(song.get("title", ""))
        flagged.append(song)
    return flagged


_NOT_STUDIO_FLAGS = (
    "is_live",
    "is_demo",
    "is_remix",
    "is_reissue",
    "is_other_artist",
    "is_cover",
    "is_non_english",
)


def is_studio_take(song: dict) -> bool:
    """A song that is the band's own, English, studio recording (as far as the
    metadata can tell). Single definition used by fetching, dedup, the album
    map, the reports and the corpus."""
    return not any(song.get(flag) for flag in _NOT_STUDIO_FLAGS)


def in_corpus(song: dict) -> bool:
    """Studio take with lyrics on a known album. A Genius page with no album
    is not a release we can place (misattributed and fan pages look like this)."""
    return (
        is_studio_take(song)
        and bool((song.get("lyrics") or "").strip())
        and bool((song.get("album") or {}).get("name"))
    )


def canonical_studio_album(name: str, studio_albums: list[str]) -> "str | None":
    """Map an album name (incl. 'X (Deluxe Edition)') to its entry in the
    explicit studio-album list, or None for singles/compilations/others."""
    base = _slugify(re.sub(r"[\[(].*?[\])]", "", name))
    return next((a for a in studio_albums if _slugify(a) == base), None)


def label_studio_albums(album_map: pd.DataFrame, artists: "list[Artist]") -> pd.DataFrame:
    """Fold edition variants into their studio album and add `kind`
    ("studio"/"other") to a song->album map."""
    studio = {a.name: a.studio_albums for a in artists}
    out = album_map.copy()
    canonical = [
        canonical_studio_album(album, studio.get(artist, []))
        for artist, album in zip(out["artist"], out["album"], strict=True)
    ]
    out["kind"] = [
        "studio" if c or not studio.get(a) else "other"
        for c, a in zip(canonical, out["artist"], strict=True)
    ]
    out["album"] = [c or album for c, album in zip(canonical, out["album"], strict=True)]
    return out.drop_duplicates(ignore_index=True)


def assign_album_groups(songs: pd.DataFrame, album_map: pd.DataFrame) -> pd.DataFrame:
    """Add `album_group`, the unit songs are resampled/permuted in. A song on a
    studio album joins that album (once, whichever edition or single the dedupe
    kept); one found only on singles/compilations joins its band's
    "Singles & rarities <year>" group. `songs`: artist, title, album, year;
    `album_map`: `label_studio_albums` output. Replaces `album` with the studio
    album where there is one."""
    studio = (
        album_map[album_map["kind"] == "studio"]
        .drop_duplicates(["artist", "title"])[["artist", "title", "album"]]
        .rename(columns={"album": "studio_album"})
    )
    out = songs.merge(studio, on=["artist", "title"], how="left")
    has = out["studio_album"].notna()
    rarities = out["artist"] + "|Singles & rarities " + out["year"].astype("Int64").astype(str)
    out["album_group"] = (out["artist"] + "|" + out["studio_album"]).where(has, rarities)
    out["album"] = out["studio_album"].where(has, out["album"])
    return out.drop(columns="studio_album")


def release_date(song: dict) -> "str | None":
    """Song's own ISO date, else the year from Genius' display strings (song,
    then album) — ~1 in 30 songs has an album but no machine-readable date."""
    if song.get("release_date"):
        return str(song["release_date"])
    for text in (
        song.get("release_date_for_display"),
        (song.get("album") or {}).get("release_date_for_display"),
    ):
        match = re.search(r"\b(19|20)\d{2}\b", text or "")
        if match:
            return match.group(0)
    return None


def _cluster_by_title(flagged_songs: list[dict], fuzzy_threshold: float) -> list[list[dict]]:
    clusters: list[list[dict]] = []
    for song in flagged_songs:
        cluster = next(
            (
                c
                for c in clusters
                if fuzz.token_sort_ratio(song["normalized_title"], c[0]["normalized_title"])
                >= fuzzy_threshold
            ),
            None,
        )
        if cluster is not None:
            cluster.append(song)
        else:
            clusters.append([song])
    return clusters


def _preferred_take(candidates: list[dict]) -> dict:
    """The version of a song to keep: the plain title over a "(New York Vocal)"
    style variant, a page with an album over an orphan one, then the longest."""
    return max(
        candidates,
        key=lambda s: (
            not re.search(r"[\[(]", s.get("title", "")),
            bool((s.get("album") or {}).get("name")),
            len(s.get("lyrics") or ""),
        ),
    )


def dedupe_studio_songs(
    songs: list[dict],
    fuzzy_threshold: float = 90.0,
    non_studio_albums: "set[str] | None" = None,
    artist: "str | None" = None,
) -> list[dict]:
    """Collapse near-duplicate titles (live/demo/remix/reissue variants, plus
    fuzzy title drift like punctuation) into one canonical version per song,
    preferring a studio take when one exists in the group. This gives one
    text per song for model training/inference — it is NOT the right table
    for per-album aggregation, since it keeps only one album. Use
    `build_song_album_map` for that instead.
    """
    clusters = _cluster_by_title(
        flag_song_versions(songs, non_studio_albums, artist), fuzzy_threshold
    )
    canonical_songs = []
    for cluster in clusters:
        studio_takes = [s for s in cluster if is_studio_take(s)]
        candidates = studio_takes or cluster
        canonical_songs.append(_preferred_take(candidates))
    return canonical_songs


def build_song_album_map(
    raw_artists: list[dict],
    fuzzy_threshold: float = 90.0,
    artists: "list[Artist] | None" = None,
) -> pd.DataFrame:
    """Map each canonical studio song to every album it was released on
    (original album, singles, compilations). A song re-released on a later
    compilation should count toward that album's mood too, not just its
    first release — this is deliberately more permissive than
    `dedupe_studio_songs`, which collapses to one text per song.
    """
    non_studio_by_artist = {a.name: set(a.non_studio_albums) for a in (artists or [])}
    rows = []
    for raw in raw_artists:
        non_studio_albums = non_studio_by_artist.get(raw["artist"])
        clusters = _cluster_by_title(
            flag_song_versions(raw["songs"], non_studio_albums, raw["artist"]), fuzzy_threshold
        )
        for cluster in clusters:
            studio_takes = [s for s in cluster if is_studio_take(s)]
            if not studio_takes:
                continue
            canonical_title = _preferred_take(studio_takes)["title"]
            for song in studio_takes:
                album = (song.get("album") or {}).get("name")
                if not album:
                    continue
                rows.append(
                    {
                        "artist": raw["artist"],
                        "title": canonical_title,
                        "album": album,
                        "release_date": release_date(song),
                    }
                )
    return pd.DataFrame(rows, columns=["artist", "title", "album", "release_date"]).drop_duplicates(
        ignore_index=True
    )


def _find_artist_id(genius: Any, artist_name: str) -> "int | None":
    """Resolve an artist's Genius ID via a lightweight artist search — no
    songs (and no lyrics) fetched yet.
    """
    response = genius.search_artists(artist_name)
    sections = response.get("sections") or []
    artist_section = next((s for s in sections if s.get("type") == "artist"), None)
    hits = [h["result"] for h in (artist_section["hits"] if artist_section else [])]
    if not hits:
        return None
    exact = next((h for h in hits if h["name"].lower() == artist_name.lower()), None)
    return int((exact or hits[0])["id"])


def _list_artist_songs_metadata(genius: Any, artist_name: str, max_songs: int) -> list[dict]:
    """List up to `max_songs` of an artist's songs with full metadata (title,
    album, release date) but without scraping any lyrics yet. `artist_songs()`
    is a cheap listing endpoint and `song()` adds album/release date per
    song — both far cheaper than the lyrics page scrape, which
    `fetch_artist_songs` defers until it knows which songs are actually
    studio takes worth keeping.
    """
    artist_id = _find_artist_id(genius, artist_name)
    if artist_id is None:
        return []

    stubs: list[dict] = []
    page: int | None = 1
    while page and len(stubs) < max_songs:
        response = genius.artist_songs(artist_id, per_page=50, page=page)
        stubs.extend(response["songs"])
        page = response.get("next_page")
    stubs = stubs[:max_songs]

    excluded_terms = genius.excluded_terms
    songs = []
    for stub in stubs:
        if stub.get("lyrics_state") != "complete" or stub.get("instrumental"):
            continue
        title = stub.get("title", "").lower()
        if any(term in title for term in excluded_terms):
            continue
        songs.append(genius.song(stub["id"])["song"])
    return songs


def fetch_artist_songs(genius: Any, artist: Artist, cache_dir: Path) -> dict:
    """Fetch every song for one artist, or reuse the cached raw response if
    one already exists — safe to re-run after a crash without
    re-downloading artists that already completed.

    Lyrics text (the expensive, copyrighted part) is fetched only for songs
    dedup keeps as a studio take — live/demo/remix/reissue versions and
    duplicate-title losers never get their lyrics page scraped at all.
    """
    cache_path = cache_dir / f"{_slugify(artist.name)}.json"
    if cache_path.exists():
        return _load_json(cache_path)

    songs = _list_artist_songs_metadata(genius, artist.name, artist.max_songs)
    flagged = flag_song_versions(songs, set(artist.non_studio_albums), artist.name)
    for cluster in _cluster_by_title(flagged, fuzzy_threshold=90.0):
        studio_takes = [s for s in cluster if is_studio_take(s)]
        for song in studio_takes:
            song["lyrics"] = genius.lyrics(song_url=song.get("url")) or ""
    for song in flagged:
        song.setdefault("lyrics", "")

    raw = {
        "artist": artist.name,
        "requested": artist.max_songs,
        "fetched_at": datetime.now(UTC).isoformat(),
        "songs": flagged,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(_dumps(raw))
    return raw


def _dumps(obj: dict) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text())


def fetch_all_artists(config: Config, artists: list[Artist]) -> list[dict]:
    """Fetch (or load from cache) every configured artist's raw Genius data."""
    import lyricsgenius

    token = os.environ.get("GENIUS_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("GENIUS_ACCESS_TOKEN not set — check your .env")

    genius = lyricsgenius.Genius(
        token,
        timeout=config.acquisition.timeout_seconds,
        sleep_time=config.acquisition.request_sleep_seconds,
        retries=config.acquisition.max_retries,
        skip_non_songs=True,
    )
    cache_dir = Path(config.paths.raw_dir) / "genius"
    return [fetch_artist_songs(genius, artist, cache_dir) for artist in artists]


def write_fetch_coverage_report(raw_artists: list[dict], docs_dir: Path = Path("docs")) -> Path:
    """Per-artist coverage: requested vs fetched vs usable, per the project's
    rule that acquisition is reviewed before any downstream phase starts.
    """
    lines = ["# Lyrics acquisition coverage report", ""]
    lines.append(
        "| artist | requested | fetched | with lyrics | with album | with year"
        " | excluded | duplicate titles |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")

    for raw in raw_artists:
        songs = raw["songs"]
        with_lyrics = sum(1 for s in songs if s.get("lyrics", "").strip())
        with_album = sum(1 for s in songs if s.get("album"))
        with_year = sum(1 for s in songs if s.get("release_date"))
        excluded = sum(1 for s in songs if s.get("instrumental") or not s.get("lyrics", "").strip())
        titles = [_normalized_title(s.get("title", "")) for s in songs]
        duplicates = len(titles) - len(set(titles))
        lines.append(
            f"| {raw['artist']} | {raw['requested']} | {len(songs)} | {with_lyrics} "
            f"| {with_album} | {with_year} | {excluded} | {duplicates} |"
        )

    docs_dir.mkdir(parents=True, exist_ok=True)
    report_path = docs_dir / "fetch_coverage.md"
    report_path.write_text("\n".join(lines))
    return report_path


def write_dedup_report(
    raw_artists: list[dict], docs_dir: Path = Path("docs"), artists: "list[Artist] | None" = None
) -> Path:
    """Per-artist dedup summary: how many raw entries collapse into how many
    canonical songs, and how many of those canonical songs are themselves
    non-studio (no studio take existed in their duplicate group).
    """
    non_studio_by_artist = {a.name: set(a.non_studio_albums) for a in (artists or [])}
    lines = ["# Lyrics dedup report", ""]
    lines.append("| artist | raw | canonical | live | demo | remix | reissue | studio |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for raw in raw_artists:
        canonical = dedupe_studio_songs(
            raw["songs"],
            non_studio_albums=non_studio_by_artist.get(raw["artist"]),
            artist=raw["artist"],
        )
        live = sum(1 for s in canonical if s["is_live"])
        demo = sum(1 for s in canonical if s["is_demo"])
        remix = sum(1 for s in canonical if s["is_remix"])
        reissue = sum(1 for s in canonical if s["is_reissue"])
        studio = sum(1 for s in canonical if in_corpus(s))
        lines.append(
            f"| {raw['artist']} | {len(raw['songs'])} | {len(canonical)} | {live} | {demo} "
            f"| {remix} | {reissue} | {studio} |"
        )

    docs_dir.mkdir(parents=True, exist_ok=True)
    report_path = docs_dir / "dedup_report.md"
    report_path.write_text("\n".join(lines))
    return report_path


def save_studio_corpus(
    raw_artists: list[dict],
    output_dir: Path,
    comparison: "Comparison | None" = None,
    artists: "list[Artist] | None" = None,
) -> Path:
    """Write the deduped, studio-only corpus used by the main analysis
    (Phase 5) to data/interim/studio_corpus.parquet. If `comparison` is
    given, also filter to its [start_year, end_year] window and write
    data/interim/studio_corpus_comparison.parquet.
    """
    non_studio_by_artist = {a.name: set(a.non_studio_albums) for a in (artists or [])}
    rows = []
    for raw in raw_artists:
        canonical = dedupe_studio_songs(
            raw["songs"],
            non_studio_albums=non_studio_by_artist.get(raw["artist"]),
            artist=raw["artist"],
        )
        for song in canonical:
            if not in_corpus(song):
                continue
            rows.append(
                {
                    "artist": raw["artist"],
                    "title": song["title"],
                    "lyrics": song["lyrics"],
                    "album": (song.get("album") or {}).get("name"),
                    "release_date": release_date(song),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    output_path = output_dir / "studio_corpus.parquet"
    df.to_parquet(output_path, index=False)

    album_map_path = output_dir / "song_albums.parquet"
    build_song_album_map(raw_artists, artists=artists).to_parquet(album_map_path, index=False)

    if comparison is not None:
        comparison_path = output_dir / "studio_corpus_comparison.parquet"
        filter_by_year_range(df, *comparison_window(comparison, artists or [])).to_parquet(
            comparison_path, index=False
        )

    return output_path


def sample_gold_set_candidates(
    df: pd.DataFrame, n_total: int = 200, seed: int = 42
) -> pd.DataFrame:
    """Stratify the studio corpus by (artist, decade) and sample a gold-set
    candidate list proportional to each stratum's size (largest-remainder
    method, so allocations sum exactly to n_total). Model-prediction
    stratification (per the plan) is added once Phase 2's model exists —
    this covers the two dimensions available today.
    """
    df = df.dropna(subset=["release_date"]).copy()
    df["decade"] = (df["release_date"].str.slice(0, 4).astype(int) // 10 * 10).astype(int)

    sizes = df.groupby(["artist", "decade"]).size()
    exact = sizes / sizes.sum() * n_total
    alloc = exact.astype(int)
    remainder = n_total - alloc.sum()
    if remainder > 0:
        top_up = (exact - alloc).sort_values(ascending=False).index[:remainder]
        alloc.loc[top_up] += 1
    alloc = alloc.clip(upper=sizes)

    rng_seed = seed
    sampled = [
        group.sample(n=int(alloc[key]), random_state=rng_seed)
        for key, group in df.groupby(["artist", "decade"])
        if alloc[key] > 0
    ]
    return (
        pd.concat(sampled)
        .sample(frac=1, random_state=rng_seed)
        .reset_index(drop=True)
        .drop(columns="lyrics")
    )


def write_gold_set_candidates(
    df: pd.DataFrame, path: Path = Path("annotation/gold_set_candidates.csv")
) -> Path:
    """Write the sampled candidates (no lyrics text, per the project's
    copyright rule) for manual annotation into gold_set.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def filter_by_year_range(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Keep only rows whose release_date falls within [start_year, end_year]
    (inclusive). Rows with no release_date are dropped since they can't be
    placed in the window.
    """
    years = pd.to_numeric(df["release_date"].str.slice(0, 4), errors="coerce")
    return df[years.between(start_year, end_year)].reset_index(drop=True)


def save_training_data(config: Config, output_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = output_dir or Path(config.paths.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emotions_path = out_dir / "emotions_train.parquet"
    vad_path = out_dir / "vad_train.parquet"

    build_emotions_dataset().to_parquet(emotions_path, index=False)
    build_vad_dataset().to_parquet(vad_path, index=False)

    return emotions_path, vad_path


def write_data_report(emotions_path: Path, vad_path: Path, docs_dir: Path = Path("docs")) -> Path:
    """Report split sizes and class distribution, per the project's rule
    that no analysis starts before this kind of coverage check is reviewed.
    """
    emotions = pd.read_parquet(emotions_path)
    vad = pd.read_parquet(vad_path)

    lines = ["# Training data report", ""]

    lines.append("## GoEmotions (emotions head)")
    lines.append("")
    lines.append("| split | rows |")
    lines.append("|---|---|")
    for split_name, count in emotions["split"].value_counts().items():
        lines.append(f"| {split_name} | {count} |")
    lines.append("")
    lines.append("Positive-label count per emotion (train split):")
    lines.append("")
    lines.append("| emotion | positives |")
    lines.append("|---|---|")
    train = emotions[emotions["split"] == "train"]
    for label in GOEMOTIONS_LABELS:
        lines.append(f"| {label} | {int(train[label].sum())} |")
    lines.append("")

    lines.append("## EmoBank (VAD head)")
    lines.append("")
    lines.append("| split | rows | valence mean | arousal mean | dominance mean |")
    lines.append("|---|---|---|---|---|")
    for split_name, group in vad.groupby("split"):
        lines.append(
            f"| {split_name} | {len(group)} | {group['valence'].mean():.3f} | "
            f"{group['arousal'].mean():.3f} | {group['dominance'].mean():.3f} |"
        )
    lines.append("")

    docs_dir.mkdir(parents=True, exist_ok=True)
    report_path = docs_dir / "data.md"
    report_path.write_text("\n".join(lines))
    return report_path
