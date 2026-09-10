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

from lyric_emotion.config import Artist, Config

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


def _scale_emobank(series: pd.Series) -> pd.Series:
    return (series - EMOBANK_MIN) / (EMOBANK_MAX - EMOBANK_MIN) * 2 - 1


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


def _normalized_title(title: str) -> str:
    """Strip parenthetical/bracketed suffixes (live, remix, remaster year,
    feat. ...) so near-duplicate releases collapse to the same key for the
    coverage report. Real dedup with fuzzy matching happens later; this is
    just a diagnostic count.
    """
    return re.sub(r"[\[(].*?[\])]", "", title).strip().lower()


def fetch_artist_songs(genius: Any, artist: Artist, cache_dir: Path) -> dict:
    """Fetch every song for one artist via lyricsgenius, or reuse the cached
    raw response if one already exists — safe to re-run after a crash
    without re-downloading artists that already completed.
    """
    cache_path = cache_dir / f"{_slugify(artist.name)}.json"
    if cache_path.exists():
        return _load_json(cache_path)

    genius_artist = genius.search_artist(artist.name, max_songs=artist.max_songs, sort="title")
    raw = {
        "artist": artist.name,
        "requested": artist.max_songs,
        "fetched_at": datetime.now(UTC).isoformat(),
        "songs": [song.to_dict() for song in genius_artist.songs] if genius_artist else [],
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
    lines.append("| artist | requested | fetched | with lyrics | with album | with year"
                  " | excluded | duplicate titles |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for raw in raw_artists:
        songs = raw["songs"]
        with_lyrics = sum(1 for s in songs if s.get("lyrics", "").strip())
        with_album = sum(1 for s in songs if s.get("album"))
        with_year = sum(1 for s in songs if s.get("release_date"))
        excluded = sum(
            1 for s in songs if s.get("instrumental") or not s.get("lyrics", "").strip()
        )
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


def save_training_data(config: Config, output_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = output_dir or Path(config.paths.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emotions_path = out_dir / "emotions_train.parquet"
    vad_path = out_dir / "vad_train.parquet"

    build_emotions_dataset().to_parquet(emotions_path, index=False)
    build_vad_dataset().to_parquet(vad_path, index=False)

    return emotions_path, vad_path


def write_data_report(
    emotions_path: Path, vad_path: Path, docs_dir: Path = Path("docs")
) -> Path:
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
