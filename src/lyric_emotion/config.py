"""Load and validate config/config.yaml and config/artists.yaml.

No analytical threshold or hyperparameter should be hardcoded in
production scripts — everything comes from these two YAML files.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


class Paths(BaseModel):
    raw_dir: str
    interim_dir: str
    processed_dir: str
    cache_dir: str
    output_dir: str


class Acquisition(BaseModel):
    source: str
    request_sleep_seconds: float
    max_retries: int
    retry_backoff_seconds: float
    timeout_seconds: int
    save_raw_responses: bool


class Training(BaseModel):
    base_model: str
    max_length: int
    # GoEmotions/EmoBank texts are short comments/sentences (p99 well under
    # 100 tokens) but a handful of outliers tokenize to 1000+ — grouping
    # those into a length-sorted batch at `max_length` OOMs a small GPU for
    # no benefit, so the short-text heads truncate much tighter.
    short_text_max_length: int = 128
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    epochs: int
    precision: str
    use_gpu_if_available: bool
    gradient_checkpointing: bool = False


class Models(BaseModel):
    emotions_dir: str
    vad_dir: str
    vad_domain_dir: str
    # Which VAD checkpoint the case study (`analyze`) scores songs with.
    # Set from `evaluate-baselines`' vad-checkpoint comparison, not by hand.
    case_study_vad: str = "vad"
    # Unit the case-study VAD checkpoint scores: whole lyric ("full", how
    # vad_domain was trained) or its sections ("sections", how the short-text
    # stage-1 head was trained). Set from the checkpoint comparison.
    case_study_vad_unit: str = "sections"
    # Checkpoints (e.g. one per training seed) whose VAD scores are averaged
    # for the case study; empty = score with `case_study_vad` alone.
    case_study_vad_ensemble: list[str] = []


class EmotionHead(BaseModel):
    dataset: str
    num_labels: int
    loss: str


class VadHead(BaseModel):
    dataset: str
    dims: list[str]
    loss: str


class DomainAdaptation(BaseModel):
    dmdd_sample_size: int
    # Case-study artists never enter the domain set (train/test contamination).
    exclude_artists: list[str] = []


class Aggregation(BaseModel):
    min_songs_per_album: int
    min_songs_per_year: int


class Comparison(BaseModel):
    # None = the span in which every artist was active (see `comparison_window`).
    start_year: int | None = None
    end_year: int | None = None


class Statistics(BaseModel):
    confidence_level: float
    bootstrap_iterations: int
    alpha: float
    multiple_testing_correction: str
    # Album-level permutations / cross-validation repeats for the
    # cluster-aware inference (songs on one album are not independent).
    permutations: int = 2000
    cv_permutations: int = 200
    cv_repeats: int = 5


class Visualization(BaseModel):
    dpi: int
    compare_periods: list[list[int]]


class Project(BaseModel):
    name: str
    random_seed: int
    language: str


class Config(BaseModel):
    project: Project
    paths: Paths
    acquisition: Acquisition
    training: Training
    emotion_head: EmotionHead
    vad_head: VadHead
    domain_adaptation: DomainAdaptation
    models: Models
    aggregation: Aggregation
    comparison: Comparison
    statistics: Statistics
    visualization: Visualization
    # Broad emotion families: single source for the gold-set categories and
    # the dashboard's headline claims. Each family score = max of its labels.
    emotion_families: dict[str, list[str]] = {}


class Artist(BaseModel):
    name: str
    max_songs: int
    # Albums that are live/session recordings but whose name doesn't contain
    # a detectable keyword (e.g. The Smiths' "Rank"), so title/album regex
    # matching can't catch them — listed explicitly per the project's rule
    # that acquisition data gets reviewed, not just pattern-matched.
    non_studio_albums: list[str] = []
    # Explicit studio-album list (deluxe/vinyl editions fold into their base
    # album); everything else a song appears on is labeled "other" (singles,
    # compilations, tributes). Empty = no distinction.
    studio_albums: list[str] = []
    # First and last year of activity (None = still active); the fair
    # comparison window is the span where all artists overlap.
    active_from: int | None = None
    active_to: int | None = None


class ArtistsConfig(BaseModel):
    artists: list[Artist]


def comparison_window(comparison: Comparison, artists: "list[Artist]") -> tuple[int, int]:
    """[start, end] years to compare: explicit config values, else the overlap
    of the artists' active years."""
    starts = [a.active_from for a in artists if a.active_from is not None]
    ends = [a.active_to for a in artists if a.active_to is not None]
    start = (
        comparison.start_year if comparison.start_year is not None else max(starts, default=None)
    )
    end = comparison.end_year if comparison.end_year is not None else min(ends, default=None)
    if start is None or end is None:
        raise ValueError("set comparison years or artists' active_from/active_to")
    return start, end


def load_config(config_dir: Path = CONFIG_DIR) -> Config:
    with (config_dir / "config.yaml").open() as f:
        return Config.model_validate(yaml.safe_load(f))


def load_artists(config_dir: Path = CONFIG_DIR) -> ArtistsConfig:
    with (config_dir / "artists.yaml").open() as f:
        return ArtistsConfig.model_validate(yaml.safe_load(f))
