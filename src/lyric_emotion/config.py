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
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    epochs: int
    precision: str
    use_gpu_if_available: bool


class EmotionHead(BaseModel):
    dataset: str
    num_labels: int
    loss: str


class VadHead(BaseModel):
    dataset: str
    dims: list[str]
    loss: str


class Aggregation(BaseModel):
    min_songs_per_album: int
    min_songs_per_year: int


class Statistics(BaseModel):
    confidence_level: float
    bootstrap_iterations: int
    alpha: float
    multiple_testing_correction: str


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
    aggregation: Aggregation
    statistics: Statistics
    visualization: Visualization


class Artist(BaseModel):
    name: str
    max_songs: int


class ArtistsConfig(BaseModel):
    artists: list[Artist]


def load_config(config_dir: Path = CONFIG_DIR) -> Config:
    with (config_dir / "config.yaml").open() as f:
        return Config.model_validate(yaml.safe_load(f))


def load_artists(config_dir: Path = CONFIG_DIR) -> ArtistsConfig:
    with (config_dir / "artists.yaml").open() as f:
        return ArtistsConfig.model_validate(yaml.safe_load(f))
