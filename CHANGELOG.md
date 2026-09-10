# Changelog

## Unreleased — Phase 1: redistributable training data

- `src/lyric_emotion/data.py`: `build_emotions_dataset()` (GoEmotions via
  `datasets`, official train/validation/test split, 28 binary label columns)
  and `build_vad_dataset()` (EmoBank CSV from GitHub, official split,
  valence/arousal/dominance rescaled from 1-5 to [-1, 1]).
- `lyric-emotion build-training` writes `data/processed/emotions_train.parquet`,
  `data/processed/vad_train.parquet`, and `docs/data.md` (split sizes +
  per-emotion positive counts + VAD means).
- Confirmed on real data: GoEmotions train split is 43,410 rows; `grief` (77)
  and `relief` (153) are as scarce as the plan warned — per-class threshold
  tuning in Phase 2 is not optional. EmoBank: 8,062/1,000/1,000
  train/dev/test rows, valence/arousal/dominance means near 0 as expected
  after centering to [-1, 1].
- Tests (`tests/test_data.py`, marked `network`): label vector is binary,
  splits don't overlap, VAD stays in range — run against the real
  downloads, not mocks.

## Unreleased — Phase 0: project scaffolding

- Initialized `uv` project (Python 3.12, `<3.14` — torch/transformers not yet
  reliable on 3.14).
- Added dependencies: torch 2.14 (cu130, GPU-verified on RTX 4070 Laptop),
  transformers 5.17, datasets 5.0, accelerate, lyricsgenius 3.12, pandas,
  pyarrow, scikit-learn, scipy, statsmodels, plotly, matplotlib, typer,
  pydantic, pyyaml, python-dotenv, rapidfuzz, tenacity, tqdm, streamlit.
- Dev tooling: pytest, ruff, mypy, pre-commit — all green.
- Repo structure: `config/`, `data/{raw,interim,processed,cache}`,
  `src/lyric_emotion/`, `tests/`, `notebooks/`, `annotation/`, `app/`,
  `output/{figures,tables}`, `docs/`.
- `src/lyric_emotion/config.py`: pydantic-validated loader for
  `config/config.yaml` and `config/artists.yaml` — no analytical parameter
  hardcoded elsewhere.
- `src/lyric_emotion/cli.py`: command skeleton (`fetch`, `build-training`,
  `train`, `predict`, `analyze`, `report`), each raising `NotImplementedError`
  until its phase lands.
- `THIRD_PARTY_NOTICES.md`: license audit for GoEmotions (Apache-2.0),
  EmoBank (CC BY-SA 4.0 — ShareAlike noted), ModernBERT-base (Apache-2.0),
  NRC-VAD v2 (non-commercial only, not redistributed). MoodyLyrics/DMDD
  license verification deferred to Phase 3.
