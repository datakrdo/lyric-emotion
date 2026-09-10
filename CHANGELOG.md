# Changelog

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
