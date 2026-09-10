# Changelog

## Unreleased — Phase 2 (in progress): fine-tuning code + smoke test

- `src/lyric_emotion/model.py`: `train_emotions()` (ModernBERT-base,
  `problem_type="multi_label_classification"`, 28 labels) and `train_vad()`
  (`problem_type="regression"`, 3 outputs), both via `transformers.Trainer`.
  Verified against Context7 docs for transformers 5.2: `processing_class=`
  replaces the deprecated `tokenizer=` Trainer arg, `problem_type` behavior
  confirmed unchanged from 4.x.
- Per-class threshold tuning for the emotions head (`_tune_thresholds`,
  via `precision_recall_curve`) — a flat 0.5 threshold is not safe given
  GoEmotions' imbalance (`grief`/`relief` confirmed in Phase 1).
- CCC (`_ccc`) + Pearson r per VAD dimension for the regression head.
- `lyric-emotion train --head emotions|vad [--smoke] [--epochs N]`.
  `--smoke` subsamples to 500 rows/split and 256 tokens for a fast
  end-to-end sanity run; real runs use `config.yaml`'s full settings
  (max_length 1024, configured epochs).
- **Smoke test passed on the RTX 4070 Laptop GPU** (bf16): both heads
  initialize their classifier layers correctly (`classifier.weight/bias`
  reported MISSING from the base checkpoint, as expected — newly
  initialized for the downstream task), train one epoch, evaluate, and
  save. Emotions: eval_loss 0.364. VAD: eval_loss 0.070, CCC/Pearson ~0.02-0.05
  (near-zero as expected — 500 rows, 1 epoch, not a real training run).
- Bug found and fixed during the smoke test: `df.groupby("split").apply()`
  drops the grouping column by default in pandas 3.0 (changed
  `include_groups` default), which broke `--smoke` subsampling with
  `KeyError: 'split'`. Fixed by switching to `.groupby("split").head(n)`.
- `tests/test_model.py`: unit tests for `_ccc` and `_tune_thresholds`
  (pure logic, no GPU/network) — CCC=1 for identical arrays and penalizes
  offset unlike Pearson r, threshold tuning finds perfect separation when
  it exists.
- pytest (9/9), ruff, mypy all green.
- Not yet done: full training runs (all epochs, full data), baseline
  comparison against `cirimus/modernbert-base-go-emotions` and the NRC-VAD
  lexicon baseline, and the training report writer producing real metrics
  instead of smoke-test noise.

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
