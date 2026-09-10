# Changelog

## Unreleased — Phase 3 (in progress): Genius acquisition client

- `src/lyric_emotion/data.py`: `fetch_artist_songs()` fetches one artist's
  full catalog via `lyricsgenius.Genius.search_artist()` (built-in
  pagination, retry, and rate-limit handling via `timeout`/`sleep_time`/
  `retries`, configured from `config.yaml`'s `acquisition` section — no
  hand-rolled backoff needed). Caches the raw response per artist under
  `data/raw/genius/<slug>.json`; re-running `fetch` skips any artist whose
  cache file already exists, so a crash mid-run doesn't force a full
  re-download.
- `write_fetch_coverage_report()`: per-artist table (requested / fetched /
  with lyrics / with album / with year / excluded / duplicate titles),
  written to `docs/fetch_coverage.md` — reviewed before any later phase
  touches this data, per the project's acquisition-review rule.
- `lyric-emotion fetch` wired to both.
- Verified live against the real Genius API (The Smiths, 3 songs): cache
  hit correctly skips the network entirely on a second run.
- `THIRD_PARTY_NOTICES.md`: closed the two pending Phase 3 license entries
  — MoodyLyrics and the Deezer Mood Detection Dataset (DMDD) both have no
  formal open license, just "free for research, cite the paper" terms.
  Same policy as NRC-VAD: never committed to the repo, downloaded into the
  gitignored cache dir.
- `annotation/GUIDELINES.md`: the gold-set annotation guide — label from
  text only, 28 GoEmotions labels (multi-label) + VAD on a 1-5 scale,
  ~25 songs/day, re-annotate 30 a week later for intra-annotator
  Cohen's kappa / ICC (no second annotator in v1 — documented as a
  limitation, not hidden).
- `tests/test_fetch.py`: pure-logic tests for title normalization,
  slugification, cache reuse, and coverage counting — no network needed.
- Not yet done: the actual full fetch run for The Cure/The Smiths, the
  cleaning/dedup step (rapidfuzz, live/remix/demo/reissue flags), and the
  gold-set sampling + annotation itself.

## Unreleased — Phase 2 (in progress): baselines + full-run fixes

- `src/lyric_emotion/evaluate.py`: `evaluate_emotions_baseline()` scores
  `cirimus/modernbert-base-go-emotions` on our GoEmotions validation split
  (confirmed its label order matches `GOEMOTIONS_LABELS` exactly, so no
  remapping needed) with the same threshold-tuned macro-F1 as our model.
  `evaluate_vad_baseline()` downloads the NRC-VAD lexicon v2.1 (55k terms,
  non-commercial research use, never committed — cached under
  `data/cache/`) and scores a mean-of-known-words baseline against the
  EmoBank dev split with the same CCC/Pearson metrics. Both run on CPU —
  one-off eval, not worth competing with the training run for GPU memory.
  `lyric-emotion evaluate-baselines` wires both into `docs/baseline_comparison.md`.
- Two real bugs found running the **full** (non-smoke) training data,
  invisible during the 500-row smoke test:
  - `batch_size=16` at `max_length=1024` OOM'd on the RTX 4070's 8GB. Fixed
    by dropping to `batch_size=4` / `gradient_accumulation_steps=8` (same
    effective batch size, 32) and enabling `gradient_checkpointing=True`.
  - One EmoBank row had `text=NaN`, present in the `dev` split but absent
    from the 500-row train-split subsample used by the smoke test — crashed
    the tokenizer. Fixed with `dropna(subset=["text"])` in
    `build_vad_dataset()`; `vad_train.parquet` and `docs/data.md`
    regenerated (10,061 rows instead of 10,062).
- `tests/test_evaluate.py`: pure-logic test for the lexicon word-averaging.

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
