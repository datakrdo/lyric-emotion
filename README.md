🇬🇧 [English](README.md) · 🇪🇸 [Español](README_es.md)

# lyric-emotion — emotion analysis of song lyrics

**Did The Cure and The Smiths actually sound as different, emotionally, as
their reputations suggest?** Both are pillars of 1980s UK post-punk/gothic
rock; both get flattened by casual description into "miserable." This
project puts a fine-grained, reproducible emotion model behind that
question instead of relying on vibes.

Most "sentiment analysis on lyrics" portfolio projects run an off-the-shelf
3-class tweet classifier and call it a day. This one instead:

1. Fine-tunes its own model — 28 emotions (GoEmotions taxonomy) plus a
   continuous valence/arousal/dominance (VAD) space — on redistributable
   corpora (GoEmotions, EmoBank), so anyone can reproduce the base training
   without an API key.
2. Adapts that model toward the lyrics domain using weakly-labeled datasets
   (MoodyLyrics, Deezer DMDD).
3. Is meant to be validated on a gold set I annotate by hand — 80 songs,
   published guidelines and a validation protocol fixed before rating,
   intra-annotator consistency reported. **That annotation is still pending**
   (see Limitations) — all results below are pre-gold-set.
4. Applies it to a real question: how did The Cure's and The Smiths'
   emotional palette shift across their careers and albums, with
   uncertainty and sample sizes reported at every step.

## Tools 🛠️

- Python, pandas, NumPy
- PyTorch, Hugging Face Transformers (ModernBERT-base fine-tuning)
- GoEmotions, EmoBank, MoodyLyrics, Deezer DMDD, NRC-VAD lexicon
- Genius API (lyrics fetched locally, never redistributed)
- Streamlit (dashboard and hand-annotation tool)
- matplotlib
- uv, ruff, mypy, pytest, GitHub Actions

## Skills 🧠

- Fine-tuning transformers: multi-label classification and multi-output regression
- Domain adaptation with weak labels
- Evaluation against public baselines, with honest reporting when the baseline wins
- Annotation design: guidelines and validation protocol fixed before rating
- Statistical inference on clustered data (album-level permutation and resampling, FDR correction)
- Packaging an ML project as an installable package with CLI, tests, and CI
- Communicating results with an interactive dashboard

## Data 📎

| stage | source | size | redistributable? |
|---|---|---|---|
| Emotions head (stage 1) | GoEmotions | 43,410 train / 5,426 val | yes (Apache-2.0) |
| VAD head (stage 1) | EmoBank | 8,062 train / 999 dev | yes (CC BY-SA 4.0) |
| Domain adaptation (stage 2) | MoodyLyrics + DMDD sample, lyrics via Genius | 4,913 weakly-labeled songs | no — IDs/labels only, lyrics fetched independently |
| Case study corpus | Genius, deduped to studio takes | The Cure 256 songs / The Smiths 75 songs | no — Genius terms don't allow redistribution |
| Gold set | hand-annotated by me | 80 songs sampled, annotation pending | n/a |

Full license terms and how each source is used: `THIRD_PARTY_NOTICES.md`.

## Design decisions 🧭

Two independent ModernBERT-base fine-tunes rather than one multi-task
model — simpler to debug, and multi-task is left as an optional ablation:

- **Emotions head**: 28-label multi-label classification, weighted BCE
  (rare labels like `grief`/`relief` have <200 positives out of 43k rows),
  per-class thresholds tuned on validation instead of a global 0.5.
- **VAD head**: 3-output regression (valence/arousal/dominance), MSE loss.
- **Stage 2 (domain adaptation)**: continues the VAD checkpoint on weakly
  labeled lyrics, valence/arousal only — the weak labels (Last.fm-tag
  derived) carry no dominance signal, so that output is left at its
  stage-1 weights.
- **Label-free consistency check**: each song's 28 emotion probabilities
  are projected to VAD via NRC-VAD-lexicon centroids and correlated
  against the regression head's own output on the same songs — a sanity
  check that doesn't need any gold-set labels.
- **Case-study scoring**: the emotions head scores each song by sections, the
  VAD head scores the whole lyric (how it was trained). Covers, live takes and
  other artists' versions are filtered out first.
- **Inference counts albums, not songs**: songs on one album sound alike, so
  differences are tested by permuting/resampling whole albums and
  separability is cross-validated on albums the model has not seen.

## Results 📊

Pre-gold-set: every number below is provisional.

- **Emotions head vs. baseline** (GoEmotions test split, thresholds tuned
  on validation): `cirimus/modernbert-base-go-emotions` edges this model
  out (0.5531 vs. 0.5127 macro F1 — same base model and dataset). Reported
  honestly: the value here is the domain adaptation and VAD head, not
  beating an equivalent model on its own benchmark.
- **VAD head vs. baseline** (EmoBank test split): clearly beats an NRC-VAD
  lexicon baseline on all three dimensions (CCC 0.71/0.47/0.32 vs.
  0.28/0.17/0.14 for valence/arousal/dominance).
- **Stage 1 vs. stage 2, same footing**: scored on the same DMDD test
  split with section-based aggregation, the domain-adapted checkpoint
  (`vad_domain`) beats the stage-1 checkpoint on both dimensions it was
  adapted for (valence CCC 0.234 vs. 0.068, arousal CCC 0.040 vs. 0.017),
  so the case study now runs on `vad_domain` — decided by
  `docs/vad_checkpoint_comparison.md`, not by hand. Dominance still comes
  from the un-adapted stage-1 weights (no dominance signal in the weak
  labels).
- **Consistency check**: emotion-probabilities-projected-to-VAD correlates
  with the regression head at Pearson r 0.67 (valence), 0.42 (arousal),
  0.24 (dominance) — the two heads mostly agree with each other, most on
  valence, least on dominance.
- **The Cure vs. The Smiths, 1982-1987** (the years both were active;
  `findings.json`; 67 and 66 songs, 8 studio albums plus per-year groups of
  singles and B-sides, 18 groups in all): no clear difference in darkness
  (Cliff's delta -0.06, album-resampled 95% interval -0.30 to 0.13), but the
  28-emotion profile tells the bands apart on unseen albums (cross-validated
  AUC 0.84, album-level permutation p = 0.005; 0.83 without the
  question-related emotions). Emotion families are called "clear" only when
  the FDR-adjusted album-level p is below 0.05 and the 95% interval stays
  beyond the smallest effect of interest (0.147): "Joy & fun" (higher in The
  Cure; delta 0.46, interval 0.25 to 0.67) qualifies. "Questioning" (higher in
  The Smiths, tracking lines that end in "?"; delta -0.35, adjusted p = 0.035)
  is suggestive: its interval, -0.57 to -0.14, just reaches the threshold.
  Valence is the mean of three `vad_domain` training seeds because a single
  seed moves the gap between -0.02 and -0.23 over 1980-89
  (`docs/vad_domain_ablation.md`). Arousal and dominance did not hold up
  (arousal r = 0.135 against DMDD) and are not reported.

## Dashboard 📈

`app/` is a self-contained Streamlit app that renders a small bundle
(`app/data/`, written by `lyric-emotion export-app`; scores only, no lyrics).
Its centerpiece is a two-axis map: one dot per song, brightness on one axis and
"overall feel" (the best out-of-sample combination of emotion scores for telling
the bands apart) on the other. Static figures live in `output/figures/`.

## Quick start 🚀

```bash
uv sync
cp .env.example .env   # fill in GENIUS_ACCESS_TOKEN and HF_TOKEN locally, never commit
uv run pytest
```

Training data and the emotion/VAD heads (Phases 1-2) require no
credentials. Fetching lyrics for the case study or the domain-adaptation
set (Phases 3-4) requires a free Genius API token.

```bash
uv run lyric-emotion build-training              # Phase 1: GoEmotions + EmoBank
uv run lyric-emotion train --head emotions       # Phase 2
uv run lyric-emotion train --head vad            # Phase 2
uv run lyric-emotion evaluate-baselines          # Phase 2/4 baselines + consistency check
uv run lyric-emotion fetch                       # Phase 3: The Cure / The Smiths corpus
uv run lyric-emotion build-domain-training       # Phase 4: MoodyLyrics + DMDD sample
uv run lyric-emotion train --head vad-domain     # Phase 4 stage 2
uv run lyric-emotion analyze && uv run lyric-emotion report   # Phase 5
uv run lyric-emotion gold-sample                              # Phase 4: draw the 80 songs
uv run streamlit run annotation/annotate.py                   # rate them (local only)
uv run lyric-emotion evaluate-gold                            # human check -> dashboard
uv run lyric-emotion export-app                               # Phase 6 bundle
uv run streamlit run app/streamlit_app.py                     # dashboard
```

## Limitations ⚠️

- **Gold-set annotation is still pending** — every comparison above is
  either against a public baseline or against the model's own weak
  training labels, not against hand-verified ground truth. Treat all
  numbers as provisional.
- Weak domain labels (MoodyLyrics/DMDD) are derived from Last.fm tags, not
  direct annotation — they cap what stage 2 can prove.
- Lyrics-only signal: valence and mood conveyed by melody, tempo, and
  instrumentation are not modeled.
- English only; metaphor, irony, and fictional narrators are not
  specifically handled — a literal reading of dark lyrics may overstate
  negative affect.
- Studio-only dedup can't fully resolve every live/reissue edge case; see
  `docs/dedup_report.md`.
- No causal claims anywhere: temporal and cross-artist differences are
  reported as associations with confidence intervals and sample sizes, not
  as causes.

What's not published here: full lyrics text, API tokens, and raw API
responses are never committed — see `.gitignore` and `THIRD_PARTY_NOTICES.md`.
Only code, configs, IDs, aggregated scores, figures, and the annotation
guidelines are public.

## License

Code: MIT (see `LICENSE`). Third-party datasets and models keep their own
licenses — see `THIRD_PARTY_NOTICES.md`.
