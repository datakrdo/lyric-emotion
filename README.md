# lyric-emotion

Fine-grained emotion analysis of song lyrics: a ModernBERT model fine-tuned
for 28-label emotion classification (GoEmotions) and continuous
valence/arousal/dominance regression (EmoBank), evaluated against a
hand-annotated gold set of 200 lyrics, and applied to compare The Cure vs
The Smiths across their discographies.

**Status:** Phase 0 — project scaffolding. Nothing trained or analyzed yet.
See `plan_proyecto_analisis_sentimiento_letras_actualizado.md` for the full
methodology and roadmap, and `CHANGELOG.md` for progress.

## Why this project

Most "sentiment analysis on lyrics" portfolio projects run an off-the-shelf
3-class tweet classifier and call it a day. This one instead:

1. Fine-tunes its own model — 28 emotions + a continuous VAD space — on
   redistributable corpora (GoEmotions, EmoBank), so anyone can reproduce
   training without an API key.
2. Adapts that model to the lyrics domain using weakly-labeled datasets
   (MoodyLyrics, Deezer DMDD), publishing the model with a full comparison
   against baselines.
3. Validates on a gold set I annotate by hand — 200 songs, published
   annotation guidelines, intra-annotator consistency reported.
4. Applies it to a real question: how did The Cure's and The Smiths'
   emotional palette shift across their careers and albums, with
   uncertainty and sample sizes reported at every step.

## Setup

```bash
uv sync
cp .env.example .env   # fill in GENIUS_ACCESS_TOKEN and HF_TOKEN locally, never commit
uv run pytest
```

## Reproducing

Training data and the emotion/VAD heads (Phases 1–2) require no credentials.
Fetching lyrics for the case study (Phase 3) requires a free Genius API
token. See `plan_proyecto_analisis_sentimiento_letras_actualizado.md` for
the full command sequence per phase.

## What's not published here

Full lyrics text, API tokens, and raw API responses are never committed —
see `.gitignore` and `THIRD_PARTY_NOTICES.md`. Only code, configs, IDs,
aggregated scores, figures, and the annotation guidelines are public.

## License

Code: MIT (see `LICENSE`). Third-party datasets and models keep their own
licenses — see `THIRD_PARTY_NOTICES.md`.
