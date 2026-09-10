# Third-Party Notices

This file tracks every external dataset, model, and code pattern this project
depends on: its license, the exact version/commit reviewed, and how it is
used. Checked before any code in Phase 1+ touches these resources.

## Datasets

### GoEmotions
- **Source:** `google-research-datasets/go_emotions` on Hugging Face Hub
- **License:** Apache License 2.0
- **Use:** Stage 1 training data for the 28-label multi-label emotion head
  (`lyric-emotion train --head emotions`). Redistributable — used via the
  `datasets` library, no local copy committed to this repo.
- **Reviewed:** 2026-09-10

### EmoBank
- **Source:** `JULIELab/EmoBank` on GitHub
- **License:** CC BY-SA 4.0 (per repository README)
- **Use:** Stage 1 training data for the VAD regression head. **Note the
  ShareAlike clause**: derivative datasets built from EmoBank must carry the
  same license. Only the trained model weights are derivative works of code,
  not of the dataset itself, but if any EmoBank-derived data artifact is
  published from this repo, it must be marked CC BY-SA 4.0 and attributed to
  JULIE Lab, Jena University.
- **Reviewed:** 2026-09-10

### MoodyLyrics
- **Source:** Çano & Morisio, "MoodyLyrics: A Sentiment Annotated Lyrics
  Dataset" (2017)
- **License:** distributes song IDs + mood-quadrant labels only, no lyrics
  text. Verify exact terms before Phase 3/4 use.
- **Use:** Stage 2 weak labels; lyrics fetched independently via Genius, not
  redistributed.
- **Reviewed:** pending (Phase 3)

### Deezer Mood Detection Dataset (DMDD)
- **Source:** `deezer/deezer_mood_detection_dataset` on GitHub
- **License:** verify before Phase 3/4 use (distributes Deezer/MSD IDs +
  valence/arousal values, no audio or lyrics).
- **Use:** Stage 2 weak labels.
- **Reviewed:** pending (Phase 3)

### NRC-VAD Lexicon v2
- **Source:** saifmohammad.com (NRC, National Research Council Canada)
- **License:** non-commercial research/educational use only.
- **Use:** lexicon baseline for VAD regression + emotion→VAD consistency
  check in Phase 4. **Not redistributed** in this repo; downloaded locally
  via a script that points to the official source, gitignored.
- **Reviewed:** 2026-09-10

## Models

### ModernBERT-base
- **Source:** `answerdotai/ModernBERT-base` on Hugging Face Hub
- **License:** Apache License 2.0
- **Use:** base model fine-tuned for both heads.

## Comparison baselines (not dependencies — cited for evaluation)

### cirimus/modernbert-base-go-emotions
- Used only as a benchmark in the evaluation table (Phase 2/4), never
  imported as a dependency of this project.

## Sources not used

`sgroenjes/LyricLens`, the "NLP Song Lyrics" tutorial project, and
`mcat18/Lyric-Analysis-Project` were reviewed during planning and judged to
be tutorial-level code offering nothing beyond what `transformers` +
`datasets` already provide directly. Not reused.
