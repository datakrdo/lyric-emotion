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
  Dataset" (2017), hosted at softeng.polito.it/erion/research.html.
  MoodyLyrics4Q (2000 songs, 4 Russell quadrants) and MoodyLyricsPN
  (5000 songs, pos/neg) both derived from Last.fm tags.
- **License:** no formal open-source license. Terms are "free to use, cite
  the paper" (academic courtesy license, not a grant to redistribute).
  Distributes song IDs/artist/title + mood labels only, no lyrics text.
- **Use:** Stage 2 weak labels. Lyrics fetched independently via Genius, not
  redistributed. The MoodyLyrics CSV itself is **not committed** to this
  repo — downloaded on demand into the gitignored cache dir, same policy as
  NRC-VAD. Cite Çano & Morisio (ACM ISMSI 2017 / AIAP 2017) in the README.
- **Reviewed:** 2026-09-10

### Deezer Mood Detection Dataset (DMDD)
- **Source:** `deezer/deezer_mood_detection_dataset` on GitHub
- **License:** none declared (`license: null` via GitHub API — no LICENSE
  file). Distributes Deezer track IDs + Million Song Dataset IDs +
  valence/arousal values only, no audio or lyrics ("for copyright reasons"
  per the repo's own README).
- **Use:** Stage 2 weak labels. Treated the same as MoodyLyrics: **not
  committed** to this repo, fetched into the gitignored cache dir, cited to
  Delbouys, Hennequin & Piccoli (the accompanying paper) in the README.
- **Reviewed:** 2026-09-10

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
