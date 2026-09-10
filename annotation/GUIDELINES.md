# Gold set annotation guidelines

This is the guide for building `gold_set.csv`: ~200 songs (The Cure and The
Smiths) annotated by hand, used in Phase 4 as the final arbiter for every
model and baseline in this project — nothing else in the pipeline is
ground truth.

## What you're labeling

Each song gets two independent sets of judgments, both from the **lyrics
text alone**:

1. **28 GoEmotions labels** (multi-label — mark every one that applies, not
   just the strongest): admiration, amusement, anger, annoyance, approval,
   caring, confusion, curiosity, desire, disappointment, disapproval,
   disgust, embarrassment, excitement, fear, gratitude, grief, joy, love,
   nervousness, optimism, pride, realization, relief, remorse, sadness,
   surprise, neutral.
2. **Valence, arousal, dominance** — each on a 1-5 integer scale (same
   convention as EmoBank, rescaled to [-1, 1] downstream by the pipeline,
   not by you):
   - **Valence**: 1 = very negative/unpleasant feeling, 5 = very
     positive/pleasant.
   - **Arousal**: 1 = very calm/low energy, 5 = very excited/high energy.
   - **Dominance**: 1 = feeling controlled/helpless, 5 = feeling in
     control/powerful.

## Hard rule: text only

Judge the lyrics as written. Do not let these influence your label:
- The artist's reputation, genre, or your prior knowledge of the song.
- The actual recording — melody, tempo, vocal delivery, production.
- Band mythology or press narrative ("The Cure are a goth band, so...").

A musically upbeat song with bleak lyrics gets bleak labels, and vice versa
— that gap is itself something the analysis (Fase 5) may end up discussing,
so don't paper over it by labeling "what the song feels like" from memory.

## How to read the lyrics

- Read the whole song once before labeling anything.
- The **narrator is not necessarily the artist**. Judge the emotional
  content of the text as spoken by its narrator/persona, not what you
  imagine the songwriter felt.
- Metaphor and irony: label the emotion the text conveys, not the literal
  surface meaning of individual words. ("I'm on top of the world" sung
  bitterly is still valence-negative if the surrounding text makes the
  irony clear from words alone — if it's ambiguous from text alone, label
  it as you read it and note the ambiguity in the `notes` column.)
- Repeated choruses count once for label purposes — don't let repetition
  inflate arousal/intensity judgments.
- If the lyrics are too sparse or incoherent to judge confidently, still
  give your best-effort labels and set `low_confidence=1` in the CSV rather
  than skipping the song.

## Multi-label emotions: how many to pick

There's no fixed number. Typical range is 1-4 emotions per song. Pick
`neutral` only when nothing else applies — it should be rare for song
lyrics, not a default.

## Process

1. Songs are sampled stratified by artist, decade, and the Phase 2 model's
   prediction (so the gold set isn't skewed toward what the model already
   gets right) — sampling is done by the pipeline, not by you.
2. Annotate in batches of ~25/day. Don't do all 200 in one sitting —
   annotator fatigue flattens judgments toward the middle of the scale.
3. One annotator (you) for v1 — there's no second rater. To still report
   reliability: **re-annotate a random 30 of the 200** about a week after
   your first pass, blind to your original labels, and report
   intra-annotator Cohen's kappa (categorical labels) and ICC (VAD scores).
   This is a documented limitation, not a substitute for real inter-annotator
   agreement — say so plainly in the README.

## CSV schema (`gold_set.csv`)

No lyrics text in this file — only IDs and labels, per the project's
copyright policy.

| column | type | notes |
|---|---|---|
| `song_id` | str | stable ID matching the fetched corpus |
| `artist` | str | "The Cure" or "The Smiths" |
| `title` | str | |
| `year` | int | release year |
| `<emotion>` | 0/1 | one column per GoEmotions label, 28 total |
| `valence` | int 1-5 | |
| `arousal` | int 1-5 | |
| `dominance` | int 1-5 | |
| `low_confidence` | 0/1 | set to 1 if the text was too sparse/ambiguous |
| `notes` | str | optional — irony, ambiguous narrator, etc. |
| `annotation_date` | date | |
| `is_reannotation` | 0/1 | 1 for the week-later repeats used for kappa |
