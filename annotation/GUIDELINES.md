# Gold set annotation guidelines

80 songs by The Cure and The Smiths (1982-1987 comparison window), rated by
hand with `streamlit run annotation/annotate.py`. They are the only ground
truth in the project; `docs/validation_protocol.md` fixes, before any rating,
how the results will be judged.

## What you rate

From the **lyrics text alone**, per song:

1. **Valence**, 1-5: 1 = very negative/unpleasant, 5 = very positive/pleasant.
2. **Arousal**, 1-5: 1 = very calm, 5 = very intense. A negative control; the
   dashboard does not report it.
3. **Up to two emotion families** (or "Neutral / unclear"): the seven in
   `config/config.yaml` (`emotion_families`), each a group of GoEmotions labels:
   Joy & fun, Love & warmth, Hope & relief, Sadness & loss, Fear & unease,
   Anger & contempt, Questioning.
4. **Low confidence** if the text is too sparse or ambiguous to judge.

## Hard rule: text only

Do not let the artist's reputation, genre, your memory of the recording
(melody, tempo, voice) or band mythology influence a rating. The page hides
artist, title and model scores for that reason. A musically upbeat song with
bleak lyrics gets bleak ratings.

## How to read

- Read the whole song once before rating.
- The narrator is not necessarily the artist: rate the emotion of the text as
  spoken by its narrator.
- Irony and metaphor: rate what the text conveys, not the literal words. If it
  is ambiguous from text alone, rate as you read it and note it.
- A repeated chorus counts once.
- Pick "Neutral / unclear" only when nothing else applies; for lyrics it
  should be rare.

## Process

- Rate in batches (~25 songs a sitting): fatigue flattens judgments to the middle.
- About a week later, tick "Re-annotation pass" in the sidebar and re-rate 15
  songs (the page picks them at random, without showing your first answers).
  `evaluate-gold` reports the intra-annotator consistency (ICC for valence and
  arousal, exact agreement for families). One reader means this is not inter-rater
  agreement, and the README says so.

## File (`gold_set.csv`)

Created by `lyric-emotion gold-sample`; contains no lyrics and no model scores.

| column | notes |
|---|---|
| `song_id`, `artist`, `title`, `year` | identity |
| `tercile` | sampling stratum (model valence tercile within the artist); not shown while rating |
| `valence`, `arousal` | integers 1-5 |
| `families` | up to two family names separated by `;` |
| `low_confidence`, `notes`, `annotation_date` | |
| `is_reannotation` | 1 for the re-rated rows |

When done: `uv run lyric-emotion evaluate-gold` writes `docs/gold_evaluation.md` and
`site/public/data/validation.json`, which the site's "Human check" row reads.
