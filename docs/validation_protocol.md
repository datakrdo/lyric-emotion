# Validation protocol (fixed before annotating)

One human reader rates a light gold set; nothing below may be changed after
looking at the results.

**Sample.** 80 songs from the 1982-87 comparison window, split by artist in
proportion to size and evenly across terciles of the model's valence within
each artist. The model's scores are hidden while rating, and so are the
artist and title.

**Ratings per song.** Valence 1-5, arousal 1-5 (a negative control: the
dashboard does not report it), up to two emotion families (the seven in
`config.yaml: emotion_families`, or "Neutral / unclear"), and a low-confidence
flag. 15 songs are re-rated at least a week later.

**Decision rules.**
- Brightness (valence): Spearman rho between the model and the reader with a
  95% bootstrap interval. *Validated* if the lower bound is >= 0.30;
  *partial* if only the point estimate is >= 0.30; otherwise *not validated*
  and the dashboard says so.
- Emotion families: AUC of the model's family score against the reader's tag,
  with a bootstrap interval. A family is shown as validated only if the lower
  bound is above 0.5. "Joy & fun" and "Questioning" are the two the
  dashboard's headline relies on.
- Emotions head: the repo's head and `cirimus/modernbert-base-go-emotions` are
  compared on the same family AUCs; the other one is adopted only if the
  interval of the paired difference excludes 0 in its favor.
- Calibration: isotonic map from model valence to the 1-5 scale, scored out
  of fold; used to label the dashboard's brightness axis only if it beats the
  constant-median baseline.

**Known limits.** n=80 gives a rho of 0.5 an interval of roughly 0.31-0.65:
a coarse validation. The reader is a single person, so consistency is reported
(ICC on the re-rated songs), not inter-rater agreement. The seven families are
broad; the fine 28-label structure behind the "overall feel" result is not
validated by this set.
