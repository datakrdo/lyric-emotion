# Training report — vad_domain head (seed 42)

Clean retrain (DMDD + MoodyLyrics). The case study scores with the mean of
seeds 42, 1 and 2 (`models.case_study_vad_ensemble`); per-seed reports are in
`training_vad_domain_both_s*.md` and the ablation in `vad_domain_ablation.md`.

```json
{
  "ccc": {
    "valence": 0.35055723786354065,
    "arousal": 0.15317507088184357
  },
  "pearson_r": {
    "valence": 0.40793905073189224,
    "arousal": 0.2027442480819897
  },
  "n_train": 3643,
  "n_val": 596,
  "n_test": 525,
  "note": "dominance not adapted \u2014 weak labels carry no dominance signal"
}
```