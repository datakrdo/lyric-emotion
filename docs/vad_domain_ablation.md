# vad_domain: clean retrain and seed ablation

Clean domain set: The Cure, The Smiths and Morrissey rows, split leaks and
repeated training texts removed (4,764 rows; test n = 525); DMDD scores
divided by a fixed 2 instead of clipped to +-1. Test CCC on the DMDD test
split, whole-lyric scoring.

| training data | seeds | CCC valence | CCC arousal |
|---|---|---|---|
| DMDD + MoodyLyrics | 42, 1, 2 | 0.335 ± 0.026 | 0.168 ± 0.014 |
| DMDD only | 42, 1, 2 | 0.289 ± 0.029 | 0.111 ± 0.038 |

MoodyLyrics helps valence by about 0.05 CCC. Pearson r (scale-free, valence)
against the previous checkpoint, which was trained on clipped labels and on
rows from the case-study artists: 0.408 / 0.413 / 0.416 (new seeds) against
0.416 (previous). Accuracy is a tie; the new checkpoints are adopted because
they exclude the case-study songs from training.

## Seed sensitivity in the case study

Cliff's delta for valence (The Cure − The Smiths), album-level bootstrap, one
`vad_domain` seed at a time (window 1980-89):

| model | delta | 95% interval | album permutation p |
|---|---|---|---|
| previous checkpoint | -0.162 | [-0.390, 0.045] | 0.217 |
| seed 42 | -0.232 | [-0.413, -0.069] | 0.032 |
| seed 1 | -0.016 | [-0.203, 0.159] | 0.874 |
| seed 2 | -0.179 | [-0.385, -0.003] | 0.142 |
| mean of 3 seeds | -0.154 | [-0.347, 0.015] | 0.181 |

Per-song valence correlates 0.93-0.95 between seeds, yet with a few dozen
albums that is enough to move the gap. The case study therefore scores with
the mean of the three seeds (`models.case_study_vad_ensemble`).
