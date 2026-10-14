# VAD checkpoint comparison (stage 1 vs stage 2, DMDD test split)

Decides which checkpoint and scoring unit `config.models.case_study_vad` /
`case_study_vad_unit` should use, by scoring both checkpoints on real song
lyrics with valence/arousal labels, whole-lyric (`full`) and by sections.
Test texts that also appear in train/validation are excluded.

| checkpoint | unit | dim | CCC | Pearson r |
|---|---|---|---|---|
| vad | full | valence | 0.0734 | 0.1982 |
| vad | full | arousal | 0.0453 | 0.1626 |
| vad | sections | valence | 0.1003 | 0.2776 |
| vad | sections | arousal | 0.0244 | 0.1030 |
| vad_domain | full | valence | 0.3506 | 0.4080 |
| vad_domain | full | arousal | 0.1530 | 0.2024 |
| vad_domain | sections | valence | 0.2499 | 0.3950 |
| vad_domain | sections | arousal | 0.0331 | 0.0980 |
| vad_domain_mean_of_3_seeds | full | valence | 0.3441 | 0.4194 |
| vad_domain_mean_of_3_seeds | full | arousal | 0.1698 | 0.2188 |

n = 525