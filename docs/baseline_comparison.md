# Baseline comparison

## Emotions head (thresholds tuned on validation, scored on the GoEmotions test split)

| model | macro F1 | micro F1 | mAP | n test |
|---|---|---|---|---|
| lyric-emotion (ours) | 0.5127 | 0.5930 | 0.5069 | 5427 |
| cirimus/modernbert-base-go-emotions | 0.5531 | 0.6152 | 0.5537 | 5427 |

## VAD head (CCC / Pearson r, EmoBank test split)

| model | dim | CCC | Pearson r |
|---|---|---|---|
| lyric-emotion (ours) | valence | 0.7083 | 0.7256 |
| lyric-emotion (ours) | arousal | 0.4707 | 0.5050 |
| lyric-emotion (ours) | dominance | 0.3227 | 0.4248 |
| NRC-VAD lexicon (mean word score) | valence | 0.2824 | 0.3756 |
| NRC-VAD lexicon (mean word score) | arousal | 0.1675 | 0.1939 |
| NRC-VAD lexicon (mean word score) | dominance | 0.1370 | 0.1487 |

n = 1000

Both models trained on the same train split and validated on the same held-out split their own thresholds/checkpoints were tuned against — not the same split used to pick the threshold or the early-stopping checkpoint.