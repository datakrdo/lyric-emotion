# Emotion -> VAD consistency check (label-free, Phase 4)

Projects each song's 28 emotion probabilities to VAD using NRC-VAD-lexicon
centroids, then correlates against the regression head's own VAD output
on the same songs. No gold-set labels involved.

| dim | Pearson r (projected vs regression head) |
|---|---|
| valence | 0.6866 |
| arousal | 0.3695 |
| dominance | 0.4744 |

n songs = 289