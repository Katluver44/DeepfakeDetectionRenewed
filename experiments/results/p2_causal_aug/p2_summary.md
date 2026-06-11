# P2 — Causal minimal augmentation (reverb+MP3) vs broad augmentation

## Verdict: **PARTIAL / NOT supported**

Finetune mlaad_robust_goat, 5 epochs, seeds [42, 123], aug_prob=0.35. Conditions differ ONLY in the augmentation set.

| condition | MLAAD-EER | ITW-EER | ITW-FPR | ITW-FNR | ITW-bal |
|---|---|---|---|---|---|
| robust_goat (ref) | 0.245 | 0.376 | 0.654 | 0.140 | 0.603 |
| none | 0.247±0.028 | 0.431±0.027 | 0.670±0.072 | 0.169±0.094 | 0.581±0.011 |
| broad | 0.287±0.022 | 0.397±0.064 | 0.598±0.073 | 0.213±0.152 | 0.594±0.039 |
| causal | 0.273±0.058 | 0.458±0.019 | 0.715±0.020 | 0.170±0.026 | 0.557±0.003 |

**causal vs broad**: ITW-FPR 0.715 vs 0.598 (Δ=+0.117); ITW-EER 0.458 vs 0.397; MLAAD-EER 0.273 vs 0.287.

## Reading
- If causal (reverb+MP3) lowers ITW false positives more than broad (noise+pitch+reverb) at equal budget, augmentation selection guided by the CAUSAL channel diagnosis beats generic augmentation — a mechanism-driven training recommendation.

## Files: p2_runs.csv, p2_aggregate.csv