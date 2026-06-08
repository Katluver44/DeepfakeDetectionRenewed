# P2: CT Feature Injection — Training Summary

**n_ct=3  |  seeds=[123]  |  epochs=5**

## Per-system metrics vs robust_goat baseline

Systems with C/T + metrics: 63

### Overall (model | Δ vs baseline)
- eer (↓ better): 0.2702   Δ=+0.0087
- auc (↑ better): 0.7983   Δ=+0.0036
- bal_acc (↑ better): 0.7788   Δ=-0.0061
- acc (↑ better): 0.7535   Δ=-0.0217

### Stratified by C = −rog@L12 (Q4 = most compact = hardest)
| Quartile | n | eer (Δ) | auc (Δ) | bal_acc (Δ) | acc (Δ) |
|---|---|---|---|---|---|
| Q1 | 16 | 0.239 (+0.016) | 0.838 (-0.004) | 0.802 (-0.017) | 0.704 (-0.075) |
| Q2 | 16 | 0.260 (+0.044) | 0.823 (-0.005) | 0.800 (-0.018) | 0.856 (+0.049) |
| Q3 | 15 | 0.242 (-0.030) | 0.816 (+0.021) | 0.801 (+0.019) | 0.792 (+0.004) |
| Q4 | 16 | 0.339 (+0.002) | 0.718 (+0.004) | 0.714 (-0.007) | 0.664 (-0.063) |

_C-Q4 cross-metric: MIXED (1/4 metrics agree) — eer↓=worse/flat, auc↑=better, bal_acc↑=worse/flat, acc↑=worse/flat_

### Stratified by T = vel_entropy@L9 (Q4 = burstiest = hardest)
| Quartile | n | eer (Δ) | auc (Δ) | bal_acc (Δ) | acc (Δ) |
|---|---|---|---|---|---|
| Q1 | 16 | 0.202 (+0.002) | 0.869 (+0.013) | 0.835 (-0.003) | 0.823 (-0.050) |
| Q2 | 16 | 0.267 (+0.001) | 0.799 (+0.017) | 0.779 (-0.006) | 0.750 (+0.036) |
| Q3 | 15 | 0.249 (+0.017) | 0.834 (-0.005) | 0.796 (-0.019) | 0.767 (-0.029) |
| Q4 | 16 | 0.362 (+0.015) | 0.694 (-0.011) | 0.706 (+0.003) | 0.675 (-0.045) |

_T-Q4 cross-metric: MIXED (1/4 metrics agree) — eer↓=worse/flat, auc↑=worse/flat, bal_acc↑=better, acc↑=worse/flat_

## Config
- Base checkpoint: /lambda/nfs/algovirginia/workspace/DeepfakeDetectionRenewed/experiments/checkpoints/mlaad_robust_goat.ckpt
- n_ct: 3 (C=−rog@L12, T=vel_entropy@L9)
- cls_head input: 768+3=771
- Fine-tune epochs: 5  |  LR encoder/head: 5e-05/5e-05