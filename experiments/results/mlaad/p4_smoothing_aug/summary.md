# P4 smoothing augmentation — mlaad_robust_goat — Summary

seeds aggregated: ['1024', '123', '42']

## Per-system metrics vs mlaad_robust_goat baseline

Systems with C/T + metrics: 63

### Overall (model | Δ vs baseline)
- eer (↓ better): 0.2988   Δ=+0.0372
- auc (↑ better): 0.7589   Δ=-0.0358
- bal_acc (↑ better): 0.7528   Δ=-0.0321
- acc (↑ better): 0.7404   Δ=-0.0348

### Stratified by C = −rog@L12 (Q4 = most compact = hardest)
| Quartile | n | eer (Δ) | auc (Δ) | bal_acc (Δ) | acc (Δ) |
|---|---|---|---|---|---|
| Q1 | 16 | 0.261 (+0.039) | 0.808 (-0.034) | 0.787 (-0.032) | 0.743 (-0.036) |
| Q2 | 16 | 0.263 (+0.048) | 0.794 (-0.034) | 0.778 (-0.039) | 0.798 (-0.009) |
| Q3 | 15 | 0.308 (+0.036) | 0.758 (-0.038) | 0.750 (-0.032) | 0.762 (-0.027) |
| Q4 | 16 | 0.363 (+0.026) | 0.676 (-0.038) | 0.696 (-0.025) | 0.660 (-0.067) |

_C-Q4 cross-metric: CONSISTENT no-improvement/regression — eer↓=worse/flat, auc↑=worse/flat, bal_acc↑=worse/flat, acc↑=worse/flat_

### Stratified by T = vel_entropy@L9 (Q4 = burstiest = hardest)
| Quartile | n | eer (Δ) | auc (Δ) | bal_acc (Δ) | acc (Δ) |
|---|---|---|---|---|---|
| Q1 | 16 | 0.243 (+0.044) | 0.829 (-0.028) | 0.801 (-0.038) | 0.779 (-0.094) |
| Q2 | 16 | 0.296 (+0.031) | 0.757 (-0.024) | 0.755 (-0.030) | 0.742 (+0.029) |
| Q3 | 15 | 0.277 (+0.044) | 0.789 (-0.050) | 0.777 (-0.038) | 0.739 (-0.057) |
| Q4 | 16 | 0.378 (+0.031) | 0.663 (-0.042) | 0.680 (-0.023) | 0.701 (-0.019) |

_T-Q4 cross-metric: CONSISTENT no-improvement/regression — eer↓=worse/flat, auc↑=worse/flat, bal_acc↑=worse/flat, acc↑=worse/flat_