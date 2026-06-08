# P2 CT feature injection (n_ct=2) — mlaad_robust_goat — Summary

seeds aggregated: ['1024', '123', '42']

## Per-system metrics vs mlaad_robust_goat baseline

Systems with C/T + metrics: 63

### Overall (model | Δ vs baseline)
- eer (↓ better): 0.2518   Δ=-0.0097
- auc (↑ better): 0.8056   Δ=+0.0109
- bal_acc (↑ better): 0.7931   Δ=+0.0082
- acc (↑ better): 0.7839   Δ=+0.0087

### Stratified by C = −rog@L12 (Q4 = most compact = hardest)
| Quartile | n | eer (Δ) | auc (Δ) | bal_acc (Δ) | acc (Δ) |
|---|---|---|---|---|---|
| Q1 | 16 | 0.218 (-0.004) | 0.850 (+0.008) | 0.826 (+0.007) | 0.747 (-0.032) |
| Q2 | 16 | 0.217 (+0.002) | 0.836 (+0.008) | 0.823 (+0.005) | 0.839 (+0.032) |
| Q3 | 15 | 0.245 (-0.027) | 0.814 (+0.018) | 0.794 (+0.012) | 0.800 (+0.012) |
| Q4 | 16 | 0.327 (-0.010) | 0.723 (+0.009) | 0.730 (+0.009) | 0.750 (+0.023) |

_C-Q4 cross-metric: CONSISTENT improvement — eer↓=better, auc↑=better, bal_acc↑=better, acc↑=better_

### Stratified by T = vel_entropy@L9 (Q4 = burstiest = hardest)
| Quartile | n | eer (Δ) | auc (Δ) | bal_acc (Δ) | acc (Δ) |
|---|---|---|---|---|---|
| Q1 | 16 | 0.190 (-0.009) | 0.873 (+0.017) | 0.846 (+0.008) | 0.840 (-0.033) |
| Q2 | 16 | 0.256 (-0.010) | 0.808 (+0.027) | 0.794 (+0.009) | 0.787 (+0.074) |
| Q3 | 15 | 0.226 (-0.006) | 0.830 (-0.009) | 0.810 (-0.005) | 0.764 (-0.033) |
| Q4 | 16 | 0.333 (-0.014) | 0.713 (+0.008) | 0.723 (+0.020) | 0.744 (+0.024) |

_T-Q4 cross-metric: CONSISTENT improvement — eer↓=better, auc↑=better, bal_acc↑=better, acc↑=better_