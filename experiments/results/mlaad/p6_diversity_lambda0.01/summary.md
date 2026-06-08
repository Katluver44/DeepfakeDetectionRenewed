# P6 representation diversity regularization (lambda=0.01) — mlaad_robust_goat — Summary

seeds aggregated: ['1024', '123', '42']

## Per-system metrics vs mlaad_robust_goat baseline

Systems with C/T + metrics: 63

### Overall (model | Δ vs baseline)
- eer (↓ better): 0.2531   Δ=-0.0084
- auc (↑ better): 0.8099   Δ=+0.0152
- bal_acc (↑ better): 0.7922   Δ=+0.0073
- acc (↑ better): 0.7894   Δ=+0.0142

### Stratified by C = −rog@L12 (Q4 = most compact = hardest)
| Quartile | n | eer (Δ) | auc (Δ) | bal_acc (Δ) | acc (Δ) |
|---|---|---|---|---|---|
| Q1 | 16 | 0.224 (+0.001) | 0.852 (+0.011) | 0.823 (+0.004) | 0.785 (+0.006) |
| Q2 | 16 | 0.221 (+0.005) | 0.842 (+0.014) | 0.826 (+0.008) | 0.853 (+0.046) |
| Q3 | 15 | 0.248 (-0.024) | 0.816 (+0.020) | 0.792 (+0.011) | 0.796 (+0.007) |
| Q4 | 16 | 0.319 (-0.017) | 0.730 (+0.016) | 0.727 (+0.007) | 0.725 (-0.002) |

_C-Q4 cross-metric: MIXED (3/4 metrics agree) — eer↓=better, auc↑=better, bal_acc↑=better, acc↑=worse/flat_

### Stratified by T = vel_entropy@L9 (Q4 = burstiest = hardest)
| Quartile | n | eer (Δ) | auc (Δ) | bal_acc (Δ) | acc (Δ) |
|---|---|---|---|---|---|
| Q1 | 16 | 0.196 (-0.003) | 0.872 (+0.016) | 0.843 (+0.005) | 0.862 (-0.011) |
| Q2 | 16 | 0.245 (-0.021) | 0.815 (+0.034) | 0.796 (+0.011) | 0.771 (+0.058) |
| Q3 | 15 | 0.231 (-0.001) | 0.830 (-0.010) | 0.811 (-0.004) | 0.782 (-0.014) |
| Q4 | 16 | 0.339 (-0.009) | 0.724 (+0.019) | 0.720 (+0.017) | 0.743 (+0.023) |

_T-Q4 cross-metric: CONSISTENT improvement — eer↓=better, auc↑=better, bal_acc↑=better, acc↑=better_