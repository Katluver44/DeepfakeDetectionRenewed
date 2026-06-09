# P5 system hardness reweight — mlaad_robust_goat — Summary

seeds aggregated: ['1024', '123', '42']

## Per-system metrics vs mlaad_robust_goat baseline

Systems with C/T + metrics: 63

### Overall (model | Δ vs baseline)
- eer (↓ better): 0.2549   Δ=-0.0067
- auc (↑ better): 0.8088   Δ=+0.0141
- bal_acc (↑ better): 0.7891   Δ=+0.0042
- acc (↑ better): 0.7781   Δ=+0.0029

### Stratified by C = −rog@L12 (Q4 = most compact = hardest)
| Quartile | n | eer (Δ) | auc (Δ) | bal_acc (Δ) | acc (Δ) |
|---|---|---|---|---|---|
| Q1 | 16 | 0.234 (+0.012) | 0.841 (-0.000) | 0.811 (-0.008) | 0.778 (-0.000) |
| Q2 | 16 | 0.222 (+0.007) | 0.838 (+0.010) | 0.820 (+0.003) | 0.825 (+0.017) |
| Q3 | 15 | 0.249 (-0.023) | 0.817 (+0.021) | 0.793 (+0.011) | 0.796 (+0.007) |
| Q4 | 16 | 0.313 (-0.023) | 0.739 (+0.026) | 0.733 (+0.012) | 0.715 (-0.012) |

_C-Q4 cross-metric: MIXED (3/4 metrics agree) — eer↓=better, auc↑=better, bal_acc↑=better, acc↑=worse/flat_

### Stratified by T = vel_entropy@L9 (Q4 = burstiest = hardest)
| Quartile | n | eer (Δ) | auc (Δ) | bal_acc (Δ) | acc (Δ) |
|---|---|---|---|---|---|
| Q1 | 16 | 0.204 (+0.005) | 0.869 (+0.012) | 0.837 (-0.001) | 0.867 (-0.006) |
| Q2 | 16 | 0.248 (-0.018) | 0.813 (+0.032) | 0.795 (+0.010) | 0.750 (+0.036) |
| Q3 | 15 | 0.239 (+0.007) | 0.822 (-0.017) | 0.805 (-0.010) | 0.757 (-0.039) |
| Q4 | 16 | 0.328 (-0.020) | 0.732 (+0.027) | 0.721 (+0.018) | 0.737 (+0.018) |

_T-Q4 cross-metric: CONSISTENT improvement — eer↓=better, auc↑=better, bal_acc↑=better, acc↑=better_