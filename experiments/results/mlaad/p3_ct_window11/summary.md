# P3 CT injection + T_window11 (n_ct=3) — mlaad_robust_goat — Summary

seeds aggregated: ['1024', '123', '42']

## Per-system metrics vs mlaad_robust_goat baseline

Systems with C/T + metrics: 63

### Overall (model | Δ vs baseline)
- eer (↓ better): 0.2778   Δ=+0.0162
- auc (↑ better): 0.7790   Δ=-0.0157
- bal_acc (↑ better): 0.7711   Δ=-0.0138
- acc (↑ better): 0.7652   Δ=-0.0100

### Stratified by C = −rog@L12 (Q4 = most compact = hardest)
| Quartile | n | eer (Δ) | auc (Δ) | bal_acc (Δ) | acc (Δ) |
|---|---|---|---|---|---|
| Q1 | 16 | 0.240 (+0.017) | 0.827 (-0.015) | 0.804 (-0.015) | 0.727 (-0.052) |
| Q2 | 16 | 0.254 (+0.038) | 0.806 (-0.022) | 0.798 (-0.020) | 0.840 (+0.033) |
| Q3 | 15 | 0.270 (-0.002) | 0.787 (-0.008) | 0.778 (-0.004) | 0.779 (-0.010) |
| Q4 | 16 | 0.347 (+0.010) | 0.696 (-0.018) | 0.705 (-0.016) | 0.716 (-0.011) |

_C-Q4 cross-metric: CONSISTENT no-improvement/regression — eer↓=worse/flat, auc↑=worse/flat, bal_acc↑=worse/flat, acc↑=worse/flat_

### Stratified by T = vel_entropy@L9 (Q4 = burstiest = hardest)
| Quartile | n | eer (Δ) | auc (Δ) | bal_acc (Δ) | acc (Δ) |
|---|---|---|---|---|---|
| Q1 | 16 | 0.214 (+0.015) | 0.853 (-0.004) | 0.827 (-0.012) | 0.841 (-0.032) |
| Q2 | 16 | 0.282 (+0.016) | 0.776 (-0.005) | 0.770 (-0.015) | 0.762 (+0.048) |
| Q3 | 15 | 0.251 (+0.019) | 0.812 (-0.028) | 0.792 (-0.023) | 0.755 (-0.041) |
| Q4 | 16 | 0.362 (+0.015) | 0.678 (-0.027) | 0.697 (-0.006) | 0.702 (-0.017) |

_T-Q4 cross-metric: CONSISTENT no-improvement/regression — eer↓=worse/flat, auc↑=worse/flat, bal_acc↑=worse/flat, acc↑=worse/flat_