# P7 CLIP C-ranking (lambda=0.1) — mlaad_robust_goat — Summary

seeds aggregated: ['1024', '123', '42']

## Per-system metrics vs mlaad_robust_goat baseline

Systems with C/T + metrics: 63

### Overall (model | Δ vs baseline)
- eer (↓ better): 0.2775   Δ=+0.0160
- auc (↑ better): 0.7853   Δ=-0.0094
- bal_acc (↑ better): 0.7717   Δ=-0.0131
- acc (↑ better): 0.7518   Δ=-0.0234

### Stratified by C = −rog@L12 (Q4 = most compact = hardest)
| Quartile | n | eer (Δ) | auc (Δ) | bal_acc (Δ) | acc (Δ) |
|---|---|---|---|---|---|
| Q1 | 16 | 0.233 (+0.011) | 0.838 (-0.004) | 0.807 (-0.012) | 0.770 (-0.009) |
| Q2 | 16 | 0.250 (+0.035) | 0.814 (-0.014) | 0.797 (-0.021) | 0.809 (+0.001) |
| Q3 | 15 | 0.273 (+0.001) | 0.788 (-0.007) | 0.776 (-0.006) | 0.757 (-0.032) |
| Q4 | 16 | 0.352 (+0.016) | 0.701 (-0.013) | 0.709 (-0.012) | 0.672 (-0.055) |

_C-Q4 cross-metric: CONSISTENT no-improvement/regression — eer↓=worse/flat, auc↑=worse/flat, bal_acc↑=worse/flat, acc↑=worse/flat_

### Stratified by T = vel_entropy@L9 (Q4 = burstiest = hardest)
| Quartile | n | eer (Δ) | auc (Δ) | bal_acc (Δ) | acc (Δ) |
|---|---|---|---|---|---|
| Q1 | 16 | 0.217 (+0.018) | 0.847 (-0.009) | 0.823 (-0.016) | 0.849 (-0.024) |
| Q2 | 16 | 0.275 (+0.010) | 0.789 (+0.008) | 0.777 (-0.008) | 0.736 (+0.022) |
| Q3 | 15 | 0.259 (+0.027) | 0.813 (-0.026) | 0.792 (-0.023) | 0.733 (-0.063) |
| Q4 | 16 | 0.358 (+0.011) | 0.694 (-0.011) | 0.696 (-0.006) | 0.688 (-0.031) |

_T-Q4 cross-metric: CONSISTENT no-improvement/regression — eer↓=worse/flat, auc↑=worse/flat, bal_acc↑=worse/flat, acc↑=worse/flat_