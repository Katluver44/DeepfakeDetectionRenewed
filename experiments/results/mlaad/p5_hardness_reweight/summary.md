# p5_hardness_reweight — Summary

seeds=[123]  epochs=5  lr=5e-05

## Per-system metrics vs robust_goat baseline

Systems with C/T + metrics: 63

### Overall (model | Δ vs baseline)
- eer (↓ better): 0.2449   Δ=-0.0166
- auc (↑ better): 0.8148   Δ=+0.0201
- bal_acc (↑ better): 0.7973   Δ=+0.0124
- acc (↑ better): 0.7856   Δ=+0.0104

### Stratified by C = −rog@L12 (Q4 = most compact = hardest)
| Quartile | n | eer (Δ) | auc (Δ) | bal_acc (Δ) | acc (Δ) |
|---|---|---|---|---|---|
| Q1 | 16 | 0.210 (-0.012) | 0.859 (+0.017) | 0.825 (+0.006) | 0.814 (+0.035) |
| Q2 | 16 | 0.219 (+0.003) | 0.836 (+0.009) | 0.825 (+0.007) | 0.789 (-0.018) |
| Q3 | 15 | 0.236 (-0.036) | 0.822 (+0.026) | 0.805 (+0.023) | 0.831 (+0.042) |
| Q4 | 16 | 0.314 (-0.022) | 0.743 (+0.029) | 0.735 (+0.014) | 0.711 (-0.016) |

_C-Q4 cross-metric: MIXED (3/4 metrics agree) — eer↓=better, auc↑=better, bal_acc↑=better, acc↑=worse/flat_

### Stratified by T = vel_entropy@L9 (Q4 = burstiest = hardest)
| Quartile | n | eer (Δ) | auc (Δ) | bal_acc (Δ) | acc (Δ) |
|---|---|---|---|---|---|
| Q1 | 16 | 0.187 (-0.013) | 0.878 (+0.021) | 0.854 (+0.016) | 0.864 (-0.009) |
| Q2 | 16 | 0.250 (-0.016) | 0.816 (+0.035) | 0.793 (+0.008) | 0.772 (+0.059) |
| Q3 | 15 | 0.229 (-0.004) | 0.833 (-0.006) | 0.811 (-0.005) | 0.775 (-0.021) |
| Q4 | 16 | 0.314 (-0.033) | 0.734 (+0.029) | 0.732 (+0.029) | 0.730 (+0.011) |

_T-Q4 cross-metric: CONSISTENT improvement — eer↓=better, auc↑=better, bal_acc↑=better, acc↑=better_

## Verdict (EER-based, cross-checked against AUC/accuracy above)
**IMPROVED** — ΔEER(C-Q4 hard)=-0.0225, ΔEER(overall)=-0.0166
  hard-bucket corroboration: ΔAUC=+0.0292, Δbal_acc=+0.0141
  C-Q4 cross-metric: MIXED (3/4 metrics agree) — eer↓=better, auc↑=better, bal_acc↑=better, acc↑=worse/flat
  T-Q4 cross-metric: CONSISTENT improvement — eer↓=better, auc↑=better, bal_acc↑=better, acc↑=better