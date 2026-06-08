# P4: Smoothing Augmentation Fine-tuning

**p_smooth=0.3  |  kernels=[3, 5, 7]  |  seeds=[1024]  |  epochs=5**

## Motivation
E3 shows temporal smoothing (MA window≥3) directly reduces C (rog) by averaging frames.
Training on smoothed bonafide speech forces the model to distinguish compact-but-real from
compact-and-synthetic. Kernels ≥11 EXCLUDED because E3 shows they reverse the C correlation.

## Per-system metrics vs robust_goat baseline

Systems with C/T + metrics: 63

### Overall (model | Δ vs baseline)
- eer (↓ better): 0.3099   Δ=+0.0484
- auc (↑ better): 0.7462   Δ=-0.0485
- bal_acc (↑ better): 0.7431   Δ=-0.0418
- acc (↑ better): 0.7529   Δ=-0.0224

### Stratified by C = −rog@L12 (Q4 = most compact = hardest)
| Quartile | n | eer (Δ) | auc (Δ) | bal_acc (Δ) | acc (Δ) |
|---|---|---|---|---|---|
| Q1 | 16 | 0.271 (+0.049) | 0.804 (-0.038) | 0.774 (-0.045) | 0.779 (+0.000) |
| Q2 | 16 | 0.281 (+0.065) | 0.773 (-0.055) | 0.765 (-0.053) | 0.772 (-0.035) |
| Q3 | 15 | 0.320 (+0.048) | 0.738 (-0.058) | 0.737 (-0.045) | 0.788 (-0.001) |
| Q4 | 16 | 0.369 (+0.032) | 0.670 (-0.044) | 0.696 (-0.025) | 0.674 (-0.053) |

_C-Q4 cross-metric: CONSISTENT no-improvement/regression — eer↓=worse/flat, auc↑=worse/flat, bal_acc↑=worse/flat, acc↑=worse/flat_

### Stratified by T = vel_entropy@L9 (Q4 = burstiest = hardest)
| Quartile | n | eer (Δ) | auc (Δ) | bal_acc (Δ) | acc (Δ) |
|---|---|---|---|---|---|
| Q1 | 16 | 0.257 (+0.058) | 0.814 (-0.042) | 0.787 (-0.052) | 0.793 (-0.081) |
| Q2 | 16 | 0.316 (+0.051) | 0.739 (-0.042) | 0.739 (-0.046) | 0.758 (+0.044) |
| Q3 | 15 | 0.283 (+0.050) | 0.787 (-0.053) | 0.769 (-0.046) | 0.744 (-0.053) |
| Q4 | 16 | 0.382 (+0.035) | 0.648 (-0.057) | 0.678 (-0.025) | 0.717 (-0.002) |

_T-Q4 cross-metric: CONSISTENT no-improvement/regression — eer↓=worse/flat, auc↑=worse/flat, bal_acc↑=worse/flat, acc↑=worse/flat_