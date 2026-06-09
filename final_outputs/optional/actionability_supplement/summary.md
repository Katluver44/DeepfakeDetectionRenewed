# P8: Cross-encoder C ensemble (WavLM L12 + wav2vec2 L8) — diagnostic gate

Utterances: 1846  |  systems: 64

## Mean per-system metrics (overall | hard-C-quartile)
| Score | Mean EER | Mean AUC | Mean bal_acc | HardQ EER | HardQ AUC |
|-------|----------|----------|--------------|-----------|-----------|
| raw_logit (no cal) | 0.2615 | 0.7947 | 0.7849 | 0.3367 | 0.7136 |
| + wavlm-only cal | 0.4466 | 0.5687 | 0.6700 | 0.4977 | 0.5073 |
| + ensemble cal | 0.4453 | 0.5662 | 0.6682 | 0.4952 | 0.5066 |

**Decision: NO-GO — ensembling does not jointly improve hard-quartile EER and AUC over WavLM-only; not worth 2× inference**
(criterion: hard-quartile EER ≤ wavlm−0.01 AND AUC ≥ wavlm;  ΔEER=-0.0025, ΔAUC=-0.0008)

## Ensemble-calibrated metrics vs raw, by C/T quartile

Systems with C/T + metrics: 63

### Overall (model | Δ vs baseline)
- eer (↓ better): 0.4453   Δ=+0.1838
- auc (↑ better): 0.5662   Δ=-0.2285
- bal_acc (↑ better): 0.6682   Δ=-0.1167
- acc (↑ better): 0.3927   Δ=-0.3825

### Stratified by C = −rog@L12 (Q4 = most compact = hardest)
| Quartile | n | eer (Δ) | auc (Δ) | bal_acc (Δ) | acc (Δ) |
|---|---|---|---|---|---|
| Q1 | 16 | 0.426 (+0.203) | 0.592 (-0.249) | 0.689 (-0.130) | 0.439 (-0.340) |
| Q2 | 16 | 0.411 (+0.195) | 0.605 (-0.223) | 0.685 (-0.133) | 0.455 (-0.353) |
| Q3 | 15 | 0.450 (+0.178) | 0.561 (-0.235) | 0.661 (-0.120) | 0.345 (-0.443) |
| Q4 | 16 | 0.495 (+0.159) | 0.507 (-0.207) | 0.637 (-0.084) | 0.329 (-0.398) |

_C-Q4 cross-metric: CONSISTENT no-improvement/regression — eer↓=worse/flat, auc↑=worse/flat, bal_acc↑=worse/flat, acc↑=worse/flat_

### Stratified by T = vel_entropy@L9 (Q4 = burstiest = hardest)
| Quartile | n | eer (Δ) | auc (Δ) | bal_acc (Δ) | acc (Δ) |
|---|---|---|---|---|---|
| Q1 | 16 | 0.400 (+0.201) | 0.623 (-0.233) | 0.696 (-0.143) | 0.469 (-0.405) |
| Q2 | 16 | 0.439 (+0.174) | 0.565 (-0.216) | 0.673 (-0.112) | 0.388 (-0.326) |
| Q3 | 15 | 0.437 (+0.205) | 0.584 (-0.256) | 0.681 (-0.134) | 0.420 (-0.376) |
| Q4 | 16 | 0.504 (+0.157) | 0.493 (-0.211) | 0.623 (-0.079) | 0.296 (-0.423) |

_T-Q4 cross-metric: CONSISTENT no-improvement/regression — eer↓=worse/flat, auc↑=worse/flat, bal_acc↑=worse/flat, acc↑=worse/flat_

## Note
This is a calibration diagnostic, not a trained model — consistent with P1's
finding that C/T are system-level (not utterance-level) signals. A GO here
means the only code change for full P8 is computing C in modules_ct.py as the
z-averaged (WavLM-L12, wav2vec2-L8) rog instead of WavLM-L12 alone.