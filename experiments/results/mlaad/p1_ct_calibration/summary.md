# P1: Inference-time Score Calibration using C and T

## Setup
- **Model**: mlaad_robust_goat.ckpt
- **Test split**: 1846 utterances (908 bonafide, 938 spoof)
- **Systems**: 63 evaluated (min 5 spoof samples)
- **Calibration**: LOO logistic regression on [raw_logit, C, T] (standardized)

## Key Diagnostic: Do C/T Discriminate Bonafide vs Spoof?

| Feature | Bonafide mean | Spoof mean | Δ(Spoof−Bonafide) |
|---------|--------------|------------|-------------------|
| C (rog@L12) | 13.28467 | 13.47886 | +0.19419 |
| T (vel_entropy@L9) | 2.61464 | 2.62186 | +0.00722 |

## Calibration Coefficients (LOO mean ± std)

| Coefficient | Mean | Std |
|------------|------|-----|
| β_logit | 1.2595 | 0.0152 |
| β_C (rog@L12) | 0.3338 | 0.0209 |
| β_T (vel_entropy@L9) | -0.4030 | 0.0177 |

## Results: All Metrics

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Mean EER (all 63 systems) | 0.2615 | 0.2568 | +0.0047 |
| Median EER | 0.2500 | 0.2478 | +0.0022 |
| Hard-quartile EER (≥75th pct) | 0.4450 | 0.4443 | +0.0007 |
| Mean AUC | 0.7947 | 0.7961 | +0.0014 |
| Hard-quartile AUC | 0.5927 | 0.5907 | -0.0020 |
| Mean Acc (Youden thr) | 0.7752 | 0.7681 | -0.0072 |
| Mean Balanced Acc | 0.7849 | 0.7874 | +0.0025 |
| Systems improved (EER) | 36/63 | — | — |

## Calibration Gain by System C Quartile

| C Quartile | ΔEER | ΔAUC | Δbal_acc | Note |
|-----------|------|------|----------|------|
| Q1 (compact (harder)) | -0.0040 | -0.0035 | -0.0000 | |
| Q2 (compact (harder)) | -0.0021 | -0.0012 | +0.0001 | |
| Q3 (spread (easier)) | +0.0168 | +0.0103 | +0.0084 | |
| Q4 (spread (easier)) | +0.0089 | +0.0005 | +0.0021 | |

## Interpretation

C differs between bonafide (13.285) and spoof (13.479) by +0.1942 — utterance-level separation exists.
β_C=0.3338 > 0: spread (high rog) → higher spoof score. ✗ Unexpected.

**Conclusion: Calibration is neutral (ΔEER=+0.0047, ΔAUC=+0.0014). C/T explain system-level hardness but not utterance-level discrimination.**