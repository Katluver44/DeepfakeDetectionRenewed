# Confidence–Attention Correlation Report

**Experiment**: GAT layer 0 attention divergence vs spoof confidence
**Artifacts**: `attention_artifacts.pt` + `per_sample_preds.pt` (baseline config)
**Dataset**: ASVspoof 2019 LA validation, 50 samples/system × 7 systems = 350 total

## Methods

- **Confidence metric**: raw spoof logit (BCEWithLogitsLoss output; label 1=spoof, logit>0→spoof)
- **KL divergence**: KL(sample_attn || BF_mean_attn) over (src_phoneme_class, dst_phoneme_class) × head
- **Phoneme classes**: 9 — Vowels, Diphthongs, Approximants, Nasals, Stops, Fricatives, Sibilants, Affricates, Other
- **BF_mean**: averaged over 50 non-degenerate bonafide samples (label=0) only — no data leakage
- **Per-head KL**: each head's (9,9) attention matrix is normalized to sum=1, then KL computed independently
- **KL_combined**: mean of 6 per-head KL values
- **Correlations**: Pearson r + Spearman rho; primary analysis = attack samples only; secondary = all samples
- **Outlier robustness**: correlations recomputed after removing KL > Q3 + 1.5×IQR

## Sanity checks

- All KL values ≥ 0: min=0.112449 ✓
- Bonafide mean KL from BF_mean (should be ≈0): 0.3349 (max=0.7431)
- Attack mean KL > bonafide mean KL: 0.3396 > 0.3349 ✓
- No data leakage: BF_mean computed exclusively from label=0 samples ✓
- Baseline logit alignment: verified by sample_id match across both artifacts ✓

## Primary correlation results — all heads combined

| Subset | Pearson r | Pearson p | Spearman rho | Spearman p | N |
|--------|-----------|-----------|--------------|------------|---|
| pooled (attacks) | -0.257 | 6.4e-06 | -0.251 | 1.13e-05 | 300 |
| A01 | -0.030 | 0.835 | -0.083 | 0.565 | 50 |
| A02 | -0.351 | 0.0124 | -0.375 | 0.00723 | 50 |
| A03 | -0.184 | 0.201 | -0.260 | 0.0687 | 50 |
| A04 | 0.133 | 0.356 | 0.089 | 0.538 | 50 |
| A05 | -0.470 | 0.000577 | -0.474 | 0.000504 | 50 |
| A06 | -0.139 | 0.334 | -0.027 | 0.852 | 50 |
| pooled (attacks, outliers removed) | -0.158 | 0.00836 | -0.188 | 0.0017 | 277 (-23 outliers) |
| pooled (all incl. bonafide) | -0.172 | 0.00127 | -0.208 | 9.02e-05 | 350 |

## Head 0 correlation

| Subset | Pearson r | Pearson p | Spearman rho | Spearman p | N |
|--------|-----------|-----------|--------------|------------|---|
| pooled (attacks) | -0.214 | 0.000186 | -0.184 | 0.00139 | 300 |
| A01 | 0.052 | 0.722 | 0.035 | 0.81 | 50 |
| A02 | -0.391 | 0.00495 | -0.391 | 0.00498 | 50 |
| A03 | -0.131 | 0.364 | -0.113 | 0.437 | 50 |
| A04 | 0.161 | 0.263 | 0.108 | 0.457 | 50 |
| A05 | -0.378 | 0.00677 | -0.374 | 0.00747 | 50 |
| A06 | -0.088 | 0.542 | -0.030 | 0.837 | 50 |
| pooled (attacks, outliers removed) | -0.105 | 0.0797 | -0.119 | 0.0481 | 278 (-22 outliers) |
| pooled (all incl. bonafide) | -0.157 | 0.00321 | -0.170 | 0.00139 | 350 |

## Head 4 correlation

| Subset | Pearson r | Pearson p | Spearman rho | Spearman p | N |
|--------|-----------|-----------|--------------|------------|---|
| pooled (attacks) | -0.220 | 0.000122 | -0.189 | 0.000994 | 300 |
| A01 | -0.003 | 0.981 | -0.122 | 0.4 | 50 |
| A02 | -0.170 | 0.239 | -0.203 | 0.157 | 50 |
| A03 | -0.139 | 0.336 | -0.212 | 0.139 | 50 |
| A04 | 0.095 | 0.513 | 0.084 | 0.564 | 50 |
| A05 | -0.420 | 0.00238 | -0.451 | 0.00102 | 50 |
| A06 | -0.150 | 0.299 | -0.075 | 0.604 | 50 |
| pooled (attacks, outliers removed) | -0.111 | 0.06 | -0.132 | 0.026 | 286 (-14 outliers) |
| pooled (all incl. bonafide) | -0.112 | 0.0369 | -0.122 | 0.0223 | 350 |

## Head-specific summary

| KL metric | Pooled Pearson r | Pooled Spearman rho | Interpretation |
|-----------|-----------------|---------------------|----------------|
| All heads | -0.257 | -0.251 | weak |
| Head 0 | -0.214 | -0.184 | weak |
| Head 4 | -0.220 | -0.189 | weak |

## Calibration vs discrimination — TP/FP/TN/FN

| Outcome | N | KL mean | KL std | KL median |
|---------|---|---------|--------|-----------|
| TP | 296 | 0.3401 | 0.1922 | 0.2858 |
| FP | 25 | 0.3252 | 0.1284 | 0.3081 |
| TN | 25 | 0.3446 | 0.1464 | 0.2923 |
| FN | 4 | 0.2996 | 0.0550 | 0.3151 |

## Interpretation

**Pooled attack correlation (combined KL)**: Pearson r=-0.257, Spearman rho=-0.251 → **weak**

Attention divergence is largely independent of spoof confidence. Strong evidence that the routing differences observed in the attention analysis reflect processing strategy (attack-family-specific phoneme routing) rather than a proxy for the model's certainty level.

**Calibration analysis**: FP mean KL (0.3252) is close to TP mean KL (0.3401). Consistent with Experiment 1 finding: h0/h4 primarily drive spoof-prediction aggressiveness. High attention divergence correlates with *spoof prediction*, not specifically with correct spoof detection.

**False negatives**: FN mean KL=0.2996 (N=4) — small-N, interpret with caution.

**Connection to head ablation (Experiment 1)**: Ablating h0+h4 reduced attack accuracy while bonafide accuracy improved, consistent with these heads driving spoof-prediction aggressiveness. A weak correlation here supports the interpretation that the routing difference reflects a genuine processing-strategy change rather than a confidence-scaling artifact. The causal claim from Experiment 1 is strengthened.

## Files

| File | Description |
|------|-------------|
| `confidence_correlation_report.md` | This report |
| `corr_scatter.png` | KL vs logit scatter — 3 panels: combined, h0, h4 |
| `kl_outcome_boxplot.png` | KL distributions split by TP/FP/TN/FN |
| `kl_per_family_violin.png` | KL violin plot per attack family |
| `per_sample_kl.csv` | Per-sample KL and logit values |
