# E5 ITW — Does the C/T hard manifold explain InTheWild difficulty?

ITW subset: spoof=3000 bona-fide=3000 | MLAAD reference utts=938 across 63 systems
Extraction sanity vs sig_layer_features: rog r=0.256, vel r=0.072

## Verdict: hypothesis **NOT supported** (NEITHER)
ITW spoof samples are enriched in the MLAAD hard region on: **NEITHER**.

## 1. Distribution shift (ITW-spoof vs MLAAD-spoof)
| axis | ITW mean | MLAAD mean | Cliff's δ | Cohen's d | MW p(ITW>MLAAD) | KS p |
|---|---|---|---|---|---|---|
| C | -10.734 | -10.726 | +0.020 | -0.019 | 1.78e-01 | 7.95e-01 |
| T | 2.731 | 2.769 | -0.303 | -0.421 | 1.00e+00 | 9.17e-33 |

## 2. Q4 hard-region enrichment (ITW spoof; null=0.25)
| axis | scheme | threshold | frac in Q4 | enrichment | binom p |
|---|---|---|---|---|---|
| C | per-system(a) | -10.936 | 0.724 | 2.89x | 0.00e+00 |
| T | per-system(a) | 2.793 | 0.236 | 0.95x | 9.61e-01 |
| C | per-sample(b) | -10.627 | 0.260 | 1.04x | 1.07e-01 |
| T | per-sample(b) | 2.829 | 0.106 | 0.42x | 1.00e+00 |
| C&T | per-sample(b) | — | 0.031 | 0.50x (null .0625) | 1.00e+00 |

## 3. Within-ITW (spoof vs bona-fide)
| axis | spoof mean | bona mean | Cliff's δ | MW p |
|---|---|---|---|---|
| C(spoof>bona) | -10.734 | -10.584 | -0.236 | 1.00e+00 |
| T(spoof>bona) | 2.731 | 2.740 | -0.077 | 1.00e+00 |

## 4. Difficulty anchor (per checkpoint)
MLAAD reference EER: overall=0.2615  Q4-C=0.3367  Q4-T=0.3472

| checkpoint | ITW EER | ρ(C,logit) | ρ(T,logit) |
|---|---|---|---|
| robust_goat | 0.2883 | -0.156 (p=9.5e-18) | +0.004 (p=8.4e-01) |
| mlaad_robust_goat | 0.4153 | -0.177 (p=2.0e-22) | -0.088 (p=1.3e-06) |
| MLAAD (p1 ref) | — | -0.146 | +0.150 |

_DISSOCIATION: ITW is genuinely hard (EER comparable to/worse than MLAAD-Q4), and the C-axis *direction* still holds within ITW (ρ(C,logit)<0, matching MLAAD) — yet the geometry tests above show ITW spoof does NOT sit in the MLAAD hard region. So ITW difficulty is real but NOT explained by elevated C/T-hardness; it points to domain shift rather than hard-manifold overlap._

## 5. Embedding-manifold similarity (WavLM-L12, standardized)
- kNN(k=15) ITW-spoof neighbours falling in MLAAD-Q4: **0.085** (chance 0.25)
- centroid distance ITW-spoof → MLAAD-Q4 = 11.59 vs → MLAAD-easy = 9.37 (closer to easy)
- silhouette {ITW, Q4, easy} = -0.031
- projection used for figures: t-SNE

## Figures
- `figures/e5_itw_ct_manifold.png` — C-T scatter + Q4 region
- `figures/e5_itw_ct_distributions.png` — per-sample C/T histograms
- `figures/e5_itw_score_vs_ct.png` — logit vs C/T
- `figures/e5_itw_manifold_pca_2d.png` / `_3d.png` — joint PCA
- `figures/e5_itw_manifold_umap_2d.png` / `_3d.png` — t-SNE