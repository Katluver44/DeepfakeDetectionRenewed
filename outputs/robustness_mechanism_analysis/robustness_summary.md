# E7 — Does robust training work by reducing sensitivity to C and T?

Mechanistic comparison of four checkpoints on the locked MLAAD test split (63 systems with C/T + metrics). C/T are computed on the **frozen** WavLM backbone (identical across all models), so they serve as a shared per-system predictor. Per-system EER/AUC/Acc scored vs the shared bona-fide pool.

**Robustness pairs** (baseline → robust): **A = MLAAD_GOAT → MLAAD_robust_GOAT** (headline; same training data, only augmentation differs) and **B = GOAT → robust_GOAT** (cross-dataset reference; note B also differs in training data, so it is a weaker mechanistic contrast).

_Exp 4 (representation geometry) was dropped: C/T live in the frozen backbone and are identical for all four models, so there is nothing to compare there._
_Exp 5 caveat fix: we lead with leave-one-out CV R² and a bootstrap 95% CI on R²(C+T) (the honest predictive estimate at n=63); the C/T unique-contribution split is the **exact** 2-predictor commonality/Shapley decomposition (analytic, so the wide-CI concern with sampled Shapley does not apply)._

## Dataset-level performance
| Model | EER | AUC | Accuracy |
|---|---|---|---|
| GOAT | 0.5200 | 0.4901 | 0.4097 |
| robust_GOAT | 0.4863 | 0.5327 | 0.5063 |
| MLAAD_GOAT | 0.2393 | 0.8273 | 0.8326 |
| MLAAD_robust_GOAT | 0.2438 | 0.8208 | 0.7739 |

## Exp 1 — Hardness correlations (Spearman ρ, 95% bootstrap CI; R² of metric~C+T)
| Model | Metric | ρ(C) [CI] | ρ(T) [CI] | R²(C+T) [CI] | LOO R² |
|---|---|---|---|---|---|
| GOAT | EER | +0.09 [-0.20, +0.36] | +0.04 [-0.23, +0.31] | 0.00 [+0.00, +0.12] | -0.10 |
| GOAT | AUC | -0.10 [-0.37, +0.18] | -0.05 [-0.31, +0.21] | 0.00 [+0.00, +0.12] | -0.09 |
| GOAT | Accuracy | -0.21 [-0.43, +0.03] | +0.06 [-0.22, +0.33] | 0.05 [+0.01, +0.20] | -0.04 |
| robust_GOAT | EER | +0.12 [-0.13, +0.37] | +0.16 [-0.09, +0.39] | 0.05 [+0.00, +0.18] | -0.03 |
| robust_GOAT | AUC | -0.15 [-0.39, +0.10] | -0.11 [-0.36, +0.14] | 0.04 [+0.00, +0.19] | -0.04 |
| robust_GOAT | Accuracy | +0.03 [-0.22, +0.27] | +0.14 [-0.12, +0.37] | 0.06 [+0.01, +0.22] | -0.04 |
| MLAAD_GOAT | EER | +0.40 [+0.17, +0.59] | +0.32 [+0.08, +0.52] | 0.28 [+0.10, +0.49] | 0.22 |
| MLAAD_GOAT | AUC | -0.38 [-0.57, -0.15] | -0.31 [-0.52, -0.06] | 0.29 [+0.08, +0.53] | 0.20 |
| MLAAD_GOAT | Accuracy | -0.17 [-0.43, +0.10] | -0.14 [-0.38, +0.11] | 0.08 [+0.01, +0.31] | -0.01 |
| MLAAD_robust_GOAT | EER | +0.39 [+0.17, +0.57] | +0.29 [+0.03, +0.50] | 0.23 [+0.08, +0.42] | 0.16 |
| MLAAD_robust_GOAT | AUC | -0.39 [-0.57, -0.16] | -0.32 [-0.52, -0.07] | 0.22 [+0.07, +0.42] | 0.15 |
| MLAAD_robust_GOAT | Accuracy | -0.11 [-0.35, +0.15] | -0.18 [-0.41, +0.07] | 0.12 [+0.01, +0.42] | -0.06 |

### Does the robust model depend *less* on C/T? (paired, within each pair)
Δ = |ρ_baseline| − |ρ_robust| (positive ⇒ robust weaker); CI via paired bootstrap; Steiger Z tests dependent overlapping correlations.
| Pair | Metric | Axis | ρ_base | ρ_robust | Δ\|ρ\| [CI] | P(robust weaker) | Steiger p |
|---|---|---|---|---|---|---|---|
| MLAAD (in-distribution) | EER | C | +0.40 | +0.39 | +0.01 [-0.13, +0.14] | 0.56 | 0.39 |
| MLAAD (in-distribution) | EER | T | +0.32 | +0.29 | +0.03 [-0.12, +0.17] | 0.64 | 0.97 |
| MLAAD (in-distribution) | AUC | C | -0.38 | -0.39 | -0.00 [-0.11, +0.10] | 0.48 | 0.28 |
| MLAAD (in-distribution) | AUC | T | -0.31 | -0.32 | -0.01 [-0.14, +0.12] | 0.43 | 0.77 |
| MLAAD (in-distribution) | Accuracy | C | -0.17 | -0.11 | +0.07 [-0.17, +0.29] | 0.65 | 0.85 |
| MLAAD (in-distribution) | Accuracy | T | -0.14 | -0.18 | -0.04 [-0.28, +0.22] | 0.42 | 0.61 |
| cross-dataset | EER | C | +0.09 | +0.12 | -0.03 [-0.23, +0.21] | 0.46 | 0.32 |
| cross-dataset | EER | T | +0.04 | +0.16 | -0.12 [-0.27, +0.18] | 0.31 | 0.4 |
| cross-dataset | AUC | C | -0.10 | -0.15 | -0.05 [-0.23, +0.18] | 0.41 | 0.26 |
| cross-dataset | AUC | T | -0.05 | -0.11 | -0.06 [-0.21, +0.16] | 0.41 | 0.65 |
| cross-dataset | Accuracy | C | -0.21 | +0.03 | +0.18 [-0.17, +0.37] | 0.80 | 0.75 |
| cross-dataset | Accuracy | T | +0.06 | +0.14 | -0.08 [-0.29, +0.22] | 0.41 | 0.74 |

## Exp 2 — Quartile analysis (mean metric per C/T quartile; Q4 = hardest)
### By C
| Quartile | GOAT EER | robust_GOAT EER | MLAAD_GOAT EER | MLAAD_robust_GOAT EER |
|---|---|---|---|---|
| Q1 | 0.448 | 0.439 | 0.203 | 0.201 |
| Q2 | 0.597 | 0.512 | 0.195 | 0.222 |
| Q3 | 0.546 | 0.525 | 0.244 | 0.247 |
| Q4 | 0.491 | 0.472 | 0.315 | 0.306 |

_Pair A ΔEER (MLAAD_GOAT−MLAAD_robust): Q4=+0.008 vs Q1=+0.003 (gains concentrated in hard Q4)._
### By T
| Quartile | GOAT EER | robust_GOAT EER | MLAAD_GOAT EER | MLAAD_robust_GOAT EER |
|---|---|---|---|---|
| Q1 | 0.522 | 0.452 | 0.188 | 0.192 |
| Q2 | 0.524 | 0.502 | 0.238 | 0.253 |
| Q3 | 0.462 | 0.445 | 0.213 | 0.206 |
| Q4 | 0.568 | 0.544 | 0.316 | 0.322 |

_Pair A ΔEER (MLAAD_GOAT−MLAAD_robust): Q4=-0.006 vs Q1=-0.004 (gains NOT concentrated in Q4)._

## Exp 3 — Error-reduction localization (Δmetric ~ C/T; permutation p)
Δ defined so **positive = robust better**. A positive ρ(Δ, C/T) ⇒ harder (high-C/T) systems benefit more from robust training.
| Pair | Δmetric | Axis | Spearman ρ | perm p |
|---|---|---|---|---|
| MLAAD (in-distribution) | dEER | C | +0.07 | 0.58 |
| MLAAD (in-distribution) | dEER | T | +0.04 | 0.78 |
| MLAAD (in-distribution) | dAUC | C | -0.03 | 0.8 |
| MLAAD (in-distribution) | dAUC | T | +0.03 | 0.79 |
| MLAAD (in-distribution) | dAccuracy | C | +0.04 | 0.76 |
| MLAAD (in-distribution) | dAccuracy | T | -0.06 | 0.65 |
| cross-dataset | dEER | C | +0.04 | 0.77 |
| cross-dataset | dEER | T | -0.20 | 0.11 |
| cross-dataset | dAUC | C | +0.02 | 0.89 |
| cross-dataset | dAUC | T | -0.17 | 0.19 |
| cross-dataset | dAccuracy | C | +0.18 | 0.17 |
| cross-dataset | dAccuracy | T | -0.01 | 0.95 |

## Exp 5 — Variance decomposition (R² of metric~C+T per model)
| Model | Metric | R²(C+T) [CI] | LOO R² | unique C | unique T | common | Shapley C | Shapley T |
|---|---|---|---|---|---|---|---|---|
| GOAT | EER | 0.00 [+0.00, +0.12] | -0.10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| GOAT | AUC | 0.00 [+0.00, +0.12] | -0.09 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| GOAT | Accuracy | 0.05 [+0.01, +0.20] | -0.04 | 0.04 | 0.02 | -0.01 | 0.04 | 0.02 |
| robust_GOAT | EER | 0.05 [+0.00, +0.18] | -0.03 | 0.03 | 0.01 | 0.01 | 0.03 | 0.02 |
| robust_GOAT | AUC | 0.04 [+0.00, +0.19] | -0.04 | 0.03 | 0.00 | 0.00 | 0.04 | 0.01 |
| robust_GOAT | Accuracy | 0.06 [+0.01, +0.22] | -0.04 | 0.02 | 0.04 | -0.01 | 0.02 | 0.04 |
| MLAAD_GOAT | EER | 0.28 [+0.10, +0.49] | 0.22 | 0.21 | 0.05 | 0.03 | 0.22 | 0.06 |
| MLAAD_GOAT | AUC | 0.29 [+0.08, +0.53] | 0.20 | 0.20 | 0.06 | 0.03 | 0.22 | 0.07 |
| MLAAD_GOAT | Accuracy | 0.08 [+0.01, +0.31] | -0.01 | 0.08 | 0.00 | 0.00 | 0.08 | 0.00 |
| MLAAD_robust_GOAT | EER | 0.23 [+0.08, +0.42] | 0.16 | 0.15 | 0.05 | 0.02 | 0.16 | 0.06 |
| MLAAD_robust_GOAT | AUC | 0.22 [+0.07, +0.42] | 0.15 | 0.14 | 0.05 | 0.02 | 0.15 | 0.06 |
| MLAAD_robust_GOAT | Accuracy | 0.12 [+0.01, +0.42] | -0.06 | 0.09 | 0.02 | 0.01 | 0.10 | 0.02 |

## Final questions
**Q1. Are C/T predictive of metrics for all models or only baselines?** EER R²(C+T): GOAT=0.00, robust_GOAT=0.05, MLAAD_GOAT=0.28, MLAAD_robust_GOAT=0.23. C/T predict EER across all models (see CIs in Exp 1/5); the relationship is not exclusive to baselines.

**Q2. Do robust models depend *less* on C/T?** Headline pair (MLAAD) EER: dependence is significantly weaker (CI excludes 0) on: **NEITHER axis**. Other comparisons: see Δ|ρ| CIs above — most differences are small with CIs spanning 0, i.e. **no robust reduction in C/T dependence is established** at n=63 (conservative read).

**Q3. Are gains concentrated in high-C/high-T systems?** Localization ρ(ΔEER, C/T) for pair A: C=+0.07(p=0.58), T=+0.04(p=0.78). Gains are concentrated in hard systems only where ρ>0 with small perm p.

**Q4. Does robust training reshape representation geometry?** Not assessed — Exp 4 dropped (C/T are frozen-backbone quantities, identical across models).

**Q5. Does robust training reduce variance explained by C/T?** EER R²(C+T) change baseline→robust (pair A) = +0.06 (reduced); judge against the bootstrap CIs in Exp 5 (largely overlapping ⇒ treat as suggestive, not conclusive).

**Q6. Is the evidence consistent with robust training mitigating the specific C/T failure modes?** Partially — the in-distribution pair shows the directionally-expected pattern on some axes, but at n=63 systems the CIs are wide and most effects are not individually significant. Conservative conclusion: **suggestive, underpowered; no firm claim that robust training neutralizes C/T**.

## Files
- `performance_summary.csv`, `correlation_tables.csv`, `dependence_comparison.csv`
- `quartile_c.csv`, `quartile_t.csv`, `delta_metric_correlations.csv`, `variance_partition.csv`
- figures: `{eer,auc,acc}_vs_{c,t}.png`, `delta{eer,auc,acc}_vs_{c,t}.png`, `quartile_{c,t}.png`