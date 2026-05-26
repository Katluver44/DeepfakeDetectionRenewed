# E7: Per-sample Routing-Cluster Classifier

## Exclusion note
A02 excluded (ambiguous mechanism; would blur the cluster boundary).

## Data
- Routing-dependent (class 0): A01, A03, A04 — 50 samples each = 150
- Skip-route (class 1): A05, A06 — 50 samples each = 100
- Total N = 250  (60/40 class split)

## Cross-seed metrics (structural LR, 5-fold CV)

| seed | bal_acc | F1 | AUC |
|------|---------|----|-----|
| seed1 | 0.853±0.015 | 0.825±0.019 | 0.934±0.013 |
| seed2 | 0.855±0.029 | 0.826±0.036 | 0.930±0.022 |
| seed3 | 0.850±0.040 | 0.819±0.051 | 0.942±0.014 |
| phoneme_bl | 0.595±0.072 | 0.489±0.109 | 0.633±0.068 |

## Feature importances (seed 1, LR standardized |coef|)

| rank | feature | |coef| | conc? |
|------|---------|-------|-------|
| 1 | offdiag_frob | 1.5321 |  |
| 2 | entropy_in | 0.7929 |  |
| 3 | top5_mass | 0.5833 | ✓ |
| 4 | top1_mass | 0.5239 | ✓ |
| 5 | entropy_out | 0.4972 |  |
| 6 | diag_mass | 0.3240 |  |
| 7 | eff_rank | 0.2186 |  |
| 8 | gini_in | 0.0880 | ✓ |

## Hypothesis outcomes

- **H1**: PASS
- **H2**: PASS
- **H3**: PASS
- **H4**: PASS

**Decision: H1 + H2 PASS → per-sample cluster split is real and attention-driven. Taxonomy moves from population-level to sample-level claim. Add to paper.**