# K7 -- ITW Speaker-Level Spread-Factor Merge (s_orth vs rog12)

Appendix experiment (exploratory hygiene follow-up to audit6 / COMPREHENSIVE_VERDICT
S2). CPU-only, deterministic, n=29 speakers.

## Background

`audit6_itw_speaker.py` found that the paper's headline "s_orth rho=+0.534
predicts ITW per-speaker hardness" survives 7-test family correction, but
plain radius-of-gyration `rog12` is *stronger* (rho=-0.561), unreported, and
collinear with s_orth (r=-0.687); neither survives partialling the other.
COMPREHENSIVE_VERDICT S2 concludes the defensible claim is "a single
spread/compactness factor predicts ITW speaker hardness," not the specific
s_orth axis-residual interpretation. This script tests that merge directly.

## Data

`experiments/results/i5_itw_transfer/speaker_table.csv`, n=29 speakers
(pre-filtered to >=10 utterances/speaker -- the same filter and table
`audit6_itw_speaker.py` loads directly; no additional filtering applied here).

## (i) Collinearity: s_orth vs rog12

rho = -0.6867 (p=0.0000)  [audit6 reference: r=-0.687]

## Marginal associations with hardness

| feature | rho | p | reported (report) |
|---|---|---|---|
| s_orth | +0.5340 | 0.0028 | +0.534 |
| rog12 | -0.5611 | 0.0015 | -0.561 |

## (ii) PCA merge: factor-1 of standardized [s_orth, rog12]

- PC1 explains 95.2% of variance (PC2: 4.8%)
- PC1 loadings (sign-oriented so higher factor = higher expected hardness):
  s_orth=+0.7071,
  rog12=-0.7071
- **factor1 vs hardness: rho = +0.5788**
  - asymptotic p = 0.0010
  - **exact permutation p (n_perm=100000, seed=0) = 0.001120**

The merged factor's association with hardness (+0.5788) is
comparable to or stronger than
either individual predictor alone (s_orth +0.5340, rog12 -0.5611),
consistent with s_orth and rog12 carrying substantially overlapping
(collinear, r=-0.687) signal rather than independent information.

## (iii) Discriminating partial regressions

| test | rho | p |
|---|---|---|
| s_orth \| rog12 | +0.2108 | 0.2723 |
| rog12 \| s_orth | +0.0034 | 0.9858 |

**Both survive p<0.05: False**
**Either survives p<0.05: False**

## (iv) Channel-confound caveat: vmean0 (layer-0 mean velocity)

| test | rho | p |
|---|---|---|
| vmean0 vs hardness | -0.4645 | 0.0111 |
| vmean0 vs factor1 | -0.8369 | 0.0000 |
| factor1 \| vmean0 | +0.0187 | 0.9232 |

## Framing (binding, per plan)

**Max defensible claim: "a single spread/compactness factor (PC1 of [s_orth, rog12]) predicts ITW speaker hardness"** -- supported: factor1
vs hardness rho=+0.5788, exact permutation p=0.001120 (n=29).

**s_orth-specific "off-axis residual" interpretation separable from plain
compactness (rog12)? False.**

NOT separable: neither s_orth nor rog12 individually survives partialling out the other at p<0.05 (matches COMPREHENSIVE_VERDICT S2).

This matches COMPREHENSIVE_VERDICT S2: s_orth and rog12 are too collinear
(r=-0.687) at n=29 to attribute the hardness association to the
specific axis-residual construction rather than generic spread/compactness.
The vmean0 (layer-0 velocity, channel proxy) association with both hardness
(-0.4645, p=0.0111) and the merged factor (-0.8369,
p=0.0000) is reported here as a caveat, not resolved -- it is
consistent with (but does not prove) a channel-confound component in the
ITW speaker-hardness association, per audit6/S2's flag.
