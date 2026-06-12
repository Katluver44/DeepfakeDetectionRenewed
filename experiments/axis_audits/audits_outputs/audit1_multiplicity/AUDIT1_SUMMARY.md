# Audit 1 — Global Multiplicity / P-Hacking Audit

**Question.** The I/J experiment series recorded dozens of predictor→hardness correlations across corpora and detectors, but every p-value in `final_outputs2` is uncorrected and the headline table presents only winners. Is the result set p-hacked?

**Method.** Enumerated every recorded predictor→hardness test from the artifacts (`i2 univariate battery` 30 features, `i3`, `i4`, `i5`, `j2`, `j4`, `j5`, `j6` stats JSONs; J4/J5 p-values replaced by exact permutation values from Audit 3) → **83 tests**. Applied Benjamini–Hochberg globally and per-family. Then ran a winner's-curse simulation: 2,000 random half-splits of the 61 MLAAD systems; on each discovery half, pick the best of 33 candidate features; measure its correlation on the held-out confirmation half.

**Results** (`master_test_table.csv`, `audit1_multiplicity.png`).

- 83 recorded tests; 25 significant uncorrected; **only 1 survives global BH at q<0.05: sd_along on MLAAD (q≈10⁻⁵)**.
- Headline claims after correction:

| claim | p (exact) | q (global BH) | q (within family) |
|---|---|---|---|
| sd_along, MLAAD, WavLM-GAT | <1e-5 | **<0.001** | <0.001 |
| s_orth, ITW | 0.0028 | 0.059 | **0.013** |
| sd_along, MLAAD, AASIST-FT | 0.0058 | 0.072 | **0.018** |
| vel_entropy, MLAAD, AASIST-FT | 0.0070 | 0.072 | **0.018** |
| P3, ASVspoof21, AASIST-ZS | 0.0204 | 0.106 | 0.134 |
| P3, ASVspoof21, WavLM-GAT | 0.0336 | 0.140 | 0.134 |

- The I2 battery's own FDR column shows **zero of 30 features pass q<0.05** (best: vel_entropy_L9, q=0.078). The paper's §3.1/§5 present these battery numbers without noting this.
- **Winner's-curse simulation:** sd_along wins 76.7% of discovery halves; when it wins, its median confirmation-half |ρ| is 0.568 and it confirms at p<0.05 in **99.2%** of splits. This is the signature of a real effect, not selection noise.

**Verdict.** The *headline MLAAD finding (sd_along) decisively survives* the harshest multiplicity treatment and an explicit winner's-curse test. The *ASVspoof21 prospective claims (P3) do not survive any correction* and must be framed as suggestive (see Audit 3). The AASIST-FT and ITW replications survive within-family but not global correction — they are supported as replications of pre-specified quantities (which is the correct framing for them), not as independent discoveries. The dynamics battery (vel_entropy etc.) should not be presented without its own FDR column, which it currently fails.
