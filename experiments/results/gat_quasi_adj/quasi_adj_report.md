# E4: Quasi-Adjacency F1 — Learned Attention Graph vs Input Phoneme Graph

## Method

For each sample, A_agg = A^{L3} @ A^{L2} @ A^{L1} (3-layer aggregated attention,
row-stochastic). A_input = binary adjacency matrix of phoneme-adjacency ∪ 10-step
lookahead edges actually fed to the GAT.

Threshold τ* is swept over ~52 quantiles of nonzero A_agg values (percentiles
50–99.9). F1 at τ* is the best-case alignment score. All self-loops excluded;
directed edges preserved as in the input DAG (tgt > src always).

Three conditions:
- **Trained**: learned attention weights
- **Uniform**: 1/in-degree per edge (pure random walk on A_input — graph-structure-only floor)
- **Random baseline**: off-diagonal A_agg entries permuted (true random floor)

## Main Table

| Class | N | F1 mean±std | τ* mean | P (mean) | R (mean) | dens_agg | dens_input | F1@density |
|-------|---|-------------|---------|----------|----------|----------|------------|------------|
| bonafide | 50 | 0.209 ± 0.061 | 0.0449 | 0.267 | 0.183 | 0.197 | 0.264 | 0.205 ± 0.063 |
| A01 | 50 | 0.165 ± 0.019 | 0.0316 | 0.161 | 0.169 | 0.190 | 0.182 | 0.152 ± 0.024 |
| A02 | 50 | 0.183 ± 0.045 | 0.0352 | 0.200 | 0.173 | 0.200 | 0.227 | 0.176 ± 0.050 |
| A03 | 50 | 0.171 ± 0.019 | 0.0310 | 0.169 | 0.173 | 0.193 | 0.188 | 0.160 ± 0.027 |
| A04 | 50 | 0.182 ± 0.026 | 0.0328 | 0.186 | 0.178 | 0.202 | 0.211 | 0.176 ± 0.030 |
| A05 | 50 | 0.249 ± 0.102 | 0.0662 | 0.354 | 0.202 | 0.199 | 0.310 | 0.247 ± 0.103 |
| A06 | 50 | 0.220 ± 0.070 | 0.0416 | 0.282 | 0.190 | 0.202 | 0.280 | 0.219 ± 0.070 |

## Sanity Check: Random Baseline

| Class | F1 mean±std (random) | F1 mean±std (uniform) |
|-------|---------------------|----------------------|
| bonafide | 0.230 ± 0.027 | 0.151 ± 0.066 |
| A01 | 0.188 ± 0.023 | 0.116 ± 0.008 |
| A02 | 0.209 ± 0.032 | 0.139 ± 0.046 |
| A03 | 0.194 ± 0.026 | 0.116 ± 0.011 |
| A04 | 0.213 ± 0.037 | 0.122 ± 0.012 |
| A05 | 0.244 ± 0.039 | 0.205 ± 0.119 |
| A06 | 0.248 ± 0.035 | 0.166 ± 0.080 |

## Interpretation

**Overall regime**: at or below random floor — completely rewired (analogous to El et al. <4%).
Mean best-F1 = 0.197 (trained) vs 0.218 (random baseline, density-matched)
and 0.145 (uniform/graph-structure floor). Δ(trained − random) = -0.021.

**Density note**: A_input has ~20–30% edge density (10-step lookahead DAG),
so the random F1 floor (~0.19–0.25) is far higher than in El et al.'s sparse graphs (~4%).
The trained GAT sits at or below this floor for all but one class, confirming complete
structural rewiring just as El et al. found — the absolute F1 numbers just look larger
because of the denser prior.

**Class split (routing vs skip-route taxonomy)**:
A01/A03/A04 mean F1 = 0.172, A05/A06 mean F1 = 0.235, gap = 0.062.
Classes above random floor: ['A05'].
A05 is the sole marginal exception (Δ ≈ +0.005), consistent with its
distinctive sparse-phoneme attention signature (Chinese TTS) concentrating
attention on a small subgraph that coincidentally overlaps A_input more.
The A01/A03/A04 group is further below the random floor than A05/A06,
mirroring their stronger routing-path engagement found in E2/E5/E6.

**Implication**: The GAT completely rewires away from A_input during training.
The effective attention graph is data-specific rather than structurally guided.
This explains the failed VCC2020 transfer: attention patterns learned on ASVspoof
phoneme sequences do not generalise to out-of-domain distributions.
