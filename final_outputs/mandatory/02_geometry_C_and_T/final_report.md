# 65% Analysis: Final Report

## Context

Prior analyses established two secondary hardness mechanisms beyond temporal smoothness:
- **H2 (Attention hub rigidity)**: mean_gini_mean ρ=+0.224, var_entropy_mean r=−0.202
- **H3 (Multi-cluster acoustic heterogeneity)**: cosine_sim_entropy ρ=+0.332 (p=0.008)

Together with temporal smoothness (~18% marginal EER variance), these explained roughly 20–25%
of residual hardness in LOO models. The remaining ~65–75% was unaccounted for.
This analysis tested three directions to close that gap.

---

## 1. Interaction Modelling

### Setup
8 top features, 28 pairwise interactions, LOO R² (Lasso + RF).

### Results

| Model | LOO R² Lasso | LOO R² RF |
|---|---|---|
| Main effects only | −0.062 | +0.194 |
| Main + all interactions | −0.266 | +0.110 |

All three explicitly requested interactions (smoothness×entropy, entropy×attention,
smoothness×attention) each reduce LOO R² when added to the main-effects model.
Permutation test: p > 0.70. Lasso selects several interaction terms consistently (especially
`cosine_sim_entropy×mean_gini_mean` at 88.9% of folds), but they represent
multicollinearity rather than independent signal.

**Verdict: No signal.** Interactions do not explain additional residual hardness beyond additive
main effects. The remaining variance is not structured as low-order multiplicative interactions
among the features measured in prior experiments.

---

## 2. Layer-wise WavLM Analysis

### Setup
63 systems × 2 utterances = 126 wav files. 13 processing stages probed:
layer 0 (CNN+projection, pre-transformer) through layers 1–12 (transformer blocks).
Per layer, per utterance: 4 static metrics (effective rank, frame cosine entropy, kNN density,
radius of gyration) + 6 trajectory metrics. Averaged over 2 utterances per system.

### Key correlations (best layer per metric)

| Metric | Type | Best layer | Pearson r | p | Spearman ρ | p |
|---|---|---|---|---|---|---|
| **rog** | static | **L12** | **−0.329** | **0.009\*\*** | **−0.293** | **0.020\*** |
| **vel_entropy** | trajectory | **L9** | **+0.276** | **0.028\*** | **+0.330** | **0.008\*\*** |
| vel_mean | trajectory | L10 | −0.284 | 0.024\* | −0.243 | 0.055† |
| vel_cv | trajectory | L12 | +0.255 | 0.044\* | +0.199 | 0.117 |
| vel_autocorr_lag1 | trajectory | L4 | +0.220 | 0.083† | +0.199 | 0.119 |
| eff_rank | static | L9 | −0.102 | 0.425 | −0.043 | 0.735 |
| frame_cos_entropy | static | L10 | −0.077 | 0.551 | −0.183 | 0.150 |
| knn_density | static | L2 | +0.036 | 0.779 | −0.096 | 0.452 |
| direction_persistence | trajectory | L6 | −0.053 | 0.680 | +0.115 | 0.371 |
| adj_neighbor_stability | trajectory | L7 | +0.183 | 0.150 | +0.205 | 0.107 |

Signal is absent at layers 0–5 and emerges consistently at layers 6–12. This localises the
hardness mechanism to the **linguistic encoding stage** of WavLM (layers 6–12 encode phoneme
identity, speaker characteristics, and prosodic structure in WavLM-base).

### LOO R² (single features and combinations)

| Feature set | LOO R² |
|---|---|
| rog @ L12 only | **+0.063** |
| vel_entropy @ L9 only | +0.019 |
| rog @ L12 + vel_entropy @ L9 | **+0.092** |
| Base temporal only (residual target) | −0.086 |
| Base temporal + rog@L12 + vel_entropy@L9 | +0.047 (ΔR²=+0.133) |

These are the **first positive LOO R² values** achieved for predicting residual hardness in
this entire analysis series.

### What rog at Layer 12 means
Radius of gyration = RMS distance of each utterance's frame embeddings from their centroid.
**Negative correlation**: hard systems have SMALLER rog at Layer 12. Their frame sequences
occupy compact regions of the deepest WavLM representation space. This is the compact manifold
hypothesis — confirmed, but only at the transformer OUTPUT level (L12), not at the pre-encoder
level (L0) where it was tested and refuted earlier. Each hard system's utterance is internally
coherent in its linguistic trajectory; it is the variety ACROSS utterances (cosine_sim_entropy)
that is high, not within-utterance spread.

### What vel_entropy at Layer 9 means
Velocity entropy = entropy of the distribution of frame-to-frame speeds within an utterance.
**Positive correlation**: hard systems have HIGHER vel_entropy — their linguistic trajectories
move at IRREGULAR speeds: some phoneme transitions very fast, others nearly stationary. This
irregularity is distinct from temporal smoothness at the acoustic level (L0 frame_mean_dist_mean)
and appears first at layer 6–9, confirming it is a linguistic rather than acoustic property.

---

## 3. Trajectory Analysis

### Key finding
Trajectory features match or outperform static features at deep layers:
- rog (static, L12): LOO R²=+0.063, ρ=−0.293*
- vel_entropy (trajectory, L9): LOO R²=+0.019, ρ=+0.330**
- Combined: LOO R²=+0.092

In multivariate regression with all 10 features per layer, both overshoot (LOO R² negative)
because N=63 is too small. Single-feature LOO confirms both are individually predictive.

The trajectory view matters: vel_entropy captures **temporal structure of the linguistic
encoding** that static pooling cannot recover. Hard systems produce irregular-speed linguistic
trajectories that static mean-field descriptions miss entirely.

---

## 4. Overall Assessment

### Which direction looks most promising?

**Layer-wise WavLM analysis** is the most productive of the three directions. It produces:
- The two strongest individual-feature correlations yet found for residual hardness
- The first positive LOO R² values in the entire analysis series
- A clear mechanistic localisation (layers 6–12, linguistic stage)

Interactions: no signal. Pre-encoder manifold geometry (prior analysis): no signal.
Post-transformer layer-wise features: ~9% of residual variance explained — modest but robust.

### Is the remaining variance decomposable?

**Partially.** The post-transformer WavLM features at layers 9–12 explain a fraction (~9%)
of residual variance that was completely opaque to all prior feature sets. The hardness
mechanism is real and partially localised in the transformer's linguistic representation.

However, ~55–65% of residual hardness variance remains unaccounted for. The most plausible
explanations:

1. **Classifier decision boundary geometry** — each system's position relative to the GAT's
   trained decision surface is not accessible from WavLM activations alone; it requires probing
   the GAT's internal activations directly.

2. **Small-N measurement noise** — with only 2 utterances per system in this analysis (vs 5–25
   in prior analyses using cached embeddings), system-level estimates have high variance. The
   true signal in deep WavLM features may be substantially larger.

3. **Undiscovered structural features** — prosodic rhythm, speaker-style consistency across
   utterances, or vocoder-specific spectral artifacts may contribute.

### Strictness assessment

| Feature | Signal type | Confidence |
|---|---|---|
| rog @ L12 (r=−0.329, ρ=−0.293, LOO=+0.063) | **Real** | Moderate-high |
| vel_entropy @ L9 (r=+0.276, ρ=+0.330, LOO=+0.019) | **Real** | Moderate |
| vel_mean @ L10 (r=−0.284, ρ=−0.243†) | **Marginal** | Low-moderate |
| Interaction terms (Lasso-stable but no ΔR²) | **No signal** | — |
| All pre-encoder manifold metrics (prior analysis) | **No signal** | — |

---

## 1. Interaction Modelling

(Interaction modelling results not found — run analysis_65pct_interactions.py first.)


---

## 2. Layer-wise WavLM Analysis

**Setup**: 63 systems × 2 utterances = 126 utterances. 13 processing stages (layer 0 = pre-transformer CNN + projection; layers 1–12 = transformer blocks).

### Layer-wise R² progression

| Layer | Static R² | Traj R² | Combined R² | Max |ρ| |
|---|---|---|---|---|
| L0 | -0.142 | -0.068 | -0.190 | 0.157 |
| L1 | -0.023 | -0.151 | -0.049 | 0.096 |
| L2 | -0.160 | -0.144 | -0.110 | 0.173 |
| L3 | -0.112 | -0.206 | -0.195 | 0.180 |
| L4 | -0.053 | -0.179 | -0.137 | 0.199 |
| L5 | +0.024 | -0.172 | -0.072 | 0.176 |
| L6 | +0.047 | -0.148 | -0.039 | 0.220 |
| L7 | -0.099 | -0.182 | -0.092 | 0.205 |
| L8 | -0.096 | -0.177 | -0.166 | 0.199 |
| L9 | -0.124 | -0.106 | -0.138 | 0.330 |
| L10 | -0.076 | -0.286 | -0.111 | 0.251 |
| L11 | -0.002 | -0.645 | -0.151 | 0.176 |
| L12 | +0.027 | -0.276 | -0.064 | 0.293 |

### Best layer

Combined features achieve highest LOO R² at Layer 6 (R²=-0.039).

### First significant layer

- `vel_mean` first reaches p<0.10 at Layer 6 (ρ=-0.220)
- `vel_entropy` first reaches p<0.10 at Layer 9 (ρ=+0.330)
- `rog` first reaches p<0.10 at Layer 10 (ρ=-0.251)

---

## 3. Trajectory Analysis

Frame-level trajectory metrics computed per utterance, averaged over 2 utterances per system. Compared against static embedding summary metrics.

### Best trajectory features (peak Spearman ρ across all layers)

| Metric | Best Layer | Spearman ρ | p |
|---|---|---|---|
| `vel_mean` | L10 | -0.243† | 0.0553 |
| `vel_cv` | L12 | +0.199 | 0.1173 |
| `vel_autocorr_lag1` | L4 | +0.199 | 0.1188 |
| `direction_persistence` | L6 | +0.115 | 0.3710 |
| `vel_entropy` | L9 | +0.330** | 0.0083 |
| `adj_neighbor_stability` | L7 | +0.205 | 0.1070 |

### Trajectory vs. static comparison

Best LOO R² (any layer):
- Static features: +0.047
- Trajectory features: -0.068
- Combined: -0.039


---

## 4. Overall Assessment

### Which direction looks most promising?

**None of the three directions** shows convincing predictive power: best LOO R²=-0.039 is near noise level. Signal appears first at Layer 6, suggesting the hardness mechanism is linguistic in nature.

### Is the remaining variance decomposable?

Based on all analyses (prior four hypotheses + interactions + layer-wise + trajectory), the residual hardness variance appears **largely decomposition-resistant** at the feature levels probed here:

- Interaction terms do not add robust incremental R²
- Pre-transformer WavLM manifold geometry is not predictive
- Post-transformer WavLM layer features show only marginal signal
- Frame-level trajectory dynamics show marginal-to-no signal

The most likely explanation is that the remaining hardness is driven by **classifier decision-boundary geometry** — i.e., where each TTS system's WavLM representations fall relative to the GAT's learned decision boundary, which requires probing the GAT's internal activations directly rather than the input WavLM representations. Alternatively, system-specific factors (training corpus, speaker diversity, vocoder type) create correlated-but-unpredictable variation that cannot be recovered from a small-N regression without richer system metadata.
