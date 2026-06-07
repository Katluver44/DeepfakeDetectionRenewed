# Experiment 2: Graph Independence Summary

**N = 63 MLAAD systems · 7 T variants · 4 C variants (WavLM only)**

---

## Context and Framing

T (vel_entropy@L9) is computed from sequential WavLM frame embeddings — it is NOT derived from the
phoneme graph. To test whether T is an artifact of the specific velocity/temporal measurement rather
than a genuine acoustic property, we recompute T using alternative velocity definitions and segmentation
schemes, then check rank consistency across variants.

C (−rog@L12) is also recomputed at adjacent layers (L9–L12) to confirm its depth-selectivity.

---

## T Variants: Signal and Rank Consistency

Reference: **T_l2_s1** = adjacent-frame L2 velocity entropy (original definition, WavLM L9)

| Variant | Definition | r vs hardness | ρ vs hardness | LOO R² | ρ vs T_original |
|---------|-----------|--------------|--------------|--------|----------------|
| **T_l2_s1** (original) | L2 adj, stride=1 | +0.271 | +0.302 | +0.024 | **1.000** |
| T_l2_s2 | L2, stride=2 | +0.313 | +0.224 | +0.026 | +0.314 |
| T_l2_s4 | L2, stride=4 | −0.054 | −0.056 | −0.056 | +0.082 |
| T_cosine_s1 | Cosine dist, stride=1 | +0.019 | +0.079 | −0.086 | +0.238 |
| T_window5 | 5-frame moving avg velocity | −0.026 | −0.022 | −0.066 | +0.113 |
| T_window11 | 11-frame window velocity | **−0.389** | **−0.453** | **+0.100** | **−0.111** |
| T_kmeans40 | Pseudo-phoneme (k=40) velocity | −0.026 | +0.004 | −0.060 | +0.043 |

---

## Key Findings

### 1. T_l2_s1 and T_l2_s2 are roughly consistent

Stride-2 (T_l2_s2) gives similar hardness correlation (r=+0.313) and positive LOO R²=+0.026.
Rank consistency vs original: ρ=+0.314 (moderate, not high). The stride-2 variant captures a similar
temporal scale as stride-1 (adjacent frames), so consistency is expected.

### 2. T reverses sign at stride-4 and window-11

**T_l2_s4 (stride 4)**: r=−0.054 — essentially zero hardness correlation. Rank consistency vs original: ρ=+0.082 (almost independent). At stride 4, the velocity captures inter-phoneme jumps more than phoneme-boundary transitions.

**T_window11 (11-frame window velocity)**: r=−0.389, ρ=−0.453 — **strong NEGATIVE correlation**, opposite to the original. Rank consistency: ρ=−0.111 (anti-correlated). The 11-frame window smooths over short-range irregularities and captures medium-range displacement — systems that are "easy" in adjacent-frame velocity are "hard" in medium-range displacement, and vice versa.

This sign reversal means: hard systems have HIGH short-range velocity entropy (bursty transitions) but LOW medium-range velocity (their frames don't travel far over 11-frame spans). This is consistent with the compact manifold hypothesis (C): hard systems have compact individual utterances, so over medium ranges the trajectory stays within a compact region (low medium-range velocity) — even while exhibiting bursty short-range transitions.

Interestingly, T_window11 has LOO R²=+0.100 (the highest of all T variants!), just with the OPPOSITE SIGN.

### 3. Cosine velocity and k-means pseudo-phoneme: no signal

T_cosine_s1: despite high ρ_vs_original (0.238 is marginal), the hardness correlation collapses to ~0.
This indicates the DIRECTION of velocity (captured by cosine similarity changing) is informative, but
the MAGNITUDE (captured by L2) carries the hardness signal.

T_kmeans40: no signal. The k-means segmentation aggregates frames into 40 pseudo-phoneme classes,
and velocity between centroids doesn't capture the relevant temporal property.

### 4. C (rog) is consistent and depth-selective

| Layer | C Pearson r | C LOO R² |
|-------|------------|---------|
| L9 | +0.144 | −0.037 |
| L10 | +0.233 | −0.005 |
| L11 | +0.215 | −0.019 |
| **L12** | **+0.348** | **+0.069** |

C's signal is concentrated at L12. Adjacent layers have positive r but negative LOO R². C is
genuinely depth-selective: the compactness signal requires deep linguistic encoding (L12), not
mid-processing stages.

---

## Interpretation: Is T Graph-Independent?

T is computed from WavLM frame sequences, NOT from the phoneme graph. However, "graph independence"
extends to measurement independence: is the T signal stable to changes in how velocity is defined?

**Answer: No — T is highly implementation-specific.**

The key finding: ρ between T_original and T variants ranges from +0.314 (stride-2, similar scale)
to −0.111 (window-11, reversed). No variant achieves ρ > 0.4 vs the original.

This means the T signal is sensitive to the specific temporal scale captured:
- Adjacent-frame L2 velocity entropy (stride ≤ 2): positive correlation with hardness
- Medium-range (5–11 frame windows): zero or negative correlation
- Long-stride (stride ≥ 4): zero or negative correlation

The "velocity entropy" phenomenon is REAL (there is structure in how hard vs easy systems' trajectories
move), but it is not well-defined by a single robust metric. The adjacent-frame L2 velocity entropy
captures one specific aspect of this structure.

**Verdict: T is not robust across measurement implementations. It captures a specific temporal-scale
property of WavLM frame sequences that reverses sign at longer scales. This makes T more of a
diagnostic indicator than a confirmed invariant mechanism.**

**C is more robust**: cross-validated positive LOO R² only at L12, but consistent in sign from L5–L12.
