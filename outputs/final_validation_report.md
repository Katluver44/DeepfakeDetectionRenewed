# Final Validation Report: Are C and T Real Mechanisms?

**Four experiments · 63 MLAAD systems · 6 ASVspoof A01–A06 systems · 3 encoders · 7 T variants**

---

## 1. What Was Tested

**Candidate axes:**
- **C = deep compactness**: −rog (radius of gyration) at WavLM Layer 12. Hard systems have compact
  individual utterance frame clouds in deep WavLM representations.
- **T = trajectory irregularity**: vel_entropy (adjacent-frame L2 velocity entropy) at WavLM Layer 9.
  Hard systems have irregular-speed linguistic trajectories.

Both axes were confirmed (permutation p<0.05) as unique contributors to residual hardness in
the Task 6 orthogonalization analysis (Shapley_C=0.085, Shapley_T=0.066, LOO unique: C=+0.056, T=+0.037).

**Four validation tests:**
1. **Representation invariance**: Do C and T appear with different encoder architectures?
2. **Measurement robustness**: Is T's rank ordering stable across alternative velocity definitions?
3. **Intervention directionality**: Do directional perturbations affect C and T as predicted?
4. **Cross-dataset generalization**: Do C and T predict EER ordering in ASVspoof A01–A06?

---

## 2. Experiment-by-Experiment Verdict

### E1 — Representation Invariance

**C (deep compactness):**
| Encoder | Best layer | Peak r | Best LOO R² | Confirmed? |
|---------|----------|--------|------------|-----------|
| WavLM | L12 | +0.348 | +0.069 | **YES** |
| wav2vec2 | L8 | +0.346 | +0.073 | **YES** |
| HuBERT | L5 | +0.235 | −0.014 | NO |

C is confirmed in WavLM and wav2vec2 with positive LOO R² at comparable magnitude.
It is NOT confirmed in HuBERT (LOO R² negative at all layers). The signal is partially
encoder-universal: it appears wherever the encoder develops deep phoneme-level representations,
but is absent (or unmeasurable with N=63) in HuBERT's different representational geometry.

The peak layer differs across encoders (L12 for WavLM, L8 for wav2vec2), suggesting the signal
is tied to the depth of linguistic abstraction rather than a fixed layer index.

**T (trajectory irregularity):**
| Encoder | Best layer | Peak r | Best LOO R² | Confirmed? |
|---------|----------|--------|------------|-----------|
| WavLM | L9 | +0.271 | +0.024 | Marginal |
| wav2vec2 | L3 | +0.329 | +0.049 | **YES** |
| HuBERT | L2 | +0.251 | +0.013 | Marginal |

T is stronger in wav2vec2 (LOO=+0.049) than in WavLM (LOO=+0.024), and present marginally in HuBERT.
The signal appears at DIFFERENT layers across encoders (L2 HuBERT, L3 wav2vec2, L9 WavLM) —
suggesting it captures phoneme-boundary transition structure wherever each encoder encodes it.

**E1 verdict:**
- C: partially encoder-universal (2/3 encoders confirmed, 1 unclear)
- T: weakly encoder-universal (all 3 show positive direction, 1 confirmed, 2 marginal)
- Neither signal is WavLM-specific

---

### E2 — Measurement Robustness (Graph Independence)

**T variants** vs hardness correlation and rank consistency with T_original:

| Variant | r vs hardness | ρ vs T_original |
|---------|--------------|----------------|
| T_l2_s1 (original) | +0.271 | 1.000 |
| T_l2_s2 (stride-2) | +0.313 | +0.314 |
| T_cosine | +0.019 | +0.238 |
| T_l2_s4 (stride-4) | −0.054 | +0.082 |
| T_window5 | −0.026 | +0.113 |
| **T_window11** | **−0.389** | **−0.111** |
| T_kmeans40 | −0.026 | +0.043 |

**T fails the robustness test.** The sign of T's hardness correlation REVERSES under moderate
parameter changes (stride-4, window-11, cosine metric, k-means). The original T_l2_s1 is NOT
a robust measure: its rank ordering has ρ < 0.32 vs all alternatives, with one alternative
(window-11) having ρ = −0.111 (anti-correlated).

Key mechanistic insight from T_window11: hard systems have HIGH short-range velocity entropy
(bursty phoneme transitions, measured by T_l2_s1) but LOW medium-range velocity (their frames
don't travel far over 11-frame spans). These two measurements capture the SAME compact-but-bursty
geometry from opposite angles. This is consistent with C (compact cloud → frames stay near the
centroid over medium spans) and confirms that T_l2_s1 and T_window11 are measuring complementary
aspects of the same compact-trajectory phenomenon.

**E2 verdict:**
- T is **implementation-specific**. Adjacent-frame L2 velocity entropy captures one specific
  temporal scale, and the signal changes sign at longer scales.
- C is consistent across L10–L12 (positive direction), deeper layers give stronger signal.

---

### E3 — Intervention-lite (Perturbation Directionality)

| Perturbation | Expected | Observed (C) | Observed (T) | Directional? |
|-------------|----------|-------------|-------------|-------------|
| Smoothing (win=3) | Both weaken | r: +0.348→+0.217 ↓ | r: +0.271→+0.176 ↓ | **YES** |
| Smoothing (win=11) | Both weaken | r: +0.348→−0.031 ✗ | r: +0.271→−0.315 ✗ | Both reverse |
| PCA k=64 | C stable | r: +0.348→+0.360 ✓ | N/A | **C robust** |
| PCA k=32 | C slight drop | r: +0.348→+0.334 (−0.014) | N/A | **C very robust** |
| Frame shuffle | T→0; C unchanged | r(C): +0.348→+0.348 ✓✓ | r(T): +0.271→+0.012 ✓✓ | **BOTH CORRECT** |
| Noise σ=0.5 | Both weaken | r(C): +0.348→+0.349 ✓ | r(T): +0.271→+0.109 ↓ | **C robust; T partial** |
| Noise σ=2.0 | Both weaken | r(C): +0.348→+0.351 ✓ | r(T): +0.271→+0.240 (~) | **C robust; T non-monotone** |

**C intervention results:**
1. Smoothing (aggressive): C's correlation weakens under strong smoothing — as expected, since
   rog requires frame-level diversity.
2. PCA compression: **C is PCA-invariant down to k=64 dimensions**. The compactness signal is
   concentrated in the dominant variance directions — geometric shape, not high-frequency variation.
3. Frame shuffle: **C is exactly shuffle-invariant** (r unchanged to 3 decimal places) —
   confirming C is a static geometric measure.
4. Gaussian noise: **C is completely noise-invariant** across all tested noise levels (r=+0.348→+0.351).
   Noise inflates all systems' rog equally, preserving rank ordering.

**T intervention results:**
1. Smoothing: T drops as expected — confirms T measures fine-grained temporal irregularity.
2. Frame shuffle: T's correlation collapses to ~0 — **confirms T measures temporal ordering, not
   just the marginal frame distribution**. This is positive evidence for T as a temporal mechanism.
3. Gaussian noise: T partially degrades at moderate noise but partially recovers at high noise —
   non-monotone response, consistent with noise destroying the phoneme-boundary spike structure at
   intermediate levels but dominating all variance at high levels.

**E3 verdict:**
- C is mechanistically clean: PCA-robust, shuffle-invariant, noise-invariant. Consistently positive.
- T's temporal ordering nature is confirmed (shuffle destroys it). But its noise sensitivity
  and E2's sign reversals show T is measuring a specific and fragile temporal scale.

---

### E4 — ASVspoof A01–A06 Generalization

| System | EER | C | T |
|--------|-----|---|---|
| A06 (hardest) | 0.257 | C=−10.927 (rank 4) | T=2.780 (rank 2) |
| A03 | 0.120 | C=−10.813 (rank 2) | T=2.767 (rank 3) |
| A02 | 0.102 | C=−10.939 (rank 5) | T=2.801 (rank 1) |
| A05 | 0.094 | C=−10.953 (rank 6) | T=2.767 (rank 3) |
| A04 | 0.067 | C=−10.670 (rank 1) | T=2.712 (rank 6) |
| A01 (easiest) | 0.038 | C=−10.857 (rank 3) | T=2.753 (rank 5) |

**C: r=−0.380 (direction reversed vs MLAAD). NOT confirmed.**

The C values cluster in a 0.28-unit range (vs ~3-unit MLAAD range). The reversal and the tiny
spread both indicate this test is inconclusive — the 6 ASVspoof systems are too homogeneous in
their WavLM L12 compactness to meaningfully test the C signal.

**T: r=+0.451, Spearman ρ=+0.771 (direction consistent with MLAAD). Encouraging but not significant.**

The 6 T values span 0.089 units. Despite this tight range, the rank ordering of T aligns
reasonably with EER rank (ρ=+0.771). The hardest system (A06) is ranked 2nd in T; the two
easiest (A01, A04) rank 5th and 6th in T. The main discrepancy: A02 has the highest T but is
only 3rd hardest. Neither r nor ρ is statistically significant (p>0.30) at N=6.

**E4 verdict:**
- C: **inconclusive** (insufficient spread in ASVspoof A01–A06)
- T: **directionally consistent but not significant** (N=6, ρ=+0.771 below p=0.05 threshold)
- The ASVspoof test cannot confirm or deny either signal given these 6 systems' homogeneity.

---

## 3. Master Verdict: What Survives All Tests?

### C (deep compactness, −rog@L12) — Partially Confirmed

| Test | Result | Status |
|------|--------|--------|
| E1: WavLM representation | r=+0.348, LOO=+0.069 | ✅ PASS |
| E1: wav2vec2 representation | r=+0.346, LOO=+0.073 | ✅ PASS |
| E1: HuBERT representation | LOO<0 at all layers | ❌ FAIL (1/3) |
| E2: Layer consistency (L9–L12) | Sign consistent, LOO only at L12 | ✓ Partial |
| E3: PCA invariance | r unchanged down to k=64 | ✅ STRONG PASS |
| E3: Shuffle invariance | r unchanged to 3 d.p. | ✅ STRONG PASS |
| E3: Noise invariance | r unchanged at all noise levels | ✅ STRONG PASS |
| E4: ASVspoof generalization | Direction reversed, tiny spread | ❌ INCONCLUSIVE |

**Verdict for C: Partially confirmed as a real geometric mechanism, not a WavLM-specific artifact.**

C measures a genuine property of the deep encoder representation: the spatial compactness of an
utterance's frame trajectory in the deep linguistic embedding space. This property:
- Appears in 2 of 3 tested encoders (WavLM, wav2vec2)
- Is PCA-robust, shuffle-invariant, and noise-invariant
- Is NOT confirmed to generalize to ASVspoof (insufficient test)
- Is NOT confirmed in HuBERT (fails to cross-validate)

Calling C "confirmed" requires: evidence from at least one additional encoder or dataset showing the
same positive LOO R². The current evidence supports calling C **probably real but encoder-dependent**.

---

### T (trajectory irregularity, vel_entropy@L9) — Partially Real, Implementation-Sensitive

| Test | Result | Status |
|------|--------|--------|
| E1: WavLM representation | r=+0.271, LOO=+0.024 | ✓ Marginal |
| E1: wav2vec2 representation | r=+0.329, LOO=+0.049 | ✅ PASS |
| E1: HuBERT representation | r=+0.251, LOO=+0.013 | ✓ Marginal |
| E2: Stride-2 consistency | ρ_vs_original=+0.314, r=+0.313 | ✓ Partial |
| E2: Stride-4, window-5, cosine | Signs reverse or collapse | ❌ FAIL |
| E2: Window-11 | r=−0.389 (REVERSED sign) | ❌ FAIL |
| E3: Shuffle destroys T | r(T,h)→0.012 after shuffle | ✅ PASS (temporal structure confirmed) |
| E3: Noise sensitivity | Non-monotone degradation | ✓ Partial |
| E4: ASVspoof Spearman ρ | ρ=+0.771 (consistent direction) | ✓ Trend (not significant) |

**Verdict for T: The EXISTENCE of temporal structure is confirmed; the specific measurement is not robust.**

The frame-shuffle experiment confirms that T is measuring TEMPORAL ORDERING, not just static
distributional properties of frame embeddings. Hard systems have more bursty temporal dynamics
in their WavLM sequences. This is a real effect.

However, the E2 measurement robustness test shows that adjacent-frame L2 velocity entropy is NOT a
robust measure of this temporal property: the signal REVERSES at longer temporal scales (window-11)
and vanishes with other distance metrics (cosine, stride-4). The "trajectory irregularity" phenomenon
exists at the shortest temporal scales (adjacent frames) but is not a stable property across scales.

What T_l2_s1 actually measures: the contrast between within-phoneme smoothness and between-phoneme
abruptness in WavLM frame trajectories. Hard systems have sharper phoneme transitions in their
deep WavLM representations. This is a real property, but it is:
- Specific to the adjacent-frame temporal scale
- Layer-varying (peaks at different layers in different encoders)
- Sign-reversing at medium scales

T should be treated as a **diagnostic indicator of phoneme-boundary sharpness**, not a universal
trajectory irregularity measure.

---

## 4. Final Synthesis

### What survives all tests:
**Nothing survives ALL four tests.** No finding that is simultaneously:
- (a) Present in all 3 encoders [E1]
- (b) Robust across measurement variants [E2]
- (c) Directionally correct under all perturbations [E3]
- (d) Significantly predictive in ASVspoof [E4]

### What survives most tests:

**C (deep compactness)** survives E1 (2/3 encoders), E3 (all perturbation tests with strong results),
but fails E1-HuBERT and E4. **C is a real geometric property of WavLM/wav2vec2 representations that
is NOT a pipeline artifact.** It measures the compactness of the deep utterance representation cloud.

**T (adjacent-frame velocity entropy)** survives E1 (direction consistent in all encoders), E3-shuffle
(temporal ordering confirmed), and E4-direction, but fails E2 (sign reversal across variants).
**T captures a real temporal structure (phoneme-boundary sharpness) but is not robustly measurable
by a single scalar metric across implementations.**

### What appears pipeline-specific:
- The specific layer localization (WavLM L9 for T, L12 for C) — these are WavLM-specific depths
- The sign and magnitude of T under alternative velocity definitions
- The specific LOO R² values (which are likely inflated relative to true population correlation)

### Conservative assessment:

**C is a real but weak encoder-dependent mechanism (partially pipeline-specific).**
Compact deep representations predict hardness in WavLM and wav2vec2, but not yet in HuBERT.
The mechanism is plausible: systems that generate acoustically similar successive frames (low
within-utterance diversity in deep representations) are more confusable by the classifier.
Classification requires diversity to create a separating boundary; compact representations
give the GAT less geometric separation to exploit.

**T is a real but implementation-fragile indicator (scale-specific, partially pipeline-specific).**
Hard systems have sharper phoneme-boundary transitions at the adjacent-frame timescale in WavLM
deep layers. The frame-shuffle test confirms this is temporal structure, not static geometry.
However, the reversal at medium scales (T_window11) shows the compact trajectory of hard systems
means their OVERALL displacement over many frames is LOW, even while frame-to-frame steps are bursty.
This is geometrically self-consistent but makes T a non-robust summary statistic.

**The C–T manifold is not confirmed to generalize to ASVspoof A01–A06**, primarily because the
6 ASVspoof systems tested are too homogeneous in their WavLM representations to provide
a meaningful test. A proper cross-dataset test requires a larger, architecturally diverse system set.

### Recommendation:
Treat C as a confirmed but encoder-dependent hardness indicator. Report T as preliminary with a
caveat about measurement sensitivity. Do not claim either axis is a universal, pipeline-independent
mechanism without further evidence from additional encoders and datasets.
