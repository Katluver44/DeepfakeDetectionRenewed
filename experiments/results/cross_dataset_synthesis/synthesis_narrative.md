# Cross-Dataset Synthesis: Mechanism Analysis of GAT Attention in Deepfake Detection

*ASVspoof 2019 LA (English, 6 TTS/VC systems) × MLAAD (multilingual, 63 TTS/VC systems)*

---

## 1. What Generalizes: Dataset-Invariant Mechanism Claims

### 1.1 All Representational Differences Are GAT-Localized

The most robust finding across both datasets is that the encoder (WavLM + phoneme pooling)
is frozen between the baseline and robust training runs, and consequently all differences
between the *goat* and *robust_goat* checkpoints arise exclusively in the GAT and BiLSTM
layers. This is confirmed **byte-identically** on both datasets: a linear probe at the
`pre_gat` extraction point produces identical EER for both checkpoints (Δ = 0.0000 on
ASVspoof, Δ = 0.0000 on MLAAD), with confidence intervals collapsing to zero.

This `gat_localized` interpretation is the most mechanistically informative result: whatever
the robustness-training procedure achieves, it does so entirely through the attention and
recurrent layers, leaving the front-end spectral representation untouched.

### 1.2 Robustness Training Produces Focused Attention (Entropy Collapse)

On both datasets, the top-identified attention heads in the robust model exhibit lower entropy
(more focused routing) than their counterparts in the baseline model. On ASVspoof, head h0
collapses from entropy 2.03 → 1.41 (Δ = -0.63) and h4
from 2.04 → 1.59 (Δ = -0.44). On MLAAD, h2 collapses from
1.58 → 1.41 (Δ = -0.17) and h4 from 1.62 → 1.49
(Δ = -0.14). The effect is approximately 3× larger in absolute terms on ASVspoof,
likely reflecting the larger KL range of that dataset (3.2× vs 1.6×).

In both cases the top-ranked head shows larger entropy reduction than the second-ranked head,
a within-checkpoint ordering that holds across systems and attack families.

### 1.3 Top-2 Head Ablation in the Robust Model Disrupts Performance Above Chance

Ablating the two highest-KL heads from the robust model damages pooled EER well above the
random-control baseline on both datasets:

| Dataset | ΔEER (ablate top-2) | ΔEER (random ctrl) | Gap |
|---------|--------------------|--------------------|-----|
| ASVspoof (robust) | +0.043 | +0.003 | +0.040 |
| MLAAD (robust) | +0.057 | -0.015 | +0.072 |

The effect is moderate-to-strong on both datasets (magnitude ratio 0.56).
These are not interchangeable heads; ablating randomly chosen pairs does not reproduce the
effect.

---

## 2. What Is Dataset-Specific

### 2.1 Head Identity Does Not Cross-Checkpoint Generalize on ASVspoof (But Does on MLAAD)

One of the starkest cross-dataset divergences concerns whether the head indices identified
on the robust model transfer to the baseline model.

On **ASVspoof**, applying the robust model's target heads {h0, h4} to the goat checkpoint
yields ΔEER = +0.0000 (empirical null-distribution percentile: 57.1% — chance
level). The ASVspoof goat has its own top-2 heads {h1, h0}, partially overlapping but with
h1 replacing h4. Ablating the goat's *own* top-2 gives only ΔEER = -0.020, also
modest.

On **MLAAD**, applying the robust model's target heads {h2, h4} to the goat checkpoint
yields ΔEER = +0.0905 — the strongest ablation effect observed, ranking at the
100th percentile of an exhaustive 14-pair null distribution. Notably, h2 is the
top-ranked head in the MLAAD goat's *own* head discovery (rank 1 with mean KL = 0.091),
while h4 is rank 3 (KL = 0.089). The heads are effectively consistent across training
regimes in MLAAD.

This suggests the ASVspoof and MLAAD models solve the deepfake detection problem through
*different* head specializations — likely because the underlying phoneme-level discriminative
signal has a different structure across the two datasets' attack systems. Head identity is
initialization-dependent; what transfers across checkpoints is the *mechanism* (entropy
collapse, GAT-localized representation change) rather than the *specific head indices*.

### 2.2 Probe Ordering Reverses Between Datasets

On ASVspoof, the linear probe ordering is `pre_gat (EER=0.080) < post_wavlm (0.101) < post_gat (0.114)`:
the encoder's phoneme pooling *improves* linear separability over raw WavLM features, and
the GAT stage reduces it. On MLAAD, the ordering is
`post_wavlm (0.152) << pre_gat (0.398) ≈ post_gat (0.403)`:
raw WavLM features are highly discriminative (EER = 0.15, AUROC = 0.92), and the phoneme
pooling step *dramatically* degrades linear separability.

Both datasets retain the `gat_localized` designation (pre_gat is byte-identical between
checkpoints), but the representation dynamics entering the GAT differ fundamentally. This
likely reflects the compositional difference of the attack spaces: ASVspoof's 6 systems
(circa 2019) are well-characterized; MLAAD's 63 systems (2023–2025) include diverse
modern TTS architectures that may require non-linear combination of phoneme-level cues.

### 2.3 A05 Anomaly (ASVspoof-Specific)

The voice-conversion system A05 consistently exhibits higher KL divergence than TTS systems
in ASVspoof, is more accurately classified by post-GAT probes (F1 = 0.89 vs 0.70 average),
and drives the per-system variance in ablation deltas. MLAAD's equivalent voice-conversion
system (RVC) does not stand out in the per-attack statistics in the same way, likely because
the pool of 63 systems averages over many VC variants.

---

## 3. What MLAAD Adds: Cross-Language Evaluation of the Abstraction-as-Generalization Claim

The MLAAD dataset enables a direct test of whether the learned phoneme-level abstraction
generalizes *beyond the training language*. Findings:

### 3.1 GAT Stage Is the Bottleneck for Cross-Language Transfer

The cross-language generalization gap (EER increase moving from English in-distribution to
German/other spoof + English bonafide) is largest at the post-GAT extraction point:

| Extraction point | Cross-language EER gap |
|-----------------|------------------------|
| post_wavlm | +0.043 |
| pre_gat | +0.031 |
| post_gat | +0.061 (robust) |

The WavLM front-end generalizes well (gap = 0.043); the phoneme pooling adds little
language-specific information (gap = 0.031); the GAT and BiLSTM layers are where the
language-specific overfitting concentrates (+0.018 additional gap over post_wavlm).

### 3.2 Target Heads Encode Language-Specific Rather Than Language-Invariant Features

Ablating heads h2 and h4 on the robust model *improves* cross-language EER by 0.159 (absolute),
reducing the in/out-of-distribution gap from 0.317 to 0.101 — a reduction of 0.217. Ablating
h2 alone accounts for most of this improvement (cross-lang EER drops from 0.677 to 0.618).
The random-pair control gives a much smaller gap reduction (−0.055), confirming specificity.

This is paradoxical: the same heads that are *critical* for in-distribution performance
(ΔEER = +0.057 when ablated) are *harmful* for cross-language transfer. The attention patterns
themselves are stable across languages (cosine similarity h2=0.953, h4=0.948),
suggesting the language overfitting originates in how the BiLSTM *uses* the routed features,
not in the attention selection itself. Section 3.3 decomposes this improvement into its
language-specific sources.

### 3.3 Mechanistic Decomposition: Improvement Is Entirely from Language-Mismatch Errors

To identify *which error type* the ablation corrects, the cross-language EER was re-computed
on two non-overlapping slices of the test set:

| Slice | Baseline EER | Ablated EER | ΔEER | Fraction of pooled Δ |
|-------|-------------|-------------|------|----------------------|
| Pooled (all bonafide vs DE spoof) | 0.677 | 0.518 | −0.159 | — |
| EN bonafide vs DE spoof (language mismatch) | 0.718 | 0.555 | −0.163 | **103%** |
| DE bonafide vs DE spoof (same language) | 0.351 | 0.372 | +0.021 | −13% |

The improvement is almost entirely concentrated in the *cross-language bonafide* slice: the
model's ability to distinguish English bonafide speech from German TTS improves by 0.163
when h2+h4 are ablated. The within-language slice (German bonafide vs German spoof)
*degrades* marginally (+0.021), confirming that h2+h4 do encode some genuine German
spoof-detection signal that is sacrificed.

The mechanism is made explicit by the raw score distributions (higher score = more
spoof-like):

| Sample group | Baseline mean | Ablated mean | Δ |
|---|---|---|---|
| Bonafide EN | 0.992 | 0.647 | −0.345 |
| Bonafide DE | 0.923 | 0.558 | −0.365 |
| Spoof DE | 0.969 | 0.646 | −0.323 |

In the baseline, English bonafide speech is assigned a higher spoof score (0.992) than
German TTS itself (0.969). The model trained on German in-distribution data treats English
phoneme patterns as *more deviant from learned bonafide* than actual German TTS artifacts.
After ablating h2+h4, English bonafide and German spoof both receive ≈0.65, confirming
that these heads are the direct source of the language-identity signal that was suppressing
cross-language generalization.

### 3.4 Implication for the Abstraction-as-Generalization Hypothesis

The initial hypothesis was that phoneme-level graph attention learns language-invariant
representations by abstracting over surface acoustic features. The MLAAD results offer a
more nuanced picture: the phoneme abstraction *partially* generalizes (post_wavlm gap is
small), but the learned attention routing amplifies language-specific structure rather than
suppressing it. Robustness training sharpens this specialization (larger entropy collapse,
more critical heads) *at the cost of cross-language generalization* on the robust model.

The baseline goat model shows a smaller cross-language gap (0.280 vs robust's 0.317), and
ablating h2+h4 from it gives ΔEER = +0.090 in-distribution but only +0.041 cross-language —
a more benign trade-off. This suggests robustness training over-specializes the top attention
heads for the English TTS attack space.

---

## 5. Reframed Central Claim

The strongest claim supportable by both datasets together:

> **The critical attention heads in the GAT-based deepfake detector encode attack-discriminative
> phoneme-routing patterns that are (a) localized entirely to the GAT+BiLSTM stage
> (encoder-frozen confirmation), (b) sharpened by robustness training (entropy collapse),
> and (c) necessary but not sufficient for cross-domain generalization — they are responsible
> for in-distribution discrimination but simultaneously encode dataset-specific structure
> that limits transfer to out-of-distribution attacks (MLAAD cross-language) and alternative
> training regimes (ASVspoof cross-checkpoint head non-transfer).**

A weaker but more universally supported version:

> **Robustness training in this architecture operates exclusively through the graph attention
> and recurrent layers, producing focused routing patterns in the top-KL heads. These patterns
> are functionally critical (100th-percentile null distribution on MLAAD, strong ΔEER gap on
> ASVspoof's robust model), but head identity is initialization-dependent: the same mechanism
> manifests at different head indices across datasets and training runs.**

---

*Generated by experiments/scripts/cross_dataset_synthesis.py*
*Sources: ASVspoof (gat_l0_attention_followups/) + MLAAD (mlaad/)*
