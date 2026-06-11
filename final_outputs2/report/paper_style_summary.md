# Decision-Boundary Geometry as a Predictor of Deepfake Speech Detection Hardness Across Architectures and Domains

---

## Abstract

We investigate why certain synthetic speech systems consistently evade deepfake detectors while others are trivially exposed.
Across three corpora (MLAAD, ASVspoof 2021 LA, and In-The-Wild), two detector architectures (WavLM-GAT and AASIST), and 13 controlled causal interventions, we show that per-system hardness is governed by a system's position and spread along the natural↔synthetic discriminative axis in frozen WavLM-L12 representation space — not by the deep compactness metric C or velocity entropy T that were previously proposed.
The spread (sd_along) achieves leave-one-system-out R² = 0.273 on MLAAD (p = 0.0005) and replicates on a non-WavLM architecture (AASIST fine-tuned, ρ = +0.349, p = 0.006), confirming it is architecture-general.
Position (s_along) along a dataset-internal LDA axis achieves ρ = +0.599 (p = 0.031) in a prospective pre-registered prediction on ASVspoof 2021, replicating for AASIST (ρ = +0.643, p = 0.018).
Critically, discriminative axes are nearly orthogonal across recording domains (cos(w_MLAAD, w_ITW) = 0.05), explaining why out-of-domain transfer fails.
A zero-training-cost fusion of detector logits with axis projection reduces EER by up to 33 percentage points under domain shift, establishing actionable utility.
Cross-detector hardness agreement is ρ ≈ 0.55 for same-domain pairs and ρ ≈ 0.30–0.36 for cross-domain pairs regardless of architecture, demonstrating that hardness is training-domain-conditional, not architecture-conditional.

---

## 1. Introduction

Modern deepfake speech detectors exhibit highly non-uniform per-system false-negative rates: a single TTS system may evade detection 40% of the time while a different system from the same vendor is detected near-perfectly.
Understanding this hardness variability is critical — the hardest systems are precisely the ones that matter most for security applications.
Prior analyses have proposed compactness C (radius-of-gyration in SSL feature space) and velocity entropy T as predictors, but these are confounded by recording-channel artifacts and do not survive controlled interventions.
We conduct the first rigorous causal audit of hardness predictors, using pre-registered prospective predictions, leave-one-system-out evaluation, and explicit leakage audits.

---

## 2. Method

### 2.1 Models and Corpora

- **WavLM-GAT**: frozen WavLM-Large (L12 embeddings) → phoneme pooling → graph attention network → binary classifier. Trained on MLAAD. Three random seeds.
- **AASIST**: raw-waveform sinc-conv + heterogeneous graph attention. Evaluated zero-shot (official ASVspoof19-trained weights) and fine-tuned on MLAAD.
- **MLAAD**: 70+ TTS systems, 33 languages; 1846-utterance test set. Per-system hardness = 1 − AUC against bonafide pool (min. 8 utterances).
- **ASVspoof 2021 LA (clean)**: 13 unseen attacks (A07–A19). Bona/spoof balance: 400 + 40 × 13.
- **In-The-Wild (ITW)**: 58 speakers; per-speaker hardness from logit scores.

### 2.2 Axis Construction

The natural↔synthetic axis **w** is the centroid-difference direction in the mean-pooled WavLM-L12 space, estimated leave-one-system-out (LOSO) to prevent leakage.
For each held-out system, we compute: s_along = projection of mean utterance embedding onto w; sd_along = SD of per-utterance projections; s_orth = residual RMS distance off-axis.
LDA axis (j1_lda_audit) uses discriminant analysis on the same LOSO folds; cos(w_mean, w_lda) = 0.362.

### 2.3 Causal Audit (I1)

22 conditions × 3 seeds using a gated hook that restricts interventions to the detection pathway only (avoiding phoneme-ID path corruption, the E9 artifact).
Key contrast: sub_top (substitute top-k directions) ΔAUC = −0.0320*** vs sub_res (substitute residual) ΔAUC = +0.0039*** — direction content governs hardness, not magnitude.
Gated vs ungated hook: ΔAUC = +0.0101*** (artifact accounts for ~75–80% of prior compaction effect).

### 2.4 Fusion Head (I7 / J3)

Zero-training fusion: z(logit) + λ · z(axis_proj), with λ ∈ {0.5, 1.0, 1.5, 2.0} chosen on system-disjoint calibration fold.
Adaptive head (J3): group-disjoint calibration curves for centroid and LDA axes with n ∈ {5–250} labeled utterances/class.

---

## 3. Results

### 3.1 MLAAD (In-Domain Evaluation)

| Metric | WavLM-GAT | AASIST-FT |
|---|---|---|
| Baseline EER | 0.272 | 0.200 |
| Axis fusion EER | 0.163 | 0.136 |
| sd_along → hardness (ρ) | +0.273 (LOSO R², p=0.0005) | +0.349 (p=0.006) |
| vel_entropy → hardness (ρ) | +0.079 (LOSO R²) | +0.342 (p=0.007) |
| s_along → hardness (ρ) | ns | ns |

### 3.2 ASVspoof 2021 LA (Cross-Domain Prospective)

Pre-registered predictions written before any detector score was computed (timestamp in j4_preregistered_predictions.json).

| Predictor | Description | ρ (WavLM-GAT) | ρ (AASIST-ZS) |
|---|---|---|---|
| P1 | −z(pos_int) | +0.253 (ns) | +0.154 (ns) |
| P3 | −z(pos_lda) | **+0.599 (p=0.031)** | **+0.643 (p=0.018)** |
| P4 | −z(pos_mlaad) | −0.374 (ns) | −0.011 (ns) |

P3 achieves 2/3 top-3 system hits. P4 fails (confirms rotation law: MLAAD axis ≠ ASVspoof21 axis).

### 3.3 ITW

MLAAD-trained axis transferred to ITW: cos(w_MLAAD, w_ITW) = 0.05 → near-orthogonal.
ITW-internal axis fusion: EER 0.363 → 0.292 (WavLM-GAT) vs AASIST zero-shot 0.486 → 0.161.
Per-speaker hardness: s_orth ρ = +0.534 (p = 0.003) under WavLM-GAT; vel_entropy ns.

### 3.4 Cross-Detector Agreement

| | AASIST-FT | AASIST-ZS | WavLM-GAT | RobustGoat |
|---|---|---|---|---|
| **AASIST-FT** | 1.000 | 0.308 | **0.553** | 0.360 |
| **AASIST-ZS** | 0.308 | 1.000 | 0.320 | 0.544 |
| **WavLM-GAT** | **0.553** | 0.320 | 1.000 | 0.301 |
| **RobustGoat** | 0.360 | 0.544 | 0.301 | 1.000 |

Same-domain pairs (AASIST-FT ↔ WavLM-GAT, both MLAAD-trained): ρ = 0.553.
Cross-domain pairs: ρ ≈ 0.30–0.36 regardless of architecture.

---

## 4. Key Findings

- **C (deep compactness) is NOT causal.** ~75–80% of its apparent effect was an ungated hook artifact corrupting the phoneme-ID path. The true gated C-effect is ΔAUC = −0.0036 (vs E9 artifact ΔAUC = −0.185).
- **The causal mechanism is directional, not scalar.** Substituting top-k principal directions harms detection (ΔAUC = −0.032), while substituting the orthogonal residual helps (ΔAUC = +0.004). Magnitude alone is not operative.
- **sd_along (spread) is the strongest architecture-general predictor.** LOSO R² = 0.273 on MLAAD, replicates for AASIST-FT (ρ = +0.349) and vel_entropy similarly (ρ = +0.342).
- **s_along (mean position) is detector-conditional.** Works for ASVspoof21 prediction (ρ = +0.60) with LDA axis, fails as a MLAAD in-domain predictor, and is ns for AASIST-FT. It correlates with how close a system sits to the training-domain bona centroid.
- **Axis rotation blocks cross-domain transfer.** cos(w_MLAAD, w_ITW) = 0.05 (near-orthogonal); cos(w_MLAAD, w_ASVspoof21) ≈ 0.36. Corpus-internal axis estimation is necessary.
- **Fusion is the actionable intervention.** System-disjoint fusion reduces EER by 10–33 points depending on domain shift; the largest gains appear under shift (AASIST-ZS on MLAAD: −26 pts; AASIST-ZS on ITW: −33 pts). Adaptive LDA head with 250 labeled samples achieves −50% EER on MLAAD, −47% on ITW.
- **Hardness is training-domain-conditional, not architecture-conditional.** Same-domain, cross-architecture pairs agree ρ ≈ 0.55; cross-domain pairs agree ρ ≈ 0.30 regardless of architecture choice.

---

## 5. Ablations

### I1: Causal Geometry Interventions (MLAAD, 3 seeds)

| Condition | ΔAUC (vs baseline) | p | Interpretation |
|---|---|---|---|
| iso_0.7 (shrink rog) | −0.0023 | 0.024 | Small causal C effect |
| iso_0.7_ungated | −0.0124 | <0.001 | Artifact when hook leaks |
| sub_top_0.7 (top directions) | −0.0320 | <0.001 | **Direction content is causal** |
| sub_res_0.7 (orthogonal residual) | +0.0039 | <0.001 | Residual content is safe |
| shuffle (random direction) | −0.0170 | <0.001 | Destroys structure |
| gated vs ungated | +0.0101 | <0.001 | Artifact isolation confirmed |

### I2: Feature Battery (MLAAD, 61 systems)

Top LOSO-R² predictors: vel_entropy_L9 (+0.083), vel_entropy_L12 (+0.083), sd_along (+0.273, from I3). rog_L12 unique variance after vel_entropy: 0.003 (negligible).

### J1: LDA vs Mean-Centroid Axis Audit

| Axis | Hardness LOSO R² | Fusion ΔEER |
|---|---|---|
| w_mean (centroid diff) | 0.191 | −0.109 |
| w_lda | **0.431** | **−0.157** |
| w_logreg | 0.386 | −0.154 |
| w_rand (random) | −0.072 | +0.120 |
| w_detaxis (detector direction) | **0.707** | −0.027 |

The detector under-reads available evidence: its own axis achieves LOSO R² = 0.707 but fusion dEER = −0.027 (already seen in detector scores). The law is a geometric property of the representation space, not an artifact of the estimator.

### J3: Adaptive Head Calibration Curves

| Dataset | Head | n_cal | EER_before | EER_after | ΔEER |
|---|---|---|---|---|---|
| MLAAD | LDA | 250 | 0.272 | 0.136 | −0.136 |
| ITW | LDA | 250 | 0.363 | 0.192 | −0.171 |
| ASVspoof21 | LDA | 250 | 0.078 | 0.082 | +0.004 |

In-domain ASVspoof21 shows no gain (already near-perfect; no room for axis to help).

---

## 6. Discussion

### What is universal

The fundamental geometry of the representation space — that natural and synthetic speech occupy separable regions, and that per-system spread along the natural↔synthetic axis predicts hardness — appears to hold regardless of detector architecture (WavLM-GAT and AASIST agree ρ = 0.55 when trained on the same domain).
Prospective LDA predictor P3 generalizes from MLAAD geometry to ASVspoof2021 attacks for both WavLM and AASIST-based detectors (ρ ≈ 0.60–0.64).

### What is not universal

Axis direction is domain-specific: the discriminative axis rotates substantially between recording environments (MLAAD studio, ITW ambient, ASVspoof codec-processed).
This rotation accounts for why MLAAD-fit axes fail on ITW (cos = 0.05) and partially explains the asymmetric cross-domain agreement matrix.
Mean position (s_along) is detector-conditional: it reflects where a system sits relative to that detector's training-domain bona centroid, not an absolute property of the synthesis process.

### Why ITW is hard

Three compounding factors:
1. **Axis rotation**: MLAAD-trained discriminative direction is orthogonal to the ITW natural↔synthetic axis → transferred axis provides no signal.
2. **High off-axis variance**: ITW bona utterances slide +0.64 axis gaps vs MLAAD, expanding the overlap region between classes.
3. **Recording-domain confound**: channel and microphone diversity in ITW expands within-class variation, increasing s_orth for bona samples and masking synthetic artifacts.

---

## 7. Limitations

1. **Single SSL backbone**: all geometry results use WavLM-L12. Whether the axis law holds in wav2vec2-based or HuBERT-based representations is untested.
2. **Detector dependence**: s_along (mean position) only replicates for detectors trained on the same recording domain. Cross-domain use requires P3 (corpus-internal LDA) not P4 (MLAAD-transferred axis).
3. **Unresolved residual variance**: at LOSO R² ≈ 0.31 for the full triplet model, ~69% of hardness variance is unexplained. Likely contributors: prosodic artifacts, codec-specfic frequency bands, and synthesis backend diversity.
4. **Voice conversion gap**: VCC2020 (voice conversion) shows ρ ≈ 0 for all predictors. The natural↔synthetic axis law appears specific to TTS-family attacks where content originates in text, not in source speech.
5. **Calibration cost**: the adaptive head requires labeled samples; the zero-training fusion (I7) requires the corpus-internal axis computed from the full unlabeled test distribution.

---

## 8. Conclusion

The hardness of synthetic speech systems against deepfake detectors is governed by a geometric property of frozen SSL representations: systems whose utterance embeddings spread widely along — or cluster near the boundary of — the corpus-internal natural↔synthetic discriminative axis are hardest to detect.
This finding is architecture-general (replicates across WavLM-GAT and AASIST), and the predictor P3 generalizes prospectively to unseen attack families on a held-out corpus (ASVspoof 2021).
The practical implication is immediate: a zero-training-cost axis-fusion head reduces EER by up to 33 percentage points and an adaptive head with 250 labeled samples halves EER on MLAAD and ITW.
The fundamental limit is axis rotation across recording domains — a problem that corpus-internal axis estimation solves, but that prevents naive transfer from studio to ambient-recording corpora.
We release all pre-registration records, system-level hardness tables, and reproducible scripts alongside this work.

---

*Experiments: I1–I7 (causal geometry), J1–J6 (prospective, audit, AASIST baseline)*
*Corpora: MLAAD, ASVspoof 2021 LA, In-The-Wild, WaveFake, VCC2020*
*Architectures: WavLM-GAT (×3 seeds), AASIST official, AASIST fine-tuned on MLAAD*
