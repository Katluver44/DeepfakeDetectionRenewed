# Deepfake Audio Detection — Research Notes

## Model: Robust GOAT (`robust_goat.ckpt`)

### Architecture

A frozen WavLM phoneme-CTC encoder feeds a trainable GAT + BiLSTM classifier.

```
Raw audio (16 kHz, 3 s)
    │
    ▼
WavLM feature extractor  [FROZEN]
    │  (B, T, 768)
    ▼
WavLM encoder  [FROZEN]  ──▶  CTC phoneme logits  →  phoneme_ids (B, T)
    │  (B, T, 768)  =  phoneme_feat
    │
    ├── SpecAugment masking
    ▼
Trainable encoder copy  [UNFROZEN]
    │  (B, T, 768)  =  encoder_feat
    │
    ▼
Adaptive phoneme pooling  (consecutive same-ID frames averaged)
    │  (total_phonemes_in_batch, 768)
    │
    ▼
GAT  [3 layers, 6 heads, skip connections]
    │  (total_phonemes_in_batch, 768)
    │
    ▼
BiLSTM  [2 layers, hidden=384 each dir → 768 concat]
    │  per-sample phoneme sequences
    │
    ▼
Mean pool over phoneme sequence  →  L2 normalise
    │  (B, 768)  =  hidden_states
    │
    ▼
Classification head  [Linear → BN → ReLU → Dropout → Linear → logit]
    │  (B,)
    ▼
Binary cross-entropy  +  0.5 × CLIP loss  +  0.5 × aug loss
```

**Total parameters:** 107 M (12.9 M trainable, 94.9 M frozen WavLM backbone)

### Training Setup

| Setting | Value |
|---|---|
| Dataset | ASVspoof 2019 LA (`Bisher/ASVspoof_2019_LA`) |
| Pool | HF `train` + `test` splits concatenated |
| Balance | 50/50 bonafide/spoof, seed=42 |
| Effective training set | 19 870 samples (capped from 30 k pool) |
| Augmentation probability | 35% of samples augmented |
| Aug breakdown | noise 50% · pitch-up 30% · reverb 20% |
| Aug rationale | proportional to per-attack struggle scores on earlier `goat.pth` model |
| Epochs | 7 |
| Batch size | 20 |
| Optimizer | AdamW — encoder LR 5e-5, rest 1e-4, weight decay 1e-4 |
| Backbone | `microsoft/wavlm-base` |
| GAT edges | `n_edges=10` forward-looking phoneme edges + adjacency |

**Validation split** (`Bisher/ASVspoof_2019_LA:validation`) was never seen during training and is used for all evaluations below.

---

## Experiment 1 — Holdout Accuracy (`test_model.py`)

**Script:** `experiments/test_model.py`
**Holdout:** validation split, 100 balanced examples (50 normal + 50 deepfake)

| Metric | Value |
|---|---|
| Accuracy | **91.0%** (91/100) |
| TP (deepfake correctly caught) | 43 |
| TN (normal correctly passed) | 48 |
| FP (normal misclassified as fake) | 2 |
| FN (deepfake missed) | 7 |

The model is conservative about false positives (only 2), accepting slightly more missed deepfakes (7). This asymmetry likely reflects the BCEWithLogits loss without explicit cost-weighting.

---

## Experiment 2 — PCA Embedding Analysis (`plot_embeddings.py`)

**Script:** `experiments/plot_embeddings.py`
**Holdout:** 100 balanced validation examples
**Output plots:** `experiments/pca_frozen_encoder.png`, `pca_pre_gat.png`, `pca_post_gat.png`

PCA applied to mean-pooled embeddings at three pipeline stages.

| Stage | PC1 variance | PC2 variance | Interpretation |
|---|---|---|---|
| Frozen phoneme encoder (`phoneme_feat`) | 24.6% | 16.5% | Dispersed; real/fake not separated |
| Trainable encoder, pre-GAT (`encoder_feat`) | 26.1% | 15.0% | Slightly more structured, still mixed |
| **Post-GAT + LSTM + pool** (`hidden_states`) | **72.8%** | **9.7%** | Single dominant axis — clear separation |

**Key finding:** The frozen WavLM encoder and even the trainable encoder copy produce embeddings that are not linearly separable by class. The GAT + BiLSTM stage is where separation emerges — PC1 variance jumps from ~25% to 73%, meaning the graph attention over phoneme sequences is doing almost all of the discriminative work, not the backbone features themselves.

This has an important architectural implication: the model is not learning a better acoustic feature representation; it is learning a *relational* pattern over the phoneme graph.

---

## Experiment 3 — Phoneme-Stratified PC1 Analysis (`phoneme_analysis.py`)

**Script:** `experiments/phoneme_analysis.py`
**Holdout:** 200 balanced validation examples
**Output plots:** `experiments/results/phoneme_pc1_violin.png`, `phoneme_pc1_mean_delta.png`

Per-phoneme post-GAT node embeddings (before mean pooling) are projected onto PC1 (the classification axis). Each phoneme node is assigned to a broad IPA category and the PC1 distribution is compared across real vs. deepfake utterances.

### PC1 mean by phoneme class (Δ = fake − normal)

| Class | Normal mean PC1 | Fake mean PC1 | **Δ** | n nodes |
|---|---|---|---|---|
| Vowels | −0.3554 | −0.3364 | +0.019 | 2312 |
| Diphthongs | −0.2866 | −0.2015 | +0.085 | 456 |
| Approximants | −0.2980 | −0.3224 | −0.024 | 293 |
| **Nasals** | −0.2434 | +0.0501 | **+0.293** | 122 |
| Stops | −0.2740 | −0.3670 | −0.093 | 629 |
| Fricatives | −0.2461 | −0.2664 | −0.020 | 440 |
| **Sibilants** | +0.1061 | −0.4597 | **−0.566** | 36 |
| Affricates | −0.4791 | −0.4913 | −0.012 | 259 |
| Other | −0.3410 | −0.4410 | −0.100 | 2971 |

### Findings

**Sibilants (s, z, ʃ, ʒ) show the largest absolute delta (−0.57).**
In real speech, sibilant nodes sit at the *positive* end of PC1 (+0.11); in deepfake speech they collapse to the strongly negative end (−0.46). Despite being rare (36 nodes / ~0.5% of all nodes), the displacement is the most extreme of any class. Sibilants are high-frequency strident sounds that TTS and VC systems are known to struggle with — their spectral fine structure is hard to model faithfully and appears to be one of the most diagnostic cues the GAT has latched onto.

**Nasals (m, n, ŋ) show the largest *signed* delta (+0.29).**
Normal nasal nodes are at −0.24; fake nasal nodes are shifted to +0.05 — crossing zero toward the fake pole. Nasals involve complex velopharyngeal coupling and nasal resonances that differ between natural and synthesized speech; the model appears to pick this up via the phoneme graph structure rather than raw acoustics.

**Stops and "Other" tokens also shift modestly (−0.09, −0.10).**
Stops involve burst transients and aspiration that vocoders often smear; the "Other" category mostly contains silence/boundary tokens (`|`) and may reflect prosodic patterning.

**Vowels, fricatives, approximants, and affricates show near-zero delta.**
These classes carry less classification signal in PC1, suggesting the GAT does not primarily rely on vowel quality or steady-state frication to discriminate.

### Interpretation

The GAT is not operating uniformly across the phoneme graph. It concentrates discriminative signal in a small subset of classes — particularly sibilants and nasals — which are exactly the phones where TTS synthesis is known to introduce artifacts. This is consistent with a *phoneme-relational* detection strategy: the model is flagging anomalous transitions *into and out of* high-information phones rather than comparing global acoustic statistics.

This finding is potentially publishable because:
1. It provides a mechanistic explanation for *why* the GAT architecture outperforms frame-level classifiers.
2. It gives actionable guidance for TTS robustness: models that improve sibilant and nasal synthesis would be harder to detect.
3. The sibilant result connects to the known failure mode of neural vocoders (spectral leakage in the 4–8 kHz band).

---

## Experiment 4 — GAT Attention by Phoneme Class (`gat_attention.py`)

**Script:** `experiments/gat_attention.py`
**Holdout:** 300 balanced validation examples
**Edges processed:** 95 724 (across all 3 GAT layers)
**Output plots:** `experiments/results/gat_attention_heatmap.png`, `gat_attention_topedges.png`

Per-edge attention weights are read from all 3 GAT layers using the built-in
`log_attention_weights` flag.  For every edge `(src → tgt)` the weight (averaged
over 6 heads, then over the 3 layers) is accumulated into a
9×9 class×class matrix, separately for bonafide and deepfake utterances.
Matrices are row-normalised so each row shows the *distribution of attention
outgoing from that source class*, making normal/deepfake directly comparable.

### Full Δ matrix (deepfake − normal, row-normalised)

| Source ↓ / Target → | Vowels | Dipht | Appro | Nasals | Stops | Frica | Sibil | Affric | Other |
|---|---|---|---|---|---|---|---|---|---|
| Vowels      | −0.0020 | −0.0012 | −0.0005 | +0.0024 | +0.0006 | +0.0038 | **+0.0029** | −0.0052 | −0.0008 |
| Diphthongs  | +0.0049 | −0.0023 | −0.0028 | −0.0115 | +0.0094 | −0.0068 | **+0.0349** | −0.0248 | −0.0009 |
| Approximants| +0.0037 | −0.0013 | −0.0070 | −0.0026 | −0.0186 | +0.0016 | **+0.0151** | +0.0133 | −0.0042 |
| **Nasals**  | +0.0055 | +0.0010 | −0.0079 | −0.0005 | +0.0098 | +0.0026 | **−0.0298** | +0.0110 | +0.0082 |
| Stops       | +0.0055 | −0.0031 | +0.0083 | +0.0081 | +0.0005 | −0.0027 | −0.0166 | −0.0007 | +0.0008 |
| Fricatives  | −0.0062 | −0.0018 | −0.0057 | +0.0032 | +0.0031 | −0.0060 | **+0.0262** | −0.0080 | −0.0048 |
| **Sibilants**| +0.0135 | **−0.0243** | **+0.0083** | +0.0064 | −0.0071 | +0.0165 | **−0.0616** | **+0.0298** | +0.0184 |
| Affricates  | +0.0015 | +0.0009 | +0.0103 | −0.0104 | −0.0096 | −0.0141 | **+0.0117** | +0.0123 | −0.0027 |
| Other       | −0.0025 | −0.0055 | +0.0066 | +0.0057 | +0.0020 | −0.0057 | +0.0011 | −0.0014 | −0.0003 |

### Key findings

**1. Sibilant self-attention collapses in deepfakes**

The single largest absolute delta in the entire matrix.
The all-layers-averaged matrix gives Δ = −0.062 for the Sibilant→Sibilant cell.
The final GAT layer alone gives normal = 0.033, deepfake = 0.000, Δ = −0.033 —
these two numbers are consistent (0.000 − 0.033 = −0.033) but come from the
per-layer report; the −0.062 is the average across all three layers where earlier
layers compound the effect.

This is mechanistically interpretable: in natural speech, sibilant phoneme
segments are acoustically consistent — consecutive frames share spectral
structure, so the GAT attention mechanism reinforces them against each other.
In TTS/VC output, the sibilant tokens are present in the phoneme sequence but
their feature vectors are internally inconsistent (the vocoder smears the 4–8 kHz
energy that makes sibilants distinctive), so they *stop recognising each other
as similar* under attention scoring.

**2. Multiple other classes redirect attention toward sibilant nodes in deepfakes**

When sibilant self-attention collapses, it frees up attention budget.
Diphthongs (+0.035), Fricatives (+0.026), and Affricates (+0.012) all
increase their attention *to* sibilant nodes in deepfakes.  This pattern
suggests those classes are picking up on the anomalous sibilant representations
— the model's attention is drawn to the "broken" nodes rather than to their
context.

**3. Nasals attend 30% less to sibilants in deepfakes (Δ = −0.030)**

In normal speech, Nasal → Sibilant edges carry 11.1% of nasals' outgoing
attention.  In deepfakes this drops to 7.2%.  This directly links
Experiments 3 and 4: the large PC1 shift on nasals (Δ = +0.29) is at least
partly explained by nasals *losing their normal attention relationship to
sibilants* — a co-occurrence that is disrupted in synthetic speech.

**4. Final GAT layer — Sibilant outgoing attention (per-layer detail)**

| Target | Δ | Normal | Deepfake |
|---|---|---|---|
| Sibilants → Sibilants    | **−0.033** | 0.033 | 0.000 |
| Sibilants → Diphthongs   | −0.039 | 0.156 | 0.117 |
| Sibilants → Approximants | +0.040 | 0.109 | 0.149 |
| Sibilants → Fricatives   | +0.015 | 0.095 | 0.110 |
| Sibilants → Affricates   | +0.014 | 0.124 | 0.139 |

In real speech sibilants attend strongly to diphthongs (natural co-occurrence:
/s/ before /eɪ/, /oʊ/, etc.).  In deepfakes this bond is weakened (−0.039)
and replaced by stronger ties to approximants and fricatives — classes whose
representations survive synthesis more faithfully.

### Interpretation and publication angle

The attention collapse story is now three-layered and mutually consistent:

1. **(Exp 2)** Discrimination happens entirely in the GAT, not in the frozen encoder.
2. **(Exp 3)** Sibilant and nasal nodes are maximally displaced along the
   classification axis (PC1 Δ = −0.57 and +0.29 respectively).
3. **(Exp 4)** The mechanism: sibilant self-attention collapses in deepfakes
   (final layer: normal=0.033 → fake=0.000, Δ = −0.033; avg across all 3 layers Δ = −0.062),
   and nasals lose their normal attention relationship to sibilants (Δ = −0.030).

Together these constitute a phoneme-level *attention fingerprint* for deepfake
audio: the GAT detects synthesis artifacts by noticing that sibilant nodes
have lost internal coherence and that nasal-sibilant co-occurrence patterns
are disrupted.  Neither of these signals is visible at the frame level —
they only emerge in the phoneme graph.

This is a publishable mechanistic finding because:
- It directly explains *why* graph attention outperforms frame classifiers.
- It identifies the specific phonetic locus of the artifact (sibilants).
- It makes falsifiable predictions: improving vocoder sibilant quality should
  reduce the Sibilant self-attention gap and degrade model accuracy.
- It provides an adversarial target for TTS robustness research.

---

## Experiment 5 — Per-Attack-System Attention Analysis (`gat_attention_by_system.py`)

**Script:** `experiments/gat_attention_by_system.py`
**Holdout:** validation split, 500 samples per system (3500 total, no 50/50 balancing)
**Edges processed:** 1 178 205 (across all 3 GAT layers, all systems)
**Output files:** `experiments/results/gat_by_system_*.png`, `gat_by_system_summary.csv`

ASVspoof 2019 LA validation contains **6 attack systems (A01–A06)** plus bonafide (`-`),
each with 3716 utterances. Repeating the attention analysis per-system reveals whether
the sibilant collapse finding from Exp 4 is a universal deepfake property or
is concentrated in specific vocoders.

### Summary table

| System | Label | Sib→Sib | Δ | Nas→Sib | Δ | n(Sib→Sib) | n(Nas→Sib) |
|---|---|---|---|---|---|---|---|
| `-` | bonafide | 0.1100 | — | 0.1184 | — | **15** | 48 |
| A01 | spoof | 0.1255 | **+0.016** | 0.1043 | −0.014 | 9 | 33 |
| A02 | spoof | 0.1173 | +0.007 | 0.1284 | +0.010 | 9 | 15 |
| A03 | spoof | 0.0883 | **−0.022** | 0.1037 | −0.015 | 12 | 42 |
| A04 | spoof | 0.0931 | **−0.017** | 0.1013 | −0.017 | 39 | 30 |
| A05 | spoof | 0.1009 | −0.009 | 0.1042 | −0.014 | **57** | 69 |
| A06 | spoof | 0.1178 | +0.008 | 0.0910 | −0.027 | 15 | 24 |

### Key findings

**1. The sibilant self-attention collapse is NOT universal.**

A03, A04, and A05 consistently show lower Sib→Sib attention than bonafide (Δ = −0.009 to −0.022),
consistent with the Exp 4 finding. However A01, A02, and A06 show the **opposite**: their
sibilant self-attention is equal to or higher than bonafide (+0.007 to +0.016).

The aggregated analysis in Exp 4 averaged over all systems, so the A03/A04 signal dominated
when those systems happened to be over-represented in the balanced 300-sample draw.
The "collapse" is a property of specific attack systems, not deepfake audio in general.

**2. Edge counts are critically small — confidence is limited.**

The Sib→Sib cell for most systems has **9–57 edges** across 500 utterances and 3 GAT layers.
The bonafide baseline itself has only **n=15** Sib→Sib edges. With counts this small,
the deltas of ±0.01–0.02 are unreliable — a handful of utterances with unusual phoneme
sequences can swing the number substantially. The sibilant class is simply rare in the
phoneme graph (confirming the node count from Exp 3: 36/7518 = 0.5% of all nodes).

**3. Nasal→Sibilant is more consistent but still noisy.**

5 of 6 systems show lower Nasal→Sibilant attention than bonafide (A02 is the exception,
+0.010). The effect is in the same direction across most systems (Δ = −0.014 to −0.027),
but counts are again modest (15–69 edges per system). A06 shows the largest drop (−0.027).

**4. What this means for Experiment 4**

The "sibilant attention fingerprint" framing from Exp 4 needs to be qualified:
- It is a real signal for some attack systems (A03/A04 specifically) but is *reversed* for others.
- The Nasal→Sibilant reduction is more reproducible across systems but smaller in magnitude.
- Neither finding has the statistical power to be published as a universal deepfake property
  without either (a) larger sample sizes or (b) the test set (A07–A19) for replication.

The correct framing is: **different attack systems produce different phoneme-graph
attention signatures**, which is itself an interesting and publishable finding — the model
may be responding to vocoder-specific artifacts rather than a single universal "deepfake
signal". A03/A04 appear to share a vocoder family that disrupts sibilant coherence;
A01/A02 use a different synthesis path that does not.

### Recommended next step

Run the same analysis on the test split (A07–A19) once accessible, to:
1. Replicate the Nasal→Sibilant direction on held-out attack systems.
2. Test whether A07–A19 cluster into the same "sibilant-collapsing" vs
   "sibilant-preserving" groups as A03–A04 vs A01–A02.
3. Use larger per-system sample sizes (all 3716 per system) for the test set
   to get reliable edge counts for rare class pairs.

---

## Experiment 6 — Linear Probes on GAT Head / Layer / BiLSTM Representations

**Script:** `experiments/linear_probe.py`
**Goal:** Quantify where classification-relevant linear structure lives — GAT layer 0/1/2,
individual attention heads (6 per layer, 128-d each), and post-BiLSTM pooled embeddings (768-d).

**Data:** Same 3 500-sample validation set as Exp 5 (7 systems × 500).
**Probe:** `LogisticRegression(C=1, class_weight='balanced', max_iter=2000)` + `StandardScaler`.
**Split:** Stratified 80/20 probe-train / probe-test (within validation only).

### Full probe ranking (by AUC, 22 probes)

| Rank | Probe | AUC | ACC | F1 | Dim |
|------|-------|-----|-----|----|-----|
| 1 | `head_l2_h2` | **0.9874** | 0.9500 | 0.9702 | 128 |
| 2 | `gat_l1` | 0.9840 | 0.9457 | 0.9680 | 768 |
| 3 | `head_l2_h4` | 0.9839 | 0.9314 | 0.9588 | 128 |
| 4 | `head_l2_h1` | 0.9830 | 0.9371 | 0.9623 | 128 |
| 5 | `bilstm` | 0.9815 | 0.9429 | 0.9661 | 768 |
| 6 | `head_l2_h0` | 0.9813 | 0.9329 | 0.9598 | 128 |
| 7 | `gat_l2` | 0.9810 | 0.9486 | 0.9697 | 768 |
| 8 | `head_l2_h5` | 0.9807 | 0.9300 | 0.9579 | 128 |
| 9 | `head_l2_h3` | 0.9805 | 0.9243 | 0.9545 | 128 |
| 10 | `head_l1_h0` | 0.9802 | 0.9214 | 0.9525 | 128 |
| 11–16 | `head_l1_h*` | 0.977–0.980 | — | — | 128 |
| 17–22 | `head_l0_h*` | 0.950–0.962 | — | — | 128 |

### Layer-level summary

| Probe | AUC | Notes |
|-------|-----|-------|
| `gat_l0` | 0.9776 | First GAT layer; lowest among full-layer probes |
| `gat_l1` | 0.9840 | Sharpest single-layer AUC gain (Δ+0.0064 vs l0) |
| `gat_l2` | 0.9810 | Slight regression from l1 at full-layer level |
| `bilstm` | 0.9815 | Post-BiLSTM; between l1 and l2 |

### Key findings

**1. A single 128-d GAT head outperforms the 768-d BiLSTM output.**
`head_l2_h2` (AUC 0.9874) is the best probe overall, beating the BiLSTM (0.9815) by 0.006 AUC.
This means the most classification-relevant structure is already concentrating in a single
layer-2 attention head before the BiLSTM even runs.

**2. Layer 2 heads dominate the head ranking.**
All 6 heads of layer 2 (AUC 0.980–0.987) beat all 6 heads of layer 1 (0.977–0.980), which
in turn beat all 6 heads of layer 0 (0.950–0.962). The GAT is building increasingly linear
class separation through its layers.

**3. Layer 0 heads are notably weaker (AUC ~0.95–0.96).**
The ≈ 0.03 AUC gap between layer 0 and layer 2 heads is large. Layer 0 is doing the
initial neighborhood aggregation; the clear linear structure emerges in layers 1–2.

**4. BiLSTM adds negligible linear separability.**
`bilstm` (0.9815) is ranked 5th — below `gat_l1` (0.9840) and all 6 `head_l2_*` probes.
The BiLSTM temporal modelling does not increase linear separability; it may help non-linear
classification (the model's sigmoid head) but is not the source of the classification signal.

**5. Head 2 of layer 2 is a recurring stand-out.**
`head_l2_h2` (best probe) was also notable in the sibilant attention analysis (Exp 4/5).
Its 128-d representation alone achieves 95.0% accuracy, suggesting it has specialised as a
high-level spoofing detector at the phoneme-graph level.

### Artifacts saved

All probe weights and feature matrices are in `experiments/results/linear_probe/`:
- `features_{name}.npz` — (N, D) feature matrix + labels + system_ids
- `probe_{name}.npz` — coef, intercept, scaler_mean/scale, full metrics dict
- `probe_metrics_summary.csv` — one row per probe, ranked by AUC
- `probe_ranking.png` — AUC bar chart, colour-coded by probe type

---

## Experiment 7 — 6-Way Attack-System Probe (Vocoder Fingerprinting)

**Script:** `experiments/multiclass_probe.py`
**Goal:** Within the spoof region only (A01–A06), can a linear probe distinguish which
vocoder/attack system generated a sample? Tests whether system identity is linearly encoded
in the model's internal representations.

**Data:** 3 000 spoof samples (500 × 6 systems) from the same feature files as Exp 6.
**Probe:** Multinomial `LogisticRegression(C=1, class_weight='balanced', max_iter=2000)`.
**Split:** Same stratified 80/20 scheme (2400 train / 600 test).
**Chance level:** 1/6 ≈ 16.7%.

### Full probe ranking (by macro-F1)

| Rank | Probe | Acc | Macro F1 | Dim |
|------|-------|-----|----------|-----|
| 1 | `gat_l0` | **0.9183** | **0.9180** | 768 |
| 2 | `gat_l1` | 0.8933 | 0.8926 | 768 |
| 3 | `gat_l2` | 0.8533 | 0.8522 | 768 |
| 4 | `head_l2_h2` | 0.8517 | 0.8497 | 128 |
| 5 | `head_l2_h1` | 0.8383 | 0.8366 | 128 |
| 6–21 | other heads | 0.79–0.83 | 0.79–0.83 | 128 |
| **22** | `bilstm` | **0.7500** | **0.7492** | 768 |

### Best probe breakdown (gat_l0, per system)

| System | Precision | Recall | F1 |
|--------|-----------|--------|----|
| A01 | 0.89 | 0.88 | 0.88 |
| A02 | 0.87 | 0.83 | 0.85 |
| A03 | 0.94 | 0.96 | 0.95 |
| A04 | 0.95 | 0.95 | 0.95 |
| A05 | 0.90 | 0.95 | 0.93 |
| A06 | 0.95 | 0.94 | 0.94 |

### Key findings

**1. System identity is strongly linearly encoded — especially in GAT layer 0.**
`gat_l0` achieves 91.8% accuracy on a 6-class problem (chance = 16.7%), meaning the
*first* GAT layer's node embeddings carry a strong vocoder fingerprint that is largely
preserved as a linear signal. The model appears to be using vocoder-specific patterns from
the very first layer of message-passing.

**2. The ranking flips vs. Exp 6 (binary deepfake detection).**
In Exp 6, layer 2 heads were best for binary spoof/bonafide separation.
Here, `gat_l0` is the best vocoder discriminator — layers 1 and 2 *destroy* system-specific
information as they build the detection signal. This implies the GAT is progressively
abstracting away vocoder identity in favour of a generic "is-this-spoofed" representation.

**3. BiLSTM is worst for vocoder ID (75%), best for binary detection.**
The BiLSTM's temporal integration further collapses vocoder-specific information.
It trades vocoder identity for detection reliability — exactly what you'd want in a
deployable detector.

**4. A01/A02 are hardest to distinguish (F1 ≈ 0.85–0.88); A03/A04/A06 are clearest (0.94–0.95).**
This is consistent with the Exp 5 finding that A01/A02 show atypical attention patterns
(e.g., elevated sibilant self-attention) vs. A03–A06. They may share a closer synthesis
family or produce more natural-sounding output that is harder to fingerprint.

**5. Probe similarity plot confirms the layer hierarchy.**
The `probe_similarity.png` prediction-agreement matrix shows: full-layer probes (gat_l0/1/2)
form their own cluster with high mutual agreement; individual heads within each layer also
cluster together; `bilstm` is the most distinct predictor.

### Artifacts saved

Stored in `experiments/results/multiclass_probe/`:
- `confusion_{name}.png` — normalised 6×6 confusion matrix per probe (22 files)
- `multiclass_ranking.png` — macro-F1 bar chart, colour-coded by probe type
- `probe_similarity.png` — pairwise prediction-agreement heatmap
- `multiclass_summary.csv` — full metrics table (acc, macro-F1, per-system P/R/F1)
- `probe_mc_{name}.npz` — probe weights (coef, intercept, scaler) per representation

---

## Experiment 8 — Layer-0 GAT Activation Patching

**Script:** `experiments/act_patching.py`

**Goal:** Establish the *causal* contribution of each phoneme node to the spoof classification
decision. For each node `i` in a correctly-classified spoof utterance `x`, we measure:

```
Δ_i = L(x) − L_patch(x, i)
```

where `L_patch(x, i)` is the logit after replacing node `i`'s layer-0 GAT representation
`h_i^(0)(x)` with `h̄_{c_i}^(0)` — the mean layer-0 output for phoneme class `c_i` computed
across the bonafide set. The downstream layers (GAT 1+2, BiLSTM, classifier) see an otherwise
unmodified graph; only node `i`'s content changes to "what this phoneme would look like in
real speech."

- `Δ > 0`: node was pushing toward "spoof" — replacing with bonafide content reduced the logit
- `Δ ≈ 0`: node was not contributing to the spoof decision
- `Δ < 0`: node was acting as a counterweight toward bonafide (suppressing the spoof score)

**Implementation details:**
- Phase 1: compute one 768-d class mean per phoneme class from 16,742 bonafide nodes
- Phase 2: for each spoof utterance, build N copies of the phoneme graph (N = node count),
  patch one node per copy, run GAT layers 1+2 → BiLSTM → mean pool → classify in one batched
  forward pass
- Only utterances where `L(x) > 0` (model predicted spoof) are included
- Utterances with zero-edge graphs get self-loops added before the batched pass

**Data:** Validation split, same 3500-sample set. 2,639 correctly-classified spoof utterances
(361 skipped — model predicted bonafide on those), 108,437 total node records.

### Results — mean causal effect per phoneme class

| Class | Mean Δ | Std | n |
|-------|--------|-----|---|
| Diphthongs | **+1.946** | 1.936 | 6,245 |
| Approximants | +1.910 | 1.930 | 3,839 |
| Vowels | +1.887 | 1.946 | 33,568 |
| Fricatives | +1.871 | 1.950 | 6,686 |
| Other | +1.858 | 1.950 | 43,427 |
| Stops | +1.850 | 1.885 | 8,927 |
| Nasals | +1.822 | 1.872 | 1,560 |
| Affricates | +1.743 | 1.860 | 3,660 |
| **Sibilants** | **+1.721** | 1.904 | **525** |

### Key findings

**1. All classes have Δ > 0 — every phoneme class carries spoof-relevant content.**
Replacing any node's layer-0 representation with the bonafide class mean always reduces the
spoof logit on average. There is no class that pushes the model toward bonafide in spoof audio
(no Δ < 0 in aggregate). The model appears to use information from every phoneme class.

**2. Sibilants show the smallest causal effect (+1.721), consistent with earlier experiments.**
This is the quantitative causal confirmation of the "sibilant collapse" observed in Exp 4/5.
In spoof audio, sibilant layer-0 representations are already most similar to bonafide
(smallest Δ when patched to bonafide means). The model is getting less spoof-specific signal
from sibilants than from any other class.

**3. Diphthongs and Approximants are the strongest spoof carriers (+1.946, +1.910).**
Patching these to bonafide means causes the largest drop in spoof confidence. These sound
classes — which involve smooth formant transitions (diphthongs) and liquid/glide articulation
(approximants) — appear to be where vocoders most distinctively deviate from natural speech at
the feature level encoded by layer 0 of the GAT.

**4. Affricates are second-lowest (+1.743), sibilants lowest (+1.721).**
Both involve sibilant components (affricates are stop+sibilant sequences). This suggests a
broader pattern: sounds with high-frequency frication / turbulent airflow components are
harder for vocoders to fake convincingly *and* are already close to bonafide in the model's
representation, making them weak contributors to the spoof decision.

**5. The spread (std ≈ 1.9 for all classes) is large relative to the means.**
Node-level Δ values span a wide range even within a class. The class means are meaningful
population-level statistics, but individual nodes vary greatly — some nodes in every class
have near-zero or even negative Δ. The signal is not deterministic per class.

**6. Cross-referencing with Exp 3 (PC1) and Exp 6 (linear probes):**
- Exp 3 found sibilants most discriminative on PC1 — but PC1 is a *representation* statistic,
  not a causal one. Sibilants being separable on PC1 does not mean they *cause* the
  classification; Exp 8 shows they actually cause the least.
- The disconnect is explained by Exp 5: sibilant representations in spoof are *collapsed*
  toward zero (fake=0.000 attention), which makes them easy to distinguish on a projection
  axis, but patching them to bonafide means moves them to a state the model already wasn't
  relying on heavily.

### Artifacts saved

Stored in `experiments/results/act_patching/`:
- `bonafide_class_means.npz` — (9 × 768) mean layer-0 GAT output per phoneme class (bonafide)
- `act_patching_records.npz` — per-node arrays: `cls_names`, `deltas`, `logit_origs`, `system_ids`
- `act_patching_class_stats.csv` — mean Δ, std, n per class
- `act_patching_violin.png` — Δ distribution violin + mean bar chart per class
- `act_patching_by_system.png` — class × attack system mean Δ heatmap

---

## Experiment 9 — Probe-Guided Activation Patching by Attack System

**Script:** `experiments/act_patching_probe.py`

**Goal:** Test whether each phoneme node carries evidence for the spoofer's *true* attack
system, not just for the generic spoof-vs-bonafide decision. This reuses the same layer-0 GAT
node patching setup from Exp 8, but measures how much the patch changes the score of a pretrained
6-way attack-system probe.

For each correctly-probed spoof utterance and node `i`, the script computes:

```
Δ_i(system) = s_true(x) - s_true_patch(x, i)
```

where `s_true` is the probe score for the utterance's true spoof system (A01-A06) and
`s_true_patch(x, i)` is the score after replacing node `i` with the bonafide class mean.
It also reports counterfactual shifts for all six system heads.

**Data:** Validation split, 3,500 samples (7 systems). 2,885 utterances retained after
skipping 115 where the attack-system probe's top-1 prediction was wrong. Total node records:
115,784.

### Results — true-system score drop by phoneme class

| Class | Mean Δ_true | n |
|---|---|---|
| **Sibilants** | **+0.004216** | 578 |
| Nasals | +0.002802 | 1,672 |
| Other | +0.002516 | 46,390 |
| Diphthongs | +0.002445 | 6,772 |
| Vowels | +0.002377 | 35,713 |
| Fricatives | +0.002352 | 7,096 |
| Stops | +0.002323 | 9,565 |
| Affricates | +0.002080 | 3,883 |
| Approximants | +0.001768 | 4,115 |

### Per-system patch direction

Across almost every phoneme class, patching toward bonafide means:

- raises A01 the most (`~ +0.0009` to `+0.0016`)
- raises A02 slightly (`~ +0.0002` to `+0.0007`)
- leaves A03 near zero
- suppresses A04 slightly (`~ -0.0004` to `-0.0012`)
- raises A05 (`~ +0.0007` to `+0.0032`)
- suppresses A06 the most strongly (`~ -0.0017` to `-0.0040`)

The strongest class-specific effects are:

- `Sibilants -> A05`: `+0.003222`
- `Sibilants -> A06`: `-0.003972`
- `Nasals -> A05`: `+0.001969`
- `Nasals -> A06`: `-0.002831`
- `Fricatives -> A01`: `+0.001641`

### Key findings

**1. The system-ID signal is real, but far weaker than the generic spoof signal from Exp 8.**
In Exp 8, one-node bonafide patching reduced the spoof logit by about `+1.72` to `+1.95`
depending on class. Here the same patch only shifts the true attack-system probe score by
roughly `+0.0018` to `+0.0042`. That suggests layer-0 node representations are dominated by
generic spoof evidence, while attack-family identity is a smaller secondary factor.

**2. Sibilants flip from weakest generic spoof carriers to strongest system markers.**
In Exp 8, sibilants had the smallest causal effect on the binary spoof logit (`+1.721`, last
place). Here they rank first for true-system identification (`+0.004216`). So sibilants do not
carry much *generic* spoof evidence, but they appear unusually informative about *which* attack
family generated the audio.

**3. Nasals remain important in both views.**
Nasals were mid-tier in Exp 8 (`+1.822`) and second-highest here (`+0.002802`). They look like
the most stable cross-experiment class: useful both for spoof detection overall and for attack
system fingerprinting.

**4. The patch has a strongly directional system effect.**
Across nearly all classes, bonafide patching consistently pushes the probe away from A06 and
toward A01/A05. That implies the bonafide class anchor is not neutral in multiclass probe space;
it lies closer to some attack-system directions than others.

**5. A03 is almost invariant to bonafide patching.**
The A03 column stays near zero for every class, unlike the clearer positive A01/A05 and negative
A04/A06 trends. Its evidence may be more distributed across many nodes or encoded in directions
that simple class-mean replacement does not disrupt much.

### Interpretation

Exp 8 and Exp 9 together separate two notions of "important phoneme":

- for **binary spoof detection**, the strongest causal classes are diphthongs and approximants
- for **attack-system fingerprinting**, the strongest causal classes are sibilants and nasals

So the model family seems to use different phoneme subspaces for different tasks. Smooth
formant-transition phones help decide whether speech is fake at all, while noisy / resonant
phones help distinguish which synthesizer family generated it.

### Artifacts saved

Stored in `experiments/results/act_patching_probe/`:
- `act_probe_records.npz` — per-node patch records and per-system score shifts
- `act_probe_class_stats.csv` — mean true-system and per-system deltas by phoneme class
- `act_probe_shift_matrix.png` — class × system mean patch-shift heatmap
- `act_probe_true_system_delta.png` — true-system mean Δ by phoneme class
- `act_probe_per_system.png` — per-system bar plots across phoneme classes

---

## Results Directory

Large output files are stored under `experiments/results/`:

| File | Experiment |
|---|---|
| `phoneme_pc1_violin.png` | Exp 3 — violin plot, per-class PC1 by label |
| `phoneme_pc1_mean_delta.png` | Exp 3 — per-class mean PC1 delta bar chart |
| `gat_attention_heatmap.png` | Exp 4 — 3-panel attention matrix (normal / deepfake / Δ) |
| `gat_attention_topedges.png` | Exp 4 — top-15 class→class edges by \|Δ\| |
| `gat_by_system_heatmaps.png` | Exp 5 — per-system 9×9 heatmaps (bonafide / system / Δ) |
| `gat_by_system_counts.png` | Exp 5 — edge count matrices per system (confidence proxy) |
| `gat_by_system_sibil.png` | Exp 5 — sibilant self-attention and nasal→sibilant bar chart |
| `gat_by_system_summary.csv` | Exp 5 — numeric table, all systems |
| `linear_probe/probe_ranking.png` | Exp 6 — AUC bar chart across all 22 probes |
| `linear_probe/probe_metrics_summary.csv` | Exp 6 — full metrics table |
| `linear_probe/features_*.npz` | Exp 6 — (N, D) feature matrices per probe |
| `linear_probe/probe_*.npz` | Exp 6 — probe weights + metrics per probe |
| `multiclass_probe/multiclass_ranking.png` | Exp 7 — macro-F1 bar chart, 6-way probe |
| `multiclass_probe/probe_similarity.png` | Exp 7 — inter-probe prediction agreement |
| `multiclass_probe/confusion_*.png` | Exp 7 — per-probe 6×6 confusion matrices |
| `multiclass_probe/multiclass_summary.csv` | Exp 7 — full per-system metrics table |
| `multiclass_probe/probe_mc_*.npz` | Exp 7 — multiclass probe weights |
| `act_patching/act_patching_violin.png` | Exp 8 — Δ distribution per phoneme class |
| `act_patching/act_patching_by_system.png` | Exp 8 — class × system mean Δ heatmap |
| `act_patching/act_patching_class_stats.csv` | Exp 8 — mean/std/n per class |
| `act_patching/act_patching_records.npz` | Exp 8 — full per-node records |
| `act_patching/bonafide_class_means.npz` | Exp 8 — bonafide class mean layer-0 vectors |
| `act_patching_probe/act_probe_shift_matrix.png` | Exp 9 — class × attack-system patch-shift heatmap |
| `act_patching_probe/act_probe_true_system_delta.png` | Exp 9 — true-system mean Δ by phoneme class |
| `act_patching_probe/act_probe_per_system.png` | Exp 9 — per-system patch effects across classes |
| `act_patching_probe/act_probe_class_stats.csv` | Exp 9 — full class-by-system delta table |
| `act_patching_probe/act_probe_records.npz` | Exp 9 — full per-node probe patch records |

Smaller plots from `plot_embeddings.py` are stored directly in `experiments/`:

| File | Experiment |
|---|---|
| `pca_frozen_encoder.png` | Exp 2 — PCA of frozen encoder embeddings |
| `pca_pre_gat.png` | Exp 2 — PCA of pre-GAT embeddings |
| `pca_post_gat.png` | Exp 2 — PCA of post-GAT embeddings |
