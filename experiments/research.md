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

## Results Directory

Large output files are stored under `experiments/results/`:

| File | Experiment |
|---|---|
| `phoneme_pc1_violin.png` | Exp 3 — violin plot, per-class PC1 by label |
| `phoneme_pc1_mean_delta.png` | Exp 3 — per-class mean PC1 delta bar chart |

Smaller plots from `plot_embeddings.py` are stored directly in `experiments/`:

| File | Experiment |
|---|---|
| `pca_frozen_encoder.png` | Exp 2 — PCA of frozen encoder embeddings |
| `pca_pre_gat.png` | Exp 2 — PCA of pre-GAT embeddings |
| `pca_post_gat.png` | Exp 2 — PCA of post-GAT embeddings |
