# E9 Pre-registration — Mixed Training (WaveFake + VCC2020 + LibriSpeech)

**Filed:** 2026-05-28  
**Status:** LOCKED — do not modify after first training run begins

---

## Training Data Composition

**Substitution note:** MLAAD English (Müller et al. 2024) is unavailable locally and not
accessible on HuggingFace. WaveFake (Jöhren & Kolossa, 2021, Zenodo:5642694) is substituted
as the TTS-attack source. WaveFake contains neural-vocoder resynthesis (MelGAN, MB-MelGAN,
Full-Band MelGAN, Parallel WaveGAN, WaveGlow, HiFi-GAN, WaveNet) of LJSpeech and VCTK — all
English read speech, comparable scope to MLAAD.

**Actual training counts (bottlenecked by VCC2020):**
- WaveFake TTS spoof: 2,480 (sampled to match VCC2020, subset of a random vocoder mix)
- VCC2020 Task 1 VC spoof: 2,480 (all available; 31 systems × 80 files)
- LibriSpeech train-clean-100 bonafide: 4,960 (to match total spoof count)
- **Total: ~9,920 (~10k), 80/20 stratified train/val split**

Deviation from spec (25k target): the 25k figure was bottlenecked by VCC2020. Per-spec
rationale: "the absolute count matters less than the composition ratio." Ratio is maintained:
spoof/bonafide = 50/50; TTS/VC within spoof = 50/50.

Train/val split: 80/20 stratified within each component, seeded. Two seeds trained.

---

## Hypotheses (pre-registered, not to be modified after seeing results)

### H1 [primary] — Gini taxonomy split

**Prediction:** The trained model on WaveFake+VCC2020+LibriSpeech will show the same
Gini-direction split as the original ASVspoof seed-1 model:
- TTS systems (WaveFake vocoders): mean per-class Gini of A_agg in-degree < bonafide Gini
  (attention-flattening direction)
- VC systems (VCC2020 T01–T33): mean per-class Gini of A_agg in-degree > bonafide Gini
  (attention-concentrating direction)

Both directions must hold for **both** new-model seeds.

**Falsification criterion:** Either direction reversed in either seed.

**Rationale:** TTS processes produce smoothly-varying phoneme sequences that lead to uniform
attention; VC converts prosody of real speech and preserves irregular phoneme-duration patterns,
leading to concentrated attention on long-duration phonemes. If this mechanism is architectural
rather than dataset-specific, it should re-emerge on WaveFake/VCC2020.

---

### H2 [primary] — TTS-vs-VC per-sample classifier

**Prediction:** A logistic regression (10 structural features) trained on the new model's
A_agg matrices will achieve balanced accuracy ≥ 0.70 on TTS (WaveFake) vs VC (VCC2020)
classification, using 5-fold CV. Result replicates within 0.10 across both new-model seeds.

**Falsification criterion:** balanced accuracy < 0.60 in either seed, or seed-to-seed gap > 0.15.

**Rationale:** If H1 holds (Gini directions separate), the same features used in E7 should
classify TTS vs VC at the sample level. WaveFake uses different vocoders than ASVspoof A01–A06
but the same underlying speech (LJSpeech/VCTK vs LibriSpeech-sampled) — the mechanism check
is whether the feature space separates despite the different attack provenance.

---

### H3 [primary] — Quasi-adjacency F1 (structural rewiring)

**Prediction:** For the new model, the best-case F1 between thresholded A_agg and binary
A_input (phoneme adjacency DAG) will be at or below the random permutation floor for all
evaluation classes. Δ(trained − random) < +0.10 for every class.

**Falsification criterion:** Δ > +0.10 for any class, indicating the new model aligns
substantially with the input graph structure.

**Rationale:** E4 showed the original model completely rewires A_input. H3 tests whether
this is an architectural property or an ASVspoof-specific artifact. If it replicates, the
rewiring finding is robust and the E4 contribution is significantly strengthened.

---

### H4 [secondary, informational] — Dominant feature identity

**Prediction:** Among the 10 structural features in the E7-equivalent classifier, the top-2
by absolute logistic regression coefficient will include at least one of {offdiag_frobenius,
gini_indegree} — the features that dominated in the original seed-1 E7.

**Not a falsification test.** Informational only: characterizes mechanism consistency.

---

## Decision Logic

| H1 | H2 | H3 | Interpretation |
|----|----|----|----------------|
| ✓ both seeds | ✓ both seeds | ✓ | **Major finding.** Pathway split is a robust property of this architecture on TTS+VC data. E8 failure reframed: mechanism re-emerges when retrained on different data. |
| ✓ | ✗ | ✓ | Structural taxonomy survives but per-sample classifiability doesn't on this data. Narrower claim: distribution-level but not sample-level discrimination. |
| ✗ | — | — | **Taxonomy is ASVspoof-specific.** Original finding stands as within-protocol. Generalization claim is dead. Write up as significant negative result. |
| ✓ | ✓ | ✗ | Mechanism replicates but new model learns to use the input graph. Weakens E4; suggests rewiring is training-data dependent. |
| ✓ | ✓ | ✓ | Full replication. All three primary hypotheses confirmed. |

---

## EER Sanity Gate

Each new checkpoint must achieve test EER ≤ 0.20 on its own validation set before proceeding
to mechanism analysis. If EER > 0.20: investigate, retrain. Do not run mechanism analysis on
a poorly-trained model.

---

## Evaluation Sets

**(A) New-domain held-out (primary for mechanism analysis):**
- 50 WaveFake samples per vocoder system (held out from training)
- 50 VCC2020 Task 1 samples per VC system (held out from training)
- 50 LibriSpeech samples (held out from training)

**(B) ASVspoof 2019 LA (cross-protocol comparison):**
- 50 bonafide + 50 per attack system (A01–A06) — from existing cached eval set
- Same 350 samples used in E2–E8

---

## Notes

- WaveFake audio is 22050 Hz; resampled to 16000 Hz by `loader.py:_ensure_sr()`.
- VCC2020 audio is 24000 Hz; resampled to 16000 Hz by same function.
- LibriSpeech is natively 16000 Hz.
- Phoneme count distribution comparison will be run before training starts (pre-training
  check (b)) and logged to this directory.
- All A_agg matrices cached post-training for re-analysis.
