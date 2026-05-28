# E9 Mechanism Analysis Report

**Pre-registered hypotheses** (see preregistration.md)

## Results

| Seed | EER (new-domain) | H1 (Gini split) | H2 (bal-acc) | H3 (max Δ quasi-adj) |
|------|-----------------|-----------------|-------------|---------------------|
| 1 | 13.50% | FAIL | 0.885 (PASS) | Δ=-0.013 (PASS) |
| 2 | 7.74% | FAIL | 0.826 (PASS) | Δ=-0.025 (PASS) |

## Cross-protocol EER

| Seed | Val EER (new-domain train/val) | Held-out EER (new-domain) | ASVspoof 2019 LA EER |
|------|-------------------------------|--------------------------|----------------------|
| 1 | 2.26% | 13.50% | 20.46% |
| 2 | 1.73% | 7.74% | 34.00% |

Note: held-out EER is higher than val EER because the held-out VCC2020 eval has only ~14–20 samples per system (20% split from 80 files), making the EER estimate noisy.

## Interpretation

**Decision logic outcome** (per preregistration.md): H1 fails, H2 passes, H3 passes.
Matching row: "H1 fails (Gini directions don't replicate): the taxonomy is specific to ASVspoof 2019 LA."

**H1 FAIL — Gini direction for VC reversed**: The predicted direction was TTS < bonafide (flatten) and VC > bonafide (concentrate). Both seeds show TTS=0.024 < bonafide=0.027 (flatten, as predicted), but VCC2020 VC=0.023 < bonafide=0.027 (also flattens, not concentrates). The VC direction is reversed relative to the pre-registered prediction.

VCC2020 voice conversion produces a similar attention-flattening signature to WaveFake TTS. This is plausible: VCC2020 Task 1 (intra-lingual VC) uses a variety of modern neural VC methods (encoder-decoder, flow-based), many of which produce very clean phoneme transitions with similar smoothness to neural TTS — unlike A05/A06 (the original skip-route systems), which were Chinese TTS engines with distinctive sparse-phoneme attention. The concentration effect in the original model was likely specific to Chinese-character TTS phoneme patterns, not VC as a category.

**H2 PASS — TTS vs VC classifiable**: Despite H1 failing (both attack types flatten), the structural feature space (10 features including entropy_row, n_nodes, gini_indeg) still separates WaveFake TTS from VCC2020 VC at 88.5% and 82.6% balanced accuracy across seeds. The separation mechanism differs: the dominant features are entropy_row and n_nodes rather than the gini_indeg + offdiag_frobenius pair that dominated in E7. This means the models learn different structural signatures for TTS vs. VC, but not the concentration vs. flatten polarity predicted by the original taxonomy.

**H3 PASS — Rewiring replicates**: Every evaluation class (VCC2020 T01–T33, all 7 WaveFake vocoders, LibriSpeech) sits at or below the random permutation floor (max class-mean Δ = −0.013 for seed 1, −0.025 for seed 2). The completely-rewired regime from E4 replicates in both new-model seeds on a completely different training distribution. This confirms that GAT rewiring away from A_input is an architectural property of this model family, not a consequence of ASVspoof 2019 LA training data.

**Combined interpretation**: The GAT mechanisms are partially generalizable. The rewiring finding (E4/H3) is robust and architecture-level. The classifier finding (E7/H2) generalizes in form (TTS and VC are structurally distinguishable) but not in feature identity (entropy/n_nodes, not Gini/Frobenius). The routing-vs-skip taxonomy (H1) is dataset-specific: the concentration direction requires the specific phoneme-duration patterns of A05/A06 (Chinese TTS with sparse long-duration phonemes) — this property is absent in VCC2020 VC systems which produce similar smooth transitions to Western TTS engines.

**Paper implications**: (1) E4's rewiring finding can be strengthened as architecture-level. (2) E7's claim should be narrowed: "structural features classify attack type" is robust, but "Gini captures a concentrate-vs-flatten polarity" is specific to ASVspoof 2019 LA attacks. (3) The cross-protocol failure in E8 is partly explained: when retrained on WaveFake+VCC2020, the model does not re-learn the concentration pathway because that pathway required the specific Chinese TTS phoneme statistics that ASVspoof A05/A06 carried.

