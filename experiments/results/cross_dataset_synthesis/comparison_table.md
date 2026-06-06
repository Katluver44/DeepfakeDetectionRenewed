# Cross-Dataset Comparison Table

Source datasets: **ASVspoof 2019 LA** (English, 6 attack systems) vs **MLAAD** (multilingual, 63 attack systems).

| # | Claim | ASVspoof | MLAAD | Verdict |
|---|-------|----------|-------|---------|
| 1 | **KL range widening (robust > baseline)** | robust=3.19x  goat=1.13x  widening=+2.06x | robust=1.63x  goat=1.42x  widening=+0.21x | ⚠️ generalizes_with_caveats |
| 2 | **Top-2 ablation effect in robust checkpoint** | Δ=+0.0433  ctrl Δ=+0.0033  gap=+0.0400 | Δ=+0.0574  ctrl Δ=-0.0146  gap=+0.0720 | ✅ generalizes_cleanly |
| 3 | **Top-2 ablation effect in baseline checkpoint (using robust's head IDs)** | Δ=+0.0000  null_pct=57.1%  (own-heads Δ=-0.0200) | Δ=+0.0905  null_pct=100.0% | ❌ does_not_generalize |
| 4 | **Null distribution percentile (cross-checkpoint ablation test)** | 57.1%  Δ_observed=+0.0000 | 100.0%  Δ_observed=+0.0905 | ❌ does_not_generalize |
| 5 | **Entropy collapse (robust_goat top heads sharper than goat's)** | top1(h0): goat=2.035 → robust=1.406 Δ=-0.628; top2(h4): goat=2.036 → robust=1.592 Δ=-0.444 | top1(h2): goat=1.584 → robust=1.413 Δ=-0.171; top2(h4): goat=1.625 → robust=1.488 Δ=-0.137 | ⚠️ generalizes_with_caveats |
| 6 | **Top-1 head differentiates more than top-2 (entropy / L2)** | h0 entropy Δ=-0.628 > h4 Δ=-0.444  ratio=0.71 | h2 L2=0.047 > h4 L2=0.034  ratio=0.73 | ✅ generalizes_cleanly |
| 7 | **Functional division of labor (general detector + attack-type encoder)** | h0: Other→Other peak, attack-insensitive; h4: Vowels→Other peak, attack-specific routing (TTS/VC p=0.641) | h2: Other→? peak; h4: Other→? peak; both target 'Other' | ⚠️ generalizes_with_caveats |
| 8 | **Per-class F1 cascade (attack-system classification performance)** | 7-class probe: goat macro_F1=0.713 post_gat; A05 highest F1=0.89 | Binary probe only (63 systems, not individually labeled in probe) | 🔬 asvspoof_specific |
| 9 | **GAT-localized: pre_gat features byte-identical between checkpoints** | pre_gat EER delta=0.0000  CI=[0.0, 0.0]  flag=gat_localized | pre_gat EER delta=0.0000  flag=gat_localized | ✅ generalizes_cleanly |
| 10 | **Linear probe ordering: post_wavlm vs pre_gat vs post_gat EER** | post_wavlm=0.1008  pre_gat=0.0800  post_gat_g=0.1138  order: pre_gat < post_wavlm < post_gat | post_wavlm=0.1522  pre_gat=0.3976  post_gat_g=0.4030  order: post_wavlm << pre_gat ≈ post_gat | ❌ does_not_generalize |
| 11 | **Cross-language generalization gap** | N/A — ASVspoof 2019 LA is English-only | post_wavlm gap=+0.0431  pre_gat gap=+0.0306  post_gat gap=+0.0615  ablating h2+h4 reduces robust gap by 0.216 | 🌍 mlaad_specific |
| 12 | **Language-invariant attention routing (cross-language cosine stability)** | N/A — monolingual dataset | h2: cos_sim=0.9528  h4: cos_sim=0.9481 (10 shared attack systems, in-dist vs cross-lang) | 🌍 mlaad_specific |

## Notes

**1. KL range widening (robust > baseline)**: ASVspoof has 3.2x range vs MLAAD 1.6x — same direction but ASVspoof effect is ~2.5x larger. Widening is present on both but weaker for MLAAD.

**2. Top-2 ablation effect in robust checkpoint**: Both datasets: ablation ΔEER well above random ctrl. Mag ratio ≈ 0.55 (ASVspoof gap=0.040, MLAAD gap=0.072).

**3. Top-2 ablation effect in baseline checkpoint (using robust's head IDs)**: ASVspoof: robust_goat's heads {h0,h4} don't transfer to goat (ΔEER=0, 57th pct). MLAAD: robust_goat's heads {h2,h4} fully transfer to goat (ΔEER=+0.090, 100th pct). Head identity is checkpoint-invariant in MLAAD but not in ASVspoof.

**4. Null distribution percentile (cross-checkpoint ablation test)**: Both tests use the same paradigm: apply robust's target heads to the goat and compare to 14-pair exhaustive null. ASVspoof result is chance-level (57%); MLAAD is at maximum (100%). This reinforces claim 3: head identity only cross-checkpoint stable in MLAAD.

**5. Entropy collapse (robust_goat top heads sharper than goat's)**: Both: robustness training sharpens top heads (lower entropy). Effect ~3x smaller in MLAAD (Δ≈−0.15) than ASVspoof (Δ≈−0.50). Same direction across all systems.

**6. Top-1 head differentiates more than top-2 (entropy / L2)**: Top-1 head shows consistently larger differentiation than top-2 on both datasets.

**7. Functional division of labor (general detector + attack-type encoder)**: ASVspoof shows clear h0/h4 functional split; TTS/VC dissociation is not significant (p=0.641). MLAAD has 63 attack systems so per-system patterns are noisier. Both top heads attend to 'Other' class primarily. Clear functional split not confirmed on MLAAD.

**8. Per-class F1 cascade (attack-system classification performance)**: ASVspoof has 6 labeled attack systems enabling per-class F1. MLAAD's 63 systems are too numerous for stable per-class estimates with the available sample sizes.

**9. GAT-localized: pre_gat features byte-identical between checkpoints**: Strongest claim: encoder weights are frozen; all robustness-training changes are localized to GAT+BiLSTM. Confirmed byte-identically on both datasets.

**10. Linear probe ordering: post_wavlm vs pre_gat vs post_gat EER**: Opposite orderings. ASVspoof: phoneme pooling (pre_gat) IMPROVES linear separability over raw WavLM — encoder adds discriminative structure. MLAAD: phoneme pooling DRAMATICALLY degrades separability (EER 0.15→0.40). Both still confirm GAT-localized since the change between checkpoints is zero at pre_gat. This may reflect the difference in dataset difficulty: ASVspoof has 6 known attack systems; MLAAD has 63 heterogeneous TTS/VC systems spanning 2025 models.

**11. Cross-language generalization gap**: MLAAD provides a direct cross-language test. Post-GAT features degrade most across languages (gap=+0.070 vs +0.043 at post_wavlm). Ablating h2+h4 on robust_goat reduces the in/out-of-distribution gap by 0.217, identifying these heads as language-overfitting components.

**12. Language-invariant attention routing (cross-language cosine stability)**: Despite ablating these heads dramatically improving cross-language EER, the attention PATTERNS themselves are stable (cos_sim≈0.95). Contradiction: heads encode language-harmful features at the routing level but their attention matrices look similar across languages. Suggests downstream representation use (BiLSTM) is the language-sensitive component, not the attention selection itself.

## Legend

- ✅ generalizes_cleanly — same direction, magnitude ratio ≥ 50%
- ⚠️ generalizes_with_caveats — same direction, magnitude ratio < 50%, or caveat applies
- ❌ does_not_generalize — opposite directions or one dataset is null
- 🌍 mlaad_specific — cross-language measurement, not possible on ASVspoof
- 🔬 asvspoof_specific — requires labeled per-attack systems (ASVspoof only)