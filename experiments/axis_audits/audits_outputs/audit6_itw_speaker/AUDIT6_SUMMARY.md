# Audit 6 — In-The-Wild Per-Speaker Hardness Claims

**Claim under test.** "Per-speaker hardness: s_orth ρ = +0.534 (p = 0.003) under WavLM-GAT; vel_entropy ns" (§3.3); corpus described as "58 speakers".

**Method.** Family-wise correction over the 7 predictors actually tested in I5; bootstrap CIs and leave-one-speaker jackknife; collinearity and partial-correlation analysis between s_orth and the *stronger but unreported* competitor rog12; confound partials (n_utts/speaker, RMS); split-half reliability of speaker hardness.

**Results** (`audit6_results.json`, `speaker_family_corrected.csv`, `audit6_itw.png`).

- **Counts:** the analyzed subset has 49 spoof speakers, of which **29** (≥10 utts) enter the hardness law. "58 speakers" is wrong for this analysis.
- **Multiplicity:** s_orth survives Holm (p=0.017) and BH (q=0.010) within the 7-test family. So the correlation is not a multiple-comparisons artifact.
- **But the attribution is.** rog12 (plain radius-of-gyration, no axis needed) is *stronger* (ρ=−0.561, p=0.0015, q=0.010) and went unreported. s_orth and rog12 are collinear (ρ=−0.687) and **neither survives partialling out the other** (s_orth|rog12: ρ=0.21, p=0.27; rog12|s_orth: ρ=0.003, p=0.99). They are one shared signal, and on present evidence the non-axis description (compactness) is at least as good as the axis one (off-axis residual). vmean0 (ρ=−0.46, q=0.026) — a *layer-0* feature, i.e., low-level audio properties — also predicts hardness, hinting at a channel/recording confound driving the shared factor.
- **Stability:** s_orth bootstrap CI [+0.20, +0.76]; jackknife [+0.50, +0.60]. rog12 CI [−0.80, −0.22]. n=29 means wide CIs; the effects are real but imprecise.
- **Reliability ceiling:** speaker-hardness split-half reliability 0.72 (Spearman–Brown ≈ 0.84); observed |ρ|≈0.53–0.56 sits well inside the ceiling (≈0.91), so noise does not explain the unshared variance.
- s_orth survives partialling n_utts (ρ=0.54) and RMS (ρ=0.46).

**Verdict. PARTLY SUPPORTED, WRONGLY ATTRIBUTED.** A robust speaker-level geometry→hardness correlation exists on ITW, but the paper's specific claim that the *off-axis residual* is the operative quantity is not separable from plain embedding compactness (rog12), which is stronger and unreported. The paper should either report both and present them as one factor, or run a discriminating experiment (e.g., regress out rog12 at the utterance level, or test on speakers matched for rog12). The "axis" framing for ITW is currently over-interpretation. The speaker count must be corrected to 29 analyzed (49 in subset).
