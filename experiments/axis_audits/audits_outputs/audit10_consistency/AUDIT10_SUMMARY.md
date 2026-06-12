# Audit 10 — Number-by-Number Consistency: Report vs Artifacts

**Question.** Does every number in `final_outputs2` trace to an artifact, and is it the number it claims to be?

**Method.** Cross-checked all 21 quantitative claims in the abstract, results tables, key-findings list, and figures against the I/J results JSONs/CSVs and the recomputations from Audits 2–9. Full table in `claims_vs_artifacts.csv`.

**Tally: 8 VERIFIED · 6 MISLABELED · 5 MISLEADING · 2 CONTRADICTED.**

**Contradicted (must be removed/corrected):**
1. "cos(w_MLAAD, w_ASVspoof21) ≈ 0.36" — true value is *negative* (−0.14 to −0.21); 0.362 is J1's cos(w_mean, w_lda) within MLAAD, an unrelated quantity.
2. iso_0.7 "small causal C effect (p=0.024)" — seeds disagree in sign; seed-level p=0.21.

**Mislabeled (fix wording/numbers):**
- table2 reports the *triplet* R² (0.316) under the name sd_along while the abstract uses sd_along-alone (0.273) — both with the same p-value.
- "ITW-internal axis *fusion* 0.363→0.292" — 0.292 is axis-alone; fusion was 0.312 (worse).
- "true gated C-effect −0.0036" — that value is the iso-vs-shift contrast; gated-vs-baseline is −0.0023.
- "58 speakers" — 29 analyzed (49 in subset).
- "13 controlled causal interventions" — I1 has 22 conditions.
- Figure 1: left panel title shows s_along's stats (R²=0.020, p=0.023) under an sd_along scatter (true R²=0.277 — understates own result); middle panel mixes I4's ρ=−0.714 (a different ASVspoof subset/axis) with a "P3 (LDA axis score)" x-axis from J4 (whose ρ=+0.599).

**Misleading (reframe):**
- sd_along "replicates on AASIST-FT (ρ=+0.349)" — true, but the LOSO-R² form of the law *fails* on AASIST-FT (r²=−0.03, unreported); metric switched silently between detectors.
- P3 "pre-registered, p=0.031" — registered primary P1 failed; P3 is a promoted secondary; fails all family corrections (Audit 3).
- "2/3 top-3 hits" — P1, P2, P3 predicted the same top-3 set; AASIST hit only 1/3 (unreported).
- "Zero-training-cost fusion, up to 33 pts" — gains come from a supervised eval-corpus LDA probe that alone beats the fusion (Audit 5).
- s_orth as *the* ITW predictor — collinear with stronger, unreported rog12 (Audit 6).

**Provenance gaps:** headline 0.273/p=0.0005 hardcoded in build_package.py (verified by recomputation, Audit 2); `itw_fusion_test.json` has no committed generating script; figure-stat wiring in build_package.py pulls wrong dictionary keys.

**Verdict.** The quantitative core traces to artifacts and mostly reproduces, but the packaging layer (build_package.py) introduced a substantial density of transcription/labeling errors — including two outright contradictions — that would not survive referee scrutiny. Every number in a camera-ready draft should be regenerated programmatically from artifacts, never hand-copied.
