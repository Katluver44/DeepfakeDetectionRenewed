# Audit 2 — The Headline sd_along Claim

**Claim under test.** "The spread (sd_along) achieves leave-one-system-out R² = 0.273 on MLAAD (p = 0.0005)."

**Red flag motivating this audit.** The value 0.273 is **hardcoded** in `final_outputs2/build_package.py` (line 317) and appears in *no* results JSON. `i3_stats.json` contains only s_along's LOSO R² (0.020) and the triplet model (0.316). The provenance of the headline number was therefore unverified.

**Method.** Full independent recomputation from raw cached artifacts (`i3 embeddings.npz` + 3-seed logits): LOSO axis construction, per-system hardness, identical ridge-LOO estimator. Fidelity vs the original `system_position.csv` confirmed (r ≥ 0.9997 on every column). Then: permutation null (4,000 perms), per-seed stability, leave-one-system jackknife, simultaneous removal of the most influential systems, confound partials (n_utts, waveform RMS, language, vel_entropy), estimator sensitivity (ridge α, IQR spread, log-hardness).

**Results** (`audit2_results.json`, `jackknife_leave_one_system.csv`, `audit2_sdalong.png`).

- **Recomputed sd_along-alone LOSO R² = 0.277, permutation p = 0.0002** — the hardcoded 0.273/0.0005 is accurate. Spearman ρ = +0.597 (p < 10⁻⁶), bootstrap CI [+0.41, +0.73].
- Per-seed: ρ = 0.573 / 0.606 / 0.598; R² = 0.248 / 0.318 / 0.223 — stable across detector seeds.
- Jackknife: ρ ∈ [0.579, 0.632] over all 61 leave-one-system folds; **0/61 folds lose significance**. Dropping the 2 or 5 most influential systems *raises* ρ (0.66).
- Confounds: partial ρ given n_utts = +0.579; given RMS = +0.583; within-English-only (n=50) ρ = 0.597. sd_along retains ρ = +0.617 after partialling vel_entropy (the converse partial drops to 0.277) — sd_along subsumes most of vel_entropy's signal, not vice versa.
- Estimator sensitivity: R² = 0.275–0.281 across ridge α ∈ {0.1, 1, 10}; IQR-based spread ρ = 0.469; log-hardness ρ = 0.598.

**Caveats found.**
- The triplet R² (0.316) and sd_along-alone R² (0.273) are conflated across the abstract and table2 (Audit 10).
- MLAAD language coverage is thin: only English has ≥6 systems in the test set, so "within-language" robustness is only demonstrable for English.

**Verdict. VERIFIED and robust.** This is the strongest claim in the paper: it survives permutation, jackknife, influence analysis, confound partials, estimator perturbations, and the Audit-1 winner's-curse simulation. The provenance hygiene (hardcoded number, no generating script) should still be fixed.
