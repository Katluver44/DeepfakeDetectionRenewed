# Audit 9 — Is "Hardness" Itself a Reliable Measurement?

**Question.** Hardness = 1−AUC of each system's utterances vs a *shared* bona pool, with as few as 8 utterances per system. Two failure modes would undermine every downstream claim: (i) shared-pool dependence (a few odd bona utterances co-move every system's hardness, inflating cross-predictor and cross-detector correlations); (ii) measurement noise capping attainable R².

**Method.** Split-half reliability of MLAAD system hardness (50 within-system splits, Spearman–Brown); bona-pool bootstrap (1,000 resamples of the bona pool, propagated into sd_along's ρ); min-utts threshold sensitivity (8/12/16/25); cross-seed hardness agreement.

**Results** (`audit9_results.json`, `min_utts_sensitivity.csv`, `audit9_reliability.png`).

- **Split-half reliability = 0.759 → full-sample reliability ≈ 0.863.** The R² ceiling for any predictor is ≈0.86; sd_along's 0.277 uses ~32% of the explainable variance. The paper's "~69% unexplained" (§7) should be restated as "~68% of *explainable* variance unexplained" with the ceiling cited.
- **Shared-bona-pool dependence is negligible:** under bona-pool bootstrap, hardness rankings are essentially invariant (rank stability ≈ 1.00) and sd_along's ρ moves within [0.592, 0.603]. The bona pool is large enough that this concern dies.
- **Threshold sensitivity supports the law:** restricting to better-measured systems *raises* the correlation (ρ = 0.597 at ≥8 utts, 0.642 at ≥12, 0.688 at ≥16) — the classic signature of attenuation by measurement noise in the small systems, i.e., the true effect is *underestimated*, not manufactured by noisy small systems.
- Cross-seed hardness agreement: ρ = 0.89–0.97 — hardness is a property of the (architecture, training domain) pair, not of a single random seed.

**Verdict. The hardness metric is sound.** None of the metric-level failure modes materialize, and the two sensitivity analyses both push in the paper's favor. Recommended reporting upgrades: cite the reliability ceiling next to all R² values; report the ≥12/≥16-utts sensitivity row as evidence of attenuation rather than fragility.
