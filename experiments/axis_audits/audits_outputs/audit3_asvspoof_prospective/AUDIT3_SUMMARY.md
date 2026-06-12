# Audit 3 — ASVspoof 2021 "Pre-Registered Prospective" Claims (J4/J5)

**Claims under test.** "P3 achieves ρ=+0.599 (p=0.031) in a prospective pre-registered prediction on ASVspoof 2021, replicating for AASIST (ρ=+0.643, p=0.018)"; "2/3 top-3 system hits".

**Red flags.** (i) n = 13 attacks. (ii) Four predictors (P1–P4) were registered; **P1 was declared primary and failed** (ρ=0.25, ns) — the headline P3 is a promoted secondary. (iii) The earlier prospective experiment J2 failed on 9 of 9 tests before J4 was designed (sequential testing). (iv) "Pre-registration" is a JSON written by the same script **63 seconds** before detector scoring.

**Method.** Exact permutation p-values (200k perms); Holm within each detector's 4-predictor family and across all 8 tests; global context from Audit 1; measurement-noise bootstrap (resampling bona pool and attack utterances, 3,000 draws); leave-one-attack-out jackknife; hypergeometric baseline for top-3 hits; replication-independence check.

**Results** (`family_corrected_pvalues.csv`, `loao_jackknife.csv`, `audit3_asvspoof.png`).

- Exact permutation p: P3 WavLM 0.034, P3 AASIST 0.020. **After Holm within-family: 0.134 / 0.082. Across the 8-test family: 0.235 / 0.164. BH q = 0.134.** Nothing survives.
- Counting J2's 9 failed prospective tests, a Bonferroni over the sequential family puts P3 at p ≈ 0.57.
- **However, robustness evidence is genuinely favorable:**
  - The two detectors' per-attack hardness profiles correlate only ρ=0.21 (p=0.48) — so the AASIST result is a quasi-independent replication, not the same ranking re-tested.
  - Measurement bootstrap: P3's ρ never crosses 0 in 3,000 draws for either detector (WavLM CI [0.33, 0.70]; AASIST CI [0.52, 0.75]).
  - Leave-one-attack-out: ρ stays in [0.49, 0.70] (WavLM) and [0.56, 0.74] (AASIST).
- Top-3 hits: P(≥2/3 by chance) = 0.108, and P1, P2, P3 all predicted the *same* top-3 set {A17, A18, A19} — the hit metric cannot distinguish the failed predictors from P3. AASIST got only 1/3 hits (unreported).
- Pre-registration: predictions provably cannot depend on detector scores (phase ordering in code), but there is no external commitment device, and registration-to-scoring gap is 63 s.

**Verdict. NOT PROVEN as a confirmatory result; PROMISING as an exploratory one.** The honest framing is: "the registered primary failed; a secondary corpus-internal LDA predictor showed ρ≈0.6 in both of two quasi-independent detector families (uncorrected p≈0.02–0.03, n=13; does not survive family-wise correction)". Any "pre-registered, p<0.05" language must be removed. The result's best property — two near-independent detector replications with stable jackknife behavior — should be foregrounded instead of the p-values. A true out-of-sample confirmation (e.g., ASVspoof 2021 DF subset, or codec conditions ≠ 'none') is the obvious decisive next experiment.
