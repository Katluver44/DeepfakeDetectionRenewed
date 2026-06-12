# Audit 8 — I1 Causal Intervention Contrasts: Inference at the Right Unit

**Claims under test.** §5's I1 table, especially: iso_0.7 "small causal C effect" (ΔAUC=−0.0023, p=0.024); sub_top −0.0320***; sub_res +0.0039***; gated-vs-ungated +0.0101*** ("artifact accounts for ~75–80%"); §4's "true gated C-effect is ΔAUC = −0.0036".

**Red flag.** I1's p-values come from utterance-level bootstrap pooling 3 seeds — utterances are pseudo-replicates of a per-seed effect. The artifact's own `seed_signs` column shows several "significant" contrasts with seeds disagreeing in sign.

**Method.** Rebuilt per-seed × per-condition AUCs from `i1_conditions.csv`; recomputed all 25 contrasts with **seed as the unit** (paired t-test, n=3; sign consistency); BH across contrasts; recomputed the artifact-share arithmetic.

**Results** (`seed_level_contrasts.csv`, `audit8_i1_seeds.png`).

- **iso_0.7 (the "small causal C effect", reported p=0.024): seeds disagree in sign ([−,+,−]); seed-level p=0.21. CONTRADICTED.** Same for iso_0.5, shift_0.5, jitter_1.0 — every "significant" iso/shift compaction effect is seed-inconsistent.
- The big directional contrasts are seed-consistent in sign: sub_top (−0.032, all seeds negative), sub_res (+0.004, all positive), shuffle (−0.017, all negative), gated-vs-ungated (+0.010, all positive, seed-level p=0.040), sub_top-vs-sub_res (−0.036, all negative, seed-level p=0.071).
- **With n=3 seeds, no contrast survives seed-level BH** (min q=0.168). The `***` markers overstate evidential strength by orders of magnitude.
- Arithmetic checks: artifact share = 81.2% (claim "75–80%" ≈ verified). "True gated C-effect −0.0036" is **mislabeled** — that value is the iso_0.7-vs-shift_0.7 contrast; gated iso_0.7 vs baseline is −0.0023.

**Verdict.** The qualitative I1 story survives: the *ungated-hook artifact* is real and large, and the *direction-content* contrasts (sub_top vs sub_res) are consistent across all three seeds with a 10× effect-size separation — that asymmetry is the mechanistically meaningful result. But (i) the claimed *residual small causal C effect does not replicate across seeds and should be retracted or labeled inconclusive*; (ii) all I1 p-values should be re-derived at the seed level (which, at n=3, mostly means reporting sign consistency + effect sizes, not stars); (iii) two numerical mislabels need fixing. Adding 3–5 more seeds for the headline contrasts would let sub_top/sub_res reach defensible seed-level significance.
