# REGEN_REPORT.md — Workstream A: programmatic regeneration report

## What was run

1. Read `experiments/axis_audits/audits_outputs/COMPREHENSIVE_VERDICT.md` in full, plus all
   ten audit scripts (`experiments/axis_audits/audit{1..10}_*.py`), `audit_common.py`, and
   every `auditN_results.json` / summary / CSV already checked into
   `experiments/axis_audits/audits_outputs/`. These were the ground-truth artifacts; nothing
   was re-derived from scratch that the audit suite hadn't already computed and cached.
2. Wrote `experiments/paper_regen/regenerate_numbers.py`, which:
   - adds `experiments/axis_audits` to `sys.path` and imports `audit_common` directly
     (`load_mlaad`, `sys_list_of`, `hardness_table`, `loso_axis_features`, `loo_r2`,
     `perm_p_loo`, `bootstrap_spearman_ci`), reusing the exact canonical loaders/estimators —
     no new methodology;
   - **live-recomputes** the core law (Section 1: sd_along LOSO R²/perm-p/Spearman ρ, per-seed
     breakdown, leave-one-system-out jackknife range, and the triplet-model R² that table2
     mislabels) directly from the cached MLAAD embeddings/logits, exactly mirroring
     `audit2_sdalong_claim.py`;
   - **live-recomputes** the AASIST-FT rank-order-vs-LOSO-R² split (Section 6) and the I2
     battery's zero-FDR-survivor result (Section 7) directly from
     `experiments/results/j6_aasist_mlaad/j6_results.json` and
     `experiments/results/i2_geometry_battery/univariate.csv`;
   - **re-reads** the cached `auditN_results.json` outputs for everything that requires
     machinery already implemented in a specific audit script (hardness reliability/min-utts
     from audit9, axis cosines from audit4, ITW/domain-shift fusion decomposition from audit5,
     global multiplicity from audit1, causal contrasts from audit8, ITW speaker counts/geometry
     from audit6) — these are Sections 2–5, 8–10;
   - cross-checks (Section 11) every live-recomputed number against the corresponding cached
     audit JSON value and records any mismatch.
3. Ran it to completion with the system `python3` (numpy 1.21.5, pandas 1.3.5, sklearn 0.23.2,
   scipy) — CPU only, no torch/GPU, no package installs:
   ```
   python3 experiments/paper_regen/regenerate_numbers.py
   ```
4. Wrote `experiments/paper_regen/CORRECTIONS.md`, a 17-row claim-by-claim table: rows 1–13 map
   1:1 onto the 13 non-`VERIFIED` rows of
   `experiments/axis_audits/audits_outputs/audit10_consistency/claims_vs_artifacts.csv`
   (6 MISLABELED + 5 MISLEADING + 2 CONTRADICTED, matching the verdict's stated "8/6/5/2"
   tally exactly); rows 14–17 are the additional disclosure items named in verdict §1–§3
   (I2 battery FDR, hardness reliability ceiling + min-utts sensitivity, AASIST rank-order vs
   LOSO-R² split) that are not literally claim/artifact mismatches in audit10 but that the
   verdict explicitly says must be added before submission.

## Confirmation: regenerated numbers vs. audit JSONs

`regenerate_numbers.py`'s Section 11 cross-check compares every number this script
live-recomputed against the value already recorded in the corresponding audit's JSON. All
checks passed within tolerance (1e-3–2e-3, accounting for minor RNG/rounding differences in
permutation p-values):

| Regenerated key | This script | Audit JSON | Status |
|---|---|---|---|
| `CORE_LAW.sd_along_LOSO_R2` | 0.2766 | 0.2766 (`audit2_results.json.headline.recomputed_loso_r2`) | OK |
| `CORE_LAW.sd_along_spearman_rho` | 0.5975 | 0.5975 (`audit2_results.json.headline.spearman_rho`) | OK |
| `AASIST_FT.sd_along_spearman_rho` | 0.3495 | 0.3495 (`j6_results.json.law.sd_along.rho`) | OK |
| `AASIST_FT.sd_along_LOSO_R2` | −0.0324 | −0.0324 (`j6_results.json.law.loo_pos_sd.r2`) | OK |

**Mismatches found: none.** `regenerated_numbers.json._mismatches_vs_audit_json` is an empty
list after the run.

**Missing artifacts: none.** `regenerated_numbers.json._missing_artifacts` is an empty list —
every audit JSON referenced (audit1, 2, 3 [unused directly but available], 4, 5, 6, 7 [unused
directly but available], 8, 9, 10) was present and loaded successfully, and both underlying
experiment artifacts used for live recomputation (`i3_position_geometry`, `i2_geometry_battery`,
`j6_aasist_mlaad`) were present.

Sections 2–5 and 8–10 of the script re-read cached audit JSONs rather than re-executing the
audit scripts' full pipelines in-process (e.g. audit5's 5-fold speaker-disjoint refit, audit8's
per-seed causal contrasts, audit9's 1000-iteration bona-pool bootstrap) — these are
computationally expensive stochastic procedures whose outputs are already the audited,
artifact-backed numbers; re-reading them satisfies "regenerate every reported number
programmatically from artifacts" without duplicating non-deterministic simulation work the
audit suite already ran and checked in. If a fully from-scratch re-execution of those audits is
also wanted, they can be re-run directly and will overwrite their own `auditN_results.json`
files (not touched by this script, which only reads them):
```
python3 experiments/axis_audits/audit1_multiplicity.py
python3 experiments/axis_audits/audit4_axis_rotation.py
python3 experiments/axis_audits/audit5_fusion_claims.py
python3 experiments/axis_audits/audit6_itw_speaker.py
python3 experiments/axis_audits/audit8_i1_causal.py
python3 experiments/axis_audits/audit9_hardness_reliability.py
```
(Not run as part of this workstream, per the instruction to write only inside
`experiments/paper_regen/` and not modify existing files/directories — those scripts write into
`experiments/axis_audits/audits_outputs/`, outside the allowed write path.)

## Final corrected values for every §3 item (COMPREHENSIVE_VERDICT.md "what is wrong and must be
fixed before submission")

1. **`cos(w_MLAAD, w_ASVspoof21) ≈ 0.36` → corrected to −0.207 (MLAAD frame) / −0.140 (raw
   frame), i.e. negative.** (`AXIS.cos_MLAAD_ASVspoof21_mlaad_frame`,
   `AXIS.cos_MLAAD_ASVspoof21_raw_frame`)
2. **ITW "fusion" 0.363→0.292 → corrected: 0.292 (artifact 0.2917) is the axis-alone EER; true
   fusion EER is 0.306 (artifact 0.3123), i.e. fusion hurts.**
   (`ITW_FUSION.axis_alone_EER_speaker_disjoint`, `ITW_FUSION.true_fusion_EER_speaker_disjoint`)
3. **"Zero-training-cost fusion" domain-shift gains → corrected: supervised LDA-alone beats
   fused** — ITW 0.101 (LDA alone) vs 0.161 (fused, bona-leaked) / 0.193 (fused, strict
   speaker-disjoint); MLAAD 0.100 (LDA alone) vs 0.116 (fused).
   (`DOMAIN_SHIFT.ITW_LDA_alone_EER`, `DOMAIN_SHIFT.ITW_fused_EER`,
   `DOMAIN_SHIFT.ITW_strict_speaker_disjoint_fused_EER`, `DOMAIN_SHIFT.MLAAD_LDA_alone_EER`,
   `DOMAIN_SHIFT.MLAAD_fused_EER`)
4. **Numerical/labeling errors, corrected:**
   - table2 "+0.316" → triplet-model LOSO R² (`CORE_LAW.triplet_LOSO_R2` = 0.3165); sd_along
     alone LOSO R² = 0.2766 (`CORE_LAW.sd_along_LOSO_R2`).
   - "true gated C-effect −0.0036" → correct value is **−0.0023**
     (`CAUSAL.gated_C_effect_dAUC_correct`); −0.0036 is actually the iso_0.7-vs-shift_0.7
     contrast.
   - "58 speakers" → **29** speakers analyzed for the hardness law (49 spoof speakers present
     in the subset). (`ITW_SPEAKERS.speakers_analyzed_for_hardness_law`,
     `ITW_SPEAKERS.spoof_speakers_in_subset`)
   - "13 interventions" → **22 conditions** (21 non-baseline). (`CAUSAL.n_conditions_correct`)
   - Figure 1 left panel → should show sd_along's own R²=0.277 (`CORE_LAW.sd_along_LOSO_R2`),
     not s_along's R²=0.020, under the sd_along scatter.
   - Figure 1 middle panel → must not mix I4's ρ=−0.714 with J4's P3 axis label; use two
     clearly separate, correctly labeled panels.
   - Tally confirmed: 8 verified / 6 mislabeled / 5 misleading / 2 contradicted (see
     CORRECTIONS.md verdict cross-check).
5. **I2 feature battery → corrected disclosure: 0/30 features survive BH-FDR q<0.05; best is
   `rog_L0` at q=0.073–0.078.** (`I2_BATTERY.n_fdr_survivors_q05` = 0,
   `I2_BATTERY.best_q_fdr` = 0.0727, `I2_BATTERY.best_feature` = rog_L0)
6. **AASIST: sd_along replicates rank-order only.** ρ=+0.3495 (p=0.0058) replicates; LOSO-R²
   form fails: r²=−0.0324, permutation p=0.1289. Claim must read "rank-order replication," not
   "the law replicates." (`AASIST_FT.sd_along_spearman_rho`, `AASIST_FT.sd_along_LOSO_R2`,
   `AASIST_FT.sd_along_LOSO_R2_perm_p`)
7. **Hardness reliability ceiling + min-utts sensitivity → must be reported alongside every
   R²:** split-half reliability ceiling R²≈0.8629 (`HARDNESS.reliability_full_R2_ceiling`);
   sd_along's 0.277 captures 32.06% of explainable variance
   (`HARDNESS.pct_of_explainable_variance_captured`); min-utts sensitivity ρ=0.5975→0.6421→0.688
   at ≥8/12/16 utts (`MIN_UTTS.rho_at_8`, `MIN_UTTS.rho_at_12`, `MIN_UTTS.rho_at_16`).
8. **Residual causal-C effect (iso_0.7, p=0.024) → CONTRADICTED, drop.** Per-seed deltas
   [−0.0029, +0.00008, −0.0042] disagree in sign; seed-level p=0.2051, not significant.
   (`CAUSAL.iso_0.7_vs_baseline_seed_deltas`, `CAUSAL.iso_0.7_vs_baseline_seed_level_p`,
   `CAUSAL.iso_0.7_seed_consistent` = False)

## Files produced

- `experiments/paper_regen/regenerate_numbers.py` — the regeneration script (CPU-only, imports
  `audit_common` from `experiments/axis_audits/`).
- `experiments/paper_regen/regenerated_numbers.json` — 79 top-level keys of "KEY = VALUE"
  headline numbers, plus `_mismatches_vs_audit_json` (empty) and `_missing_artifacts` (empty).
- `experiments/paper_regen/CORRECTIONS.md` — the 17-row corrections table described above.
- `experiments/paper_regen/REGEN_REPORT.md` — this file.

No existing files were modified; nothing outside `experiments/paper_regen/` was written; no git
commands were run; no packages were installed; no GPU/torch was used.
