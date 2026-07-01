# MOSS@COLM Submission Plan — Geometric-Axis Hardness Law

## Context
The goal is a workshop-scale (MOSS@COLM) submission built on the geometric-axis
hardness work in `final_outputs2` / `experiments/results`. The user wants to lead
with the i7 MLAAD axis-fusion result (EER 0.272 → 0.163, "zero training cost") and
believes it is conclusive given the BH-FDR survival and the 10-audit red-team battery
(`experiments/axis_audits/audits_outputs/COMPREHENSIVE_VERDICT.md`).

The audit supports a **narrower and stronger** framing than "fusion is the result":
- The publishable core is the **law**: per-system spread along the corpus-internal
  natural↔synthetic axis (`sd_along`) in frozen WavLM-L12 predicts detection hardness.
  It is the only test (of 83) that survives *global* BH-FDR (q < 0.001). [Audit 1,2,9]
- i7 MLAAD fusion is **clean** and a fair "zero-cost" demo — but only on MLAAD. [Audit 5]
- The axis **rotates** across recording domains (cross-corpus cos at chance floor;
  MLAAD↔ASVspoof21 negative) — a finding, not a bug. Axis must be estimated
  corpus-internally. [Audit 4]
- The cross-domain "zero-training fusion" gains are **misattributed**; supervised
  LDA-alone beats fusion, cross-domain fusion hurts, ITW had speaker leakage. [Audit 5]

Intended outcome: a submittable paper whose spine is the law + axis-rotation, with i7
as the actionable MLAAD demo, all flagged numbers corrected, and (ideally) one
genuinely held-out confirmation landed.

## Venue fit — MOSS framing (revised)
MOSS = "Methods and Opportunities at Small Scale" (ICML 2025 edition; scoped at ≤1 GPU):
methodology of small-scale, training-free, compute-limited studies that yield
generalizable insight. **Caveat to resolve first:** a MOSS edition *at COLM 2026* is
unconfirmed — the only MOSS found is ICML. The nearest COLM 2026 analog is "Scientific
Understanding of Foundation Models" (science-ai-2026.github.io), **deadline June 23 2026,
already passed**. Confirm the actual target venue + deadline before drafting.

**For MOSS, lead with the method/efficiency, carry the law as the mechanism** (this
REVERSES the "lead with the law" steer below, which was calibrated to a general interp
venue):
- The fusion is a textbook MOSS contribution: training-free, frozen-WavLM,
  unsupervised-centroid intervention, EER 0.272→0.163, gains concentrated on the hardest
  quartile (Q4 AUC +0.22), single GPU. Frame it as "training-free / label-light
  adaptation in frozen SSL space," NOT "data efficiency" (it's compute/label efficiency).
- The law is demoted from headline to *justification*: fusion isn't tuning because the
  axis score targets exactly the systems the detector fails on. That law-as-cheap-insight
  is the MOSS sweet spot.
- Takeaway must be the transferable lesson ("corpus-internal geometric axes in frozen SSL
  are a cheap training-free lever; here's where they work and where they rotate away"),
  not "we improved a deepfake detector."
- MOSS reviewers are methodology-sensitive: keep "zero-cost fusion" scoped to MLAAD;
  present axis rotation honestly as *why* cross-domain transfer fails.

## Recommended approach

**Write now. For MOSS: lead with the training-free fusion/efficiency result, with the
law as the mechanism. (For a general interp venue instead: lead with the law.)** The core
result is done and audited; the paper does not depend on new experiments. Run the
held-out confirmation in parallel.

### Workstream A — Mandatory pre-submission corrections (audit §3)
Blocking; reviewers will find these.
- Regenerate every reported number programmatically from cached artifacts.
- Fix the two *contradicted* items: the false "cos(w_MLAAD, w_ASVspoof21) ≈ 0.36"
  (true value negative) and the residual causal-C effect (seeds disagree in sign — drop).
- Fix the six mislabels (speaker count 58→29, "13 interventions"→22 conditions,
  Figure 1 panel mix-ups, table2 "+0.316" triplet-R²-as-sd_along, etc.).
- Remove all "zero-training-cost fusion under domain shift" language; keep "zero-cost"
  only for MLAAD i7. Reframe domain-shift as "supervised lightweight adaptation in
  frozen-SSL space" and report LDA-alone alongside fused.
- Claim AASIST as "rank-order replication" only; do not silently switch to LOSO-R².
- Report the hardness reliability ceiling (0.86) alongside all R²; add the min-utts
  sensitivity row (ρ 0.60→0.64→0.69).

### Workstream B — ITW fusion reanalysis (planned exp a; cleanup, not headline)
`experiments/results/i7_axis_fusion/itw_fusion_test.json` is mislabeled and has no
committed generating script.
- Commit a script that regenerates the artifact from raw scores.
- Make bona folds strictly speaker-disjoint (leakage 0.312→0.193 territory).
- Report: internal-axis-alone (0.292) beats fine-tuned detector; fusion (0.312) *hurts*;
  reframe accordingly. Files: `experiments/axis_audits/audit5_fusion_claims.py`,
  `experiments/axis_audits/audit6_itw_speaker.py`, `experiments/results/i5_itw_transfer/`.

### Workstream C — ASVspoof2019 internal-axis fusion (planned exp b; second-corpus demo)
- Estimate the axis **internally on ASVspoof2019** (do NOT transfer MLAAD's — negative cos).
- Frame as a fusion/robustness demo on robust_goat, NOT as a second test of the
  system-hardness law (only ~6–13 systems → underpowered; see E4).
- Reference `experiments/results/j4_asvspoof21/`, `j5_aasist/`, the i7 fusion scorer.

### Workstream D — Held-out confirmation (highest-leverage follow-up; do in parallel)
Converts the ASVspoof21 "prospective" claim from suggestive (S1) to proven.
- One pre-committed predictor (P3), one genuinely held-out condition (ASVspoof21-DF or
  codec conditions), externally timestamped before scoring.
- Reference `experiments/axis_audits/audit3_asvspoof_prospective.py`,
  `experiments/results/j2_prospective/`, `j3_axis_adaptive/`.

### Optional follow-ups (strengthen, not gate)
- Channel-confound control: partial out layer-0 velocity / channel proxy; show law
  survives (directly answers "not a dataset artifact"). [Audit 6 / S2]
- Merge `s_orth`/`rog12` into one ITW spread factor or run a discriminating experiment.
- Re-power I1 causal contrasts at seed level (n=3 → 5 seeds) or drop them.

## Verification
- A: diff every paper number against a regeneration script's output; zero mismatches.
- B/C: per-fold ΔEER tables, speaker-disjoint splits asserted in code, axis-internal cos
  with MLAAD reported (expected near-zero/negative).
- D: timestamp artifact (git tag / external commit) precedes the scoring run; single
  predictor only.

## Decision (locked)
Pursue **all of A–D** for the strongest submission. Execution order:
1. **D first / in parallel** (longest pole — needs an externally-timestamped pre-commit
   *before* scoring; start the clock immediately so it isn't the bottleneck).
2. **A** (number/label fixes + programmatic regeneration) — unblocks honest writing.
3. **B** (ITW reanalysis cleanup) — protects against the mislabel finding.
4. **C** (ASVspoof2019 internal-axis fusion demo) — second-corpus demonstration.
5. Optional follow-ups (channel-confound control, s_orth/rog12 merge, I1 re-power)
   folded into appendices as time allows.
Draft the paper around the law from the start; slot D's confirmation in when it lands.
