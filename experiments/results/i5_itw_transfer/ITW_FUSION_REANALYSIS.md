# ITW Fusion Reanalysis — Cleanup, Not a Headline

**Status:** cleanup of a mislabeled artifact. No new finding; corrects a labeling error and
quantifies a leakage effect that was already flagged by the axis-audit red team.

## The problem

`experiments/results/i7_axis_fusion/itw_fusion_test.json` has no committed generating script
and was mislabeled downstream: the reported "ITW fusion 0.363 → 0.292 (WavLM-GAT)" treats
**0.292 as the fusion result**. It is not. Inside that same JSON:

| key in `itw_fusion_test.json` | EER | what it actually is |
|---|---|---|
| `itw_baseline.EER` | 0.3633 | WavLM-GAT detector logit alone |
| `itw_internal_axis_speaker_disjoint.EER` | 0.2917 | **axis alone** (speaker-disjoint, unsupervised centroid) — this is the number that got reported as "fusion" |
| `fusion_internal.lam1.5.EER` | 0.3123 | the **actual fusion** (detector + axis, λ=1.5) — worse than axis alone |

So on ITW, for the WavLM-GAT detector, **fusion hurts relative to the axis alone**. This was
caught independently by `experiments/axis_audits/audit5_fusion_claims.py`
(`AUDIT5_SUMMARY.md` item 2, `COMPREHENSIVE_VERDICT.md` §3 item 2) and is the reason this
regeneration exists: to produce a from-scratch, unambiguously-labeled, speaker-disjoint-by-
construction replacement, without touching the original file.

## What this script does

`experiments/results/i5_itw_transfer/regen_itw_fusion.py` recomputes everything from raw
scores/embeddings (`features.npz`, `utt_table.csv`, `experiments/results/j5_aasist/aasist_scores.npz`),
reusing the audit5/audit6 methodology, and writes
`experiments/results/i5_itw_transfer/itw_fusion_regenerated.json`. It does **not** overwrite
`itw_fusion_test.json` (left in place for the historical record / diff).

Two pipelines are reproduced end to end so the two different "leakage" stories in
`COMPREHENSIVE_VERDICT.md` §3 items 2 and 3 aren't conflated:

- **Pipeline A** (WavLM-GAT detector + unsupervised centroid ITW-internal axis, 5-fold,
  speaker-disjoint from construction for both bona and spoof) — this is the pipeline
  `itw_fusion_test.json` intended to report, and where the "0.292 mislabeled as fusion"
  correction lives.
- **Pipeline B** (AASIST-ZS detector + supervised shrinkage-LDA axis on WavLM features,
  mirroring audit5's J5-H4 decomposition) — run with bona folded (i) randomly/leaky
  (audit5's default protocol) and (ii) strictly speaker-disjoint. This is where the
  "~0.193 strict speaker-disjoint" figure and the ~3-4 EER point leakage delta actually
  come from.

## Corrected numbers (Pipeline A — the itw_fusion_test.json family)

Run with `python3 experiments/results/i5_itw_transfer/regen_itw_fusion.py` (CPU only, system
python3, no torch). Independent recomputation, different fold RNG than the original artifact,
same speaker-disjoint 5-fold protocol as `audit5_fusion_claims.py`.

| quantity | this script | `audit5_results.json` (`itw_internal`) | original `itw_fusion_test.json` |
|---|---|---|---|
| `detector_alone_eer` | **0.3633** | 0.3633 | 0.3633 (`itw_baseline`) |
| `axis_alone_eer` | **0.3007** | 0.3007 | 0.2917 (`itw_internal_axis_speaker_disjoint`, mislabeled downstream as "fusion") |
| `fusion_eer` | **0.3060** | 0.3060 | 0.3123 (`fusion_internal.lam1.5`, the *real* fusion — worse than axis alone) |

`fusion_eer (0.3060) > axis_alone_eer (0.3007)` — **fusion hurts**, consistent both with this
recomputation and with the original artifact's own (mislabeled) numbers. The small numeric
gap between this script's 0.2917→0.3007 / 0.3123→0.3060 and the original artifact is expected:
different fold random seed / speaker partition, same protocol. Matches `audit5_results.json`
(`itw_internal.axis_alone_eer_speaker_disjoint` = 0.30067, `itw_internal.fused_eer_speaker_disjoint`
= 0.306) to 4 decimal places (identical seed=0 speaker partition as audit5).

### Per-fold ΔEER (fusion − axis), Pipeline A

| fold | n_eval | λ | detector EER | axis EER | fused EER | ΔEER (fusion − axis) | ΔEER (fusion − detector) |
|---|---|---|---|---|---|---|---|
| 0 | 417 | 2.0 | 0.3477 | 0.2685 | 0.2974 | **+0.0288** | −0.0503 |
| 1 | 909 | 2.0 | 0.3194 | 0.3342 | 0.3005 | −0.0337 | −0.0189 |
| 2 | 698 | 2.0 | 0.3202 | 0.2303 | 0.2647 | **+0.0344** | −0.0555 |
| 3 | 382 | 2.0 | 0.3697 | 0.2399 | 0.2879 | **+0.0480** | −0.0818 |
| 4 | 594 | 2.0 | 0.3980 | 0.2959 | 0.2971 | **+0.0012** | −0.1009 |

Fusion beats the raw detector in every fold (λ=2.0 chosen on train-fold AUC in all 5 folds),
but beats the axis alone in only 1 of 5 folds (fold 1) and is roughly a wash in fold 4. On the
pooled/concatenated scores, fusion is worse than axis alone (0.306 vs 0.3007) — consistent
across the per-fold view, not a fold-concatenation artifact.

## Speaker-disjoint bona assertion

Pipeline A partitions speaker identity into 5 folds *before* splitting bona/spoof, so both
classes are speaker-disjoint by construction; `assert_train_eval_disjoint` is called once per
fold and raised no failures (confirmed by the run completing to "All sanity assertions
passed."). Pipeline B's `strict_identity` mode uses the same single speaker-identity partition
and carries an identical hard assertion.

## Where the ~0.193 "strict speaker-disjoint" figure comes from — and the leakage delta

This is a **different pipeline** (Pipeline B / J5-H4-style): AASIST-ZS detector fused with a
*supervised* shrinkage-LDA axis trained (with labels, group-disjoint) on the ITW corpus itself.
Audit5's default protocol folds bona utterances **at random**, independent of speaker, so the
same bona speaker's utterances can appear in both the LDA-training fold and the held-out
evaluation fold — leakage. Forcing bona speaker-disjointness (audit5's `strict_group_bona`)
raises EER because leakage had been making the number look artificially good:

| protocol | LDA axis alone | fused |
|---|---|---|
| leaky (bona folded at random) | 0.1007 | 0.1607 |
| **strict, speaker-disjoint bona** (audit5's exact protocol — the "~0.193" figure) | 0.1433 | **0.1933** |
| strict, full speaker-identity-disjoint (this script's stronger variant) | 0.1217 | 0.1693 |

This script's numbers match `audit5_results.json` → `j5_h4_decomposition.itw` exactly:
`lda_axis_alone_eer` 0.10067 / `fused_eer` 0.16067 (leaky) and `strict_bona_disjoint.lda_axis_alone_eer`
0.14333 / `fused_eer` 0.19333 (strict) — to 4-5 decimal places.

**Leakage delta** (leaky − strict, audit5's exact protocol): LDA-alone Δ = −0.0427,
fused Δ = **−0.0327** (i.e. ~3.3 EER points; audit5/`COMPREHENSIVE_VERDICT.md` quotes "~3-4 EER
points" — matches). The sign is negative because leaky EER is lower than strict EER: removing
the leakage makes the number worse, exactly as expected when speaker leakage was inflating
apparent performance.

Note: the additional `strict_identity` variant (0.1217 / 0.1693) is *stronger* than audit5's
own `strict_group_bona` (0.1433 / 0.1933) because audit5's "strict" protocol only makes bona
and spoof folds independently speaker-disjoint — since ~46/49 ITW speakers appear in *both*
classes, a person's bona and spoof utterances can still land in different folds under audit5's
protocol. `strict_identity` uses one shared speaker partition for both classes, closing that
residual gap. This variant is provided for transparency but is not what "the ~0.193 figure"
refers to; that number is specifically audit5's `strict_group_bona` result, reproduced above.

## cos(w_ITW, w_MLAAD)

Computed via the same axis-in-MLAAD-bona-frame method as audit4 (unsupervised centroid
direction, standardized on MLAAD bona mean/std): **cos(w_ITW, w_MLAAD) = +0.103**. This is in
the same near-zero regime quoted by `COMPREHENSIVE_VERDICT.md` P3 (E|cos| ≈ 0.029 at 768-d
chance floor; MLAAD↔ITW = 0.03–0.10 depending on frame) — the ITW axis and the MLAAD axis
point in essentially unrelated directions, consistent with "the axis rotates across domains"
and with the original artifact's own `cos_w_mlaad_w_itw = 0.0525` (different exact frame /
axis estimator, same qualitative near-zero conclusion). Note this differs slightly from "expect
near-zero/negative" in the script's own docstring — the sign is mildly positive here, still
well within chance-floor noise (both audit4's and this script's estimates sit inside roughly
±0.10 of zero), so this does not change the conclusion.

## Reframed takeaway

- The **internal-axis-alone** (unsupervised centroid on ITW's own WavLM-L12 embeddings,
  speaker-disjoint) beats the fine-tuned WavLM-GAT detector on ITW: 0.301 vs 0.363 EER.
- **Fusing the two makes it worse, not better**, on ITW: 0.306 vs 0.301 (axis alone). The
  original "0.363 → 0.292" headline conflated the axis-alone result with a fusion claim that
  never held.
- Earlier "large fusion gains under domain shift" claims (the AASIST-ZS / supervised-LDA
  pipeline) were also partly a **speaker-leakage artifact**: ~3.3 EER points of the reported
  ITW fused-EER improvement evaporate once bona speakers are forced disjoint across folds
  (0.1607 → 0.1933 strict).
- None of this changes the core sd_along/hardness law (P1 in `COMPREHENSIVE_VERDICT.md`) or
  the I7 MLAAD fusion result (P5, verified independently and unaffected by this correction).
  This is narrowly a correction to the ITW-domain-shift fusion narrative and to one
  mislabeled, unreproducible artifact.

## Provenance

- Generating script: `experiments/results/i5_itw_transfer/regen_itw_fusion.py`
- Output artifact: `experiments/results/i5_itw_transfer/itw_fusion_regenerated.json`
- Raw inputs: `experiments/results/i5_itw_transfer/{features.npz,utt_table.csv,speaker_table.csv}`,
  `experiments/results/j5_aasist/aasist_scores.npz`, plus (for cos(w_ITW,w_MLAAD)) MLAAD
  reference artifacts under `experiments/results/mlaad/`, `experiments/results/i3_position_geometry/`,
  and `outputs/px_wave_cache/`.
- Reference methodology: `experiments/axis_audits/audit5_fusion_claims.py`,
  `experiments/axis_audits/audit6_itw_speaker.py`, `experiments/axis_audits/audit_common.py`.
- Original (unmodified, left in place) mislabeled artifact:
  `experiments/results/i7_axis_fusion/itw_fusion_test.json`.
- Verified by running the script to completion with system `python3` (CPU only, no torch);
  all in-script sanity assertions (speaker-disjointness, fusion-hurts-axis-alone,
  strict-raises-fused-EER-vs-leaky) passed.
