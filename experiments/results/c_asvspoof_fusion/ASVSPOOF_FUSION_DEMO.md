# Workstream C — ASVspoof-internal-axis fusion demo (WavLM-GAT / robust_goat)

**Status: fusion/robustness DEMO, not a second test of the system-hardness law.**
n_systems = 13 (ASVspoof attacks A07–A19) is far too few to power a LOSO-R² /
permutation test of the sd_along→hardness law that was established and
stress-tested at n=61 MLAAD systems (verdict P1). This experiment only asks
the narrower, adequately-powered question I7 asked on MLAAD: *does fusing a
frozen, corpus-internal linear axis projection with the fine-tuned detector's
logit reduce EER, under a leakage-free (system-disjoint, train-fold-only
fitting) evaluation?* All fusion metrics are utterance-level (n=1600), which
is well powered; nothing here should be read as a system-level regression.

All numbers below come from a real run of
`experiments/results/c_asvspoof_fusion/asvspoof_internal_fusion.py`
(CPU-only, no torch/GPU, no new package installs, no network access beyond
one offline read of an already-cached HF dataset's metadata column — see
"Data provenance" below).

## Detector: robust_goat = WavLM-GAT, ASVspoof training domain

`robust_goat.ckpt` (+ seeds `robust_goat_seed3.ckpt`, `robust_goat_seed7.ckpt`)
is the **same architecture** as the MLAAD detector used in the I7 reference
fusion (`mlaad_robust_goat*.ckpt`): a frozen WavLM-L12 encoder feeding a
phoneme-GAT head ("WavLM-GAT"). The only difference is the **training
domain** — robust_goat is trained on ASVspoof/robust data, the I7 detector is
trained on MLAAD. This is therefore the *same-architecture / different-
training-domain* cell of the verdict's cross-detector-agreement matrix
(`COMPREHENSIVE_VERDICT.md` P4 / audit7: `aasist_zs~robust_goat` is flagged
there as the second same-domain — ASVspoof/ASVspoof — pair; robust_goat
itself is WavLM-GAT, not a distinct architecture from the MLAAD detector).
AASIST~RobustGoat is the cross-*architecture* pair in that matrix; this demo
does not touch AASIST.

## Corpus and data provenance

**Corpus**: ASVspoof 2019 LA eval, attacks A07–A19 (13 systems) — this is
robust_goat's native training/eval domain. 1600 utterances (800 bonafide,
800 spoof), the exact selection used by
`experiments/scripts/i1_geometry_causal_decomp.py` (seed 42) and replicated
by `experiments/scripts/i4_asvspoof_position.py`.

**Why not `experiments/results/j4_asvspoof21/`?** J4's cached `logits.npz`
scores ASVspoof-2021-LA-clean audio with the **MLAAD-trained** detector
(`mlaad_robust_goat*.ckpt`) — the opposite cross-domain direction from what
this workstream needs. Similarly, `j2_prospective/robustgoat_mlaad_logits.npz`
is robust_goat scored *out-of-domain* on MLAAD systems. Neither cache pairs
the ASVspoof-**trained** robust_goat detector with ASVspoof-corpus embeddings
at the per-utterance level.

**What we used instead**: `experiments/results/i4_asvspoof_position/features.npz`
(`X12`: frozen WavLM-L12 mean-pooled embeddings, 1600×768) matched 1:1 with
`experiments/results/i1_geometry_causal_decomp/i1_logits.npz`
(`s1__baseline`/`s3__baseline`/`s7__baseline`: robust_goat's 3-seed
per-utterance logits on the *identical* 1600-utterance selection — i4's
generating script asserts this equality against i1's cached labels). Both
derive from the same seeded selection over the HF dataset
`Bisher/as_vspoof_2019_la`.

**Per-utterance attack-system id** is not itself cached in either .npz, so
the script reconstructs it deterministically and offline: it re-runs the
identical seeded (`np.random.default_rng(42)`) selection logic against the
dataset's `system_id` metadata column only (no audio download, no GPU,
`HF_DATASETS_OFFLINE=1`, using the arrow cache already on disk under
`data/asvspoof_2019_la/` and the project venv's existing `datasets` package —
no new packages were installed). The reconstruction is verified byte-exact:
recomputed per-attack hardness matches `experiments/results/i4_asvspoof_position/attack_table.csv`
to max|Δ| = 8.9e-17, and the recomputed label vector matches
`i1_logits.npz`'s cached labels exactly. The resulting attack-id array is
cached to `_asvspoof2019_attacks_cache.npy` so re-runs never need `datasets`
or any data access again.

## Internal axis construction and the rotation check

The natural↔synthetic axis is estimated **internally on ASVspoof** — an
unsupervised centroid difference `w = mean(z(spoof)) − mean(z(bona))` in a
frame standardized on the (training-fold) bonafide pool, unit-normalized.
This is the identical construction used by I3/I4's LOAO axis and I7's fusion
axis; it uses only bona/spoof labels, never any detector output, and in the
fusion folds below it is refit on train-fold data only (never on held-out
attacks).

Documenting the axis-rotation phenomenon (verdict P3 — the natural↔synthetic
axis rotates across corpora, so an MLAAD-fit axis must not be transferred to
ASVspoof):

| comparison | cosine |
|---|---|
| w(ASVspoof-2019-internal) · w(MLAAD-internal), own standardization frames | **−0.078** |
| w(ASVspoof-2019-internal) · w(MLAAD-internal), common MLAAD frame | **−0.085** |
| w(ASVspoof-2019-internal) · w(ASVspoof-2021-internal) | **+0.747** |

The ASVspoof-internal axis is near-zero/negative against the MLAAD axis
(consistent with the verdict's finding that cos(w_MLAAD, w_ASVspoof21) is
negative, not the erroneous +0.36 previously reported), confirming the axis
must be estimated internally rather than transferred. As an internal
consistency check, the two ASVspoof year-corpora's internally-fit axes
(2019 LA vs. the J4 2021-LA-clean subset) agree strongly with each other
(cos ≈ 0.75) despite both rotating away from MLAAD — i.e. "ASVspoof-ness" is
a stable direction across ASVspoof vintages even though it is unrelated to
"MLAAD-ness."

## Fusion protocol

Mirrors `experiments/scripts/i7_axis_fusion.py` exactly:

- 5-fold, **attack-disjoint** splits over the 13 systems; bonafide utterances
  split into independent random folds.
- Per fold: standardization stats and the axis fit on the train fold only;
  `lambda ∈ {0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3}` selected by train-fold AUC of
  `z(logit) + lambda·z(axis_projection)`; held-out fold scored with that
  fold's fitted axis/lambda/standardization only (no leakage).
- Repeated for the 3 robust_goat seeds (`s1`, `s3`, `s7`).

## Per-fold ΔEER (fused − detector-alone)

| fold | test attacks | n_test | dEER (s1) | dEER (s3) | dEER (s7) | dEER (mean) | axis-alone EER |
|---|---|---|---|---|---|---|---|
| 0 | A07, A14, A19 | 335 | −0.0003 | +0.0091 | +0.0062 | **+0.0050** | 0.2388 |
| 1 | A10, A16, A17 | 359 | −0.0245 | −0.0062 | −0.0163 | **−0.0157** | 0.2118 |
| 2 | A09, A12, A13 | 365 | −0.0056 | −0.0056 | −0.0056 | **−0.0056** | 0.0118 |
| 3 | A11, A18 | 283 | +0.0247 | +0.0103 | −0.0072 | **+0.0093** | 0.3085 |
| 4 | A08, A15 | 258 | 0.0000 | −0.0051 | −0.0114 | **−0.0055** | 0.0587 |

Mean dEER across folds: **−0.0025** (a small, inconsistent-in-sign reduction
— folds 0 and 3 get slightly *worse* under fusion, folds 1, 2, 4 get
slightly better). This is a materially different picture from I7's MLAAD
result, where dEER was negative in all 5 folds × 3 seeds.

## Headline EERs (out-of-fold, whole-corpus)

| scorer | EER (mean over 3 seeds) | per-seed EER |
|---|---|---|
| **detector-alone** (robust_goat, WavLM-GAT) | **0.0781** | s1=0.0713, s3=0.0919, s7=0.0713 |
| **axis-alone** (frozen internal-axis projection) | **0.1725** | — (axis has no seed) |
| **fused** (detector + λ·axis) | **0.0777** | s1=0.0750, s3=0.0837, s7=0.0744 |

Paired utterance-level bootstrap (B=2000) on mean dEER (fused − detector):
**dEER = −0.0004, 95% CI [−0.0075, +0.0062], p = 0.80** — indistinguishable
from zero.

## Interpretation

Unlike the MLAAD result (I7: 0.272 → 0.163, a large, seed-consistent,
all-folds-negative gain — verdict P5), **fusion does essentially nothing for
robust_goat on its native ASVspoof domain**: detector-alone EER (0.078) and
fused EER (0.078) are statistically indistinguishable, and the per-fold sign
of ΔEER is inconsistent. The most likely explanation is a ceiling effect:
robust_goat is already highly accurate in-domain (EER ≈ 0.07–0.09, AUC ≈
0.97–0.98), leaving little room for a much weaker frozen linear probe
(axis-alone EER 0.17, roughly 2× worse than the detector) to add
information the fine-tuned GAT head hasn't already captured. This is the
opposite regime from I7's MLAAD/domain-shifted settings, where the detector
itself is comparatively weak (EER 0.27–0.49) and the frozen axis captures
real complementary signal. The honest reading is: **axis fusion helps when
the base detector is far from ceiling and/or evaluated under domain shift;
it does not reliably help an already-strong in-domain detector.**

## Scoping caveats (read before citing any number above)

1. **Demo, not a law-test.** n_systems = 13 is underpowered for any
   system-level regression (sd_along → hardness); this script does not
   attempt one. All EER/ΔEER numbers here are utterance-level (n=1600),
   which is adequately powered for the fusion question asked.
2. **Detector is WavLM-GAT, ASVspoof-trained** — same architecture as the
   MLAAD detector in I7, different training domain. Do not describe this as
   a cross-architecture replication.
3. **Corpus is ASVspoof 2019 LA**, not 2021 — chosen because it is the only
   cache pairing ASVspoof-*trained* robust_goat's per-utterance logits with
   matched frozen-WavLM embeddings. A secondary cosine check against the
   J4 ASVspoof-2021-LA-clean embeddings shows the two ASVspoof vintages'
   internal axes agree closely (cos ≈ 0.75), so this substitution does not
   appear to change the qualitative rotation finding.
4. **Attack-id reconstruction required one offline metadata read** of an
   already-fully-cached HuggingFace dataset (`Bisher/as_vspoof_2019_la`,
   `system_id` column only) using the project venv's pre-existing `datasets`
   package, in `HF_DATASETS_OFFLINE=1` mode — no network access, no audio
   download, no GPU, no new packages. This was verified byte-exact against
   the pre-existing `i4_asvspoof_position/attack_table.csv` cache
   (max|Δhard| = 8.9e-17) before being trusted, and is itself cached
   (`_asvspoof2019_attacks_cache.npy`) so subsequent runs need no data
   access at all.
5. **No fabricated numbers.** Every figure in this document was produced by
   `asvspoof_internal_fusion.py` in this same directory and is reproducible
   by re-running it (see `asvspoof_fusion_results.json` and
   `asvspoof_fusion_per_fold.csv` for the machine-readable versions).

## Files in this directory

- `asvspoof_internal_fusion.py` — the CPU-only script that produces every
  number in this document.
- `asvspoof_fusion_results.json` — full results (headline EERs, per-fold
  ΔEER, cosines, lambdas, provenance).
- `asvspoof_fusion_per_fold.csv` — per-fold, per-seed EER/ΔEER table.
- `asvspoof_fusion_scores.npz` — raw out-of-fold scores (labels, attacks,
  axis projection, fused/detector logits per seed) for independent re-audit.
- `_asvspoof2019_attacks_cache.npy` — cached per-utterance attack-id array
  (see "Data provenance" above).
