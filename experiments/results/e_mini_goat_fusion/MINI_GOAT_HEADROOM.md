# Workstream E — Does the axis-fusion lever pay off when the detector is weak?

**Verdict: YES.** A deliberately weak WavLM-GAT detector ("mini_goat", trained
on only 500 ASVspoof files) gets a large, statistically decisive EER
reduction from the ASVspoof-internal natural↔synthetic axis fusion
(ΔEER = **−0.0300**, 95% CI [−0.0456, −0.0125], **p = 0.0000**). The
near-ceiling `robust_goat` detector (trained the normal way) gets essentially
nothing from the identical fusion recipe (ΔEER = +0.0044, CI
[−0.0075, +0.0156], p = 0.559, consistent with Workstream C's published
ΔEER ≈ −0.0004, p = 0.798). This is exactly the headroom-dependent pattern
the hypothesis predicted: **the axis is a free lever only when the detector
has room to improve.**

All numbers below come from real training + real scoring runs on the GPU
(one A10, ~1 minute of training, ~2 minutes of scoring/fusion). No numbers
were fabricated; the validation gate (item 2) is the check that the scoring
pipeline itself is trustworthy before trusting mini_goat's numbers.

---

## 1. How weak is mini_goat?

`mini_goat` is the identical WavLM-GAT architecture as `robust_goat`
(`Phoneme_GAT_lit`, `cfg.PhonemeGAT = {backbone: wavlm, use_GAT: True,
n_edges: 10, use_aug: True, use_pool: True, use_clip: True}`), trained with
the same optimizer/loss recipe as `experiments/scripts/train_mlaad_regular.py`.
The only manipulation was training data volume and epoch budget:

| | mini_goat | robust_goat |
|---|---|---|
| Train files | 500 (250 bonafide + 250 spoof), ASVspoof `train` split (A01–A06) | ~15,000 (full ASVspoof protocol) |
| Epochs | 4 | 8 (from checkpoint metadata) |
| Wall-clock | **60 seconds** | (not re-measured here) |
| Val set | 150 utts (75+75), ASVspoof `validation` split (A01–A06) | — |

Val-EER by epoch (`training_logs/training_curves.csv`):

| epoch | train_loss | val_loss | val_eer | val_auc |
|---|---|---|---|---|
| 0 | 0.770 | 0.532 | 0.1333 | 0.9335 |
| 1 | 0.435 | 0.509 | 0.1200 | 0.9502 |
| **2** | 0.447 | 0.372 | **0.0933 (best)** | 0.9422 |
| 3 | 0.261 | 0.905 | 0.1600 | 0.9004 |

Best val-EER 0.093 at epoch 2; the final (epoch-3) checkpoint saved as
`models/mini_goat.ckpt` had drifted back up to val-EER 0.160 (mild
overfitting on 500 files — expected and on-thesis; a separate "best"
checkpoint is also saved to `experiments/checkpoints/mini_goat-best-epoch=02-val-eer=0.0933.ckpt`).
On the actual 1600-utterance A07–A19 eval set (unseen attacks, the same
selection robust_goat is scored on), mini_goat's **detector-alone EER is
0.1438** — roughly double robust_goat's 0.071–0.078. This is a real,
substantially-degraded-but-not-degenerate detector: exactly the "has
headroom" regime the hypothesis needs.

Training data is utterance- **and** attack-disjoint from the eval set by
construction: mini_goat trains only on the HF `train` split (attacks
A01–A06 + bonafide), and is scored on the HF `test` split (attacks A07–A19 +
bonafide) — the standard ASVspoof-2019-LA unseen-attack protocol, verified
programmatically in `prepare_mini_goat_data.py` (`assert train_attacks.isdisjoint(eval_attacks)`).

## 2. Validation gate: does this scoring path reproduce robust_goat's published i1 baseline?

`score_and_fuse_mini_goat.py` re-implements i1's exact data selection
(`np.random.default_rng(42)`, same `EVAL_ATTACKS = A07..A19`, same
800 spoof + 800 bona `sel`) and scoring call (`Phoneme_GAT.__call__(wav,
num_frames, use_aug=False, stage="val")` on the underlying `lit.model`, not
the Lightning wrapper). Before trusting any mini_goat number, `robust_goat.ckpt`
was rescored through this exact path and checked against the cached
`i1_geometry_causal_decomp/i1_logits.npz` (`s1__baseline`):

| metric | this run | i1 cache | diff |
|---|---|---|---|
| EER | 0.07125 | 0.07125 | **0.0 (exact)** |
| AUC | 0.9822 | 0.9805 | 0.0018 |
| Pearson corr (per-utt logits) | 0.9577 | — | — |

**Gate: PASS** (criteria: `|ΔEER| ≤ 0.01` and `|ΔAUC| ≤ 0.01`; both satisfied,
EER matched to 4 decimal places).

One finding worth flagging explicitly: per-utterance logit correlation is
only ~0.96, and some individual utterances' logits differ substantially
(max |Δlogit| = 5.66) between the two runs of the *same checkpoint on the
same utterances*. Root cause (found by reading `phoneme_GAT/modules.py`):
`Phoneme_GAT.__call__` applies SpecAugment-style time masking
(`_mask_hidden_states`, `mask_time_prob=0.05`) **unconditionally** — it is
not gated by `stage == "train"` or `model.training` (lines 327–355, 546,
581). This makes the forward pass intrinsically stochastic even in eval
mode: a fresh process draws different random masks, so bit-identical or
even highly-correlated per-utterance logits across independent runs should
**not** be expected — this affects `i1_geometry_causal_decomp.py`'s own
baseline condition equally. EER/AUC (rank-based, aggregate) are therefore
the correct reproduction criteria, and they pass cleanly. This is reported
for transparency, not treated as a gate failure.

## 3. Head-to-head fusion comparison

Fusion method is Workstream C's exact recipe
(`experiments/results/c_asvspoof_fusion/asvspoof_internal_fusion.py`),
re-run with mini_goat's and a freshly-rescored robust_goat's logits as the
detector score, both scored through the identical pipeline above:

- Axis: unsupervised centroid difference `w = mean(z(spoof_trainfold)) −
  mean(z(bona_trainfold))`, unit-normalized, standardized on train-fold
  bonafide only. Estimated on the same frozen WavLM-L12 embeddings
  (`i4_asvspoof_position/features.npz`, X12) for both detectors — the axis
  itself does not depend on which detector is being fused.
- 5-fold, attack(system)-disjoint splits over the 13 A07–A19 systems (bona
  split into independent random folds).
- λ ∈ {0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3} selected by train-fold AUC.
- Bootstrap: paired-utterance bootstrap, B = 2000, on mean ΔEER (fused − detector-alone).

| detector | detector-alone EER | axis-alone EER | fused EER | ΔEER | 95% CI | p |
|---|---|---|---|---|---|---|
| **mini_goat** (weak, 500 files) | **0.1438** | 0.1725 | **0.1138** | **−0.0300** | [−0.0456, −0.0125] | **0.0000** |
| robust_goat (regated, this run) | 0.0713 | 0.1725 | 0.0756 | +0.0044 | [−0.0075, +0.0156] | 0.559 |
| robust_goat (Workstream C, published, 3-seed mean) | 0.0781 | 0.1725 | 0.0777 | −0.0004 | [−0.0075, +0.0062] | 0.798 |

(Full table: `headline_comparison.csv`.)

The axis-alone EER (0.1725) is identical across rows because it is the same
axis/embeddings evaluated on the same 1600 utterances — only the detector
being fused changes.

### Per-fold ΔEER (fused − detector-alone)

mini_goat (`mini_goat_fusion_per_fold.csv`): **negative in all 5 folds** —
fusion helps consistently, not just on average.

| fold | test systems | λ | EER detector | EER fused | ΔEER |
|---|---|---|---|---|---|
| 0 | A07,A14,A19 | 1.00 | 0.2239 | 0.2063 | −0.0177 |
| 1 | A10,A16,A17 | 1.00 | 0.1003 | 0.0946 | −0.0056 |
| 2 | A09,A12,A13 | 0.75 | 0.1123 | 0.0303 | **−0.0821** |
| 3 | A11,A18 | 0.75 | 0.2335 | 0.2119 | −0.0216 |
| 4 | A08,A15 | 1.00 | 0.1237 | 0.0454 | **−0.0783** |

robust_goat regated (`robust_goat_regate_fusion_per_fold.csv`): mixed sign,
consistent with the near-zero aggregate.

| fold | test systems | λ | EER detector | EER fused | ΔEER |
|---|---|---|---|---|---|
| 0 | A07,A14,A19 | 0.25 | 0.0986 | 0.1074 | +0.0088 |
| 1 | A10,A16,A17 | 0.25 | 0.0727 | 0.0332 | −0.0395 |
| 2 | A09,A12,A13 | 0.25 | 0.0111 | 0.0000 | −0.0111 |
| 3 | A11,A18 | 0.25 | 0.0606 | 0.0813 | +0.0206 |
| 4 | A08,A15 | 0.25 | 0.0423 | 0.0196 | −0.0227 |

(Combined: `per_fold_comparison.csv`.) Note also that the λ the fold-level
AUC search picks for mini_goat is consistently large (0.75–1.0, i.e. the
axis is weighted comparably to the detector logit itself), vs. robust_goat's
consistently small λ = 0.25 across all 5 folds — the fusion procedure itself
"discovers" that the axis is more useful when the detector is weaker.

## 4. Interpretation

This is a clean confirmation of the headroom hypothesis on ASVspoof itself,
using the same detector architecture, the same corpus, the same internal
axis construction, and the same I7-style fold protocol as Workstream C's
null result on `robust_goat`. The only thing that changed is how much
training signal the detector was given (500 files / 4 epochs vs. the full
protocol). With headroom, fusing a frozen, training-free, unsupervised
natural↔synthetic axis into the detector's logit cuts EER by roughly a
fifth (0.144 → 0.114, a 21% relative reduction) with a bootstrap p-value of
effectively zero and a 95% CI that excludes zero by a wide margin, and the
improvement holds in every one of the 5 attack-disjoint folds. Once the
detector is trained to near-ceiling performance on its own (robust_goat,
EER ~0.07–0.08), the same axis with the same construction and the same
fold protocol does nothing distinguishable from noise (CI straddles zero,
p ≈ 0.56–0.80 across two independent robust_goat scoring runs). Taken
together with Workstream C's original ASVspoof result and the MLAAD I7
result (P5: EER 0.272 → 0.163, where the base detector was also far from
ceiling), the evidence is consistent: **the axis-fusion lever's payoff is
inversely related to how much headroom the underlying detector has left**
— it is a genuinely useful free correction for weak/undertrained detectors
and a no-op (neither harmful nor helpful) for detectors that are already
near their ceiling on that corpus.

---

## Provenance / how to reproduce

1. `prepare_mini_goat_data.py` — builds the 500-file balanced train set
   (seed=42, HF `Bisher/as_vspoof_2019_la` `train` split, attacks A01–A06)
   and 150-file balanced val set (`validation` split, also A01–A06) as
   preprocessed 48000-sample (3 s @ 16 kHz) `.pt` tensors + `train.json`/
   `val.json`, under `experiments/data/mini_goat_processed/`.
2. `train_mini_goat.py` — trains `Phoneme_GAT_lit` (identical cfg to
   `train_mlaad_regular.py` / robust_goat) for 4 epochs, batch size 10, no
   reverb augmentation. Saves `models/mini_goat.ckpt` (final epoch) and
   `experiments/checkpoints/mini_goat-best-epoch=02-val-eer=0.0933.ckpt`
   (best-val-EER checkpoint) + `training_logs/` (curves, hyperparameters,
   final metrics).
3. `score_and_fuse_mini_goat.py` — reproduces i1's exact 1600-utterance
   selection, validates the scoring path against `i1_logits.npz`, scores
   both `mini_goat.ckpt` and `robust_goat.ckpt`, and runs Workstream C's
   exact fusion recipe (reusing `i4_asvspoof_position/features.npz` X12
   embeddings for the axis) for both detectors. Outputs:
   `mini_goat_logits.npz`, `robust_goat_regate_logits.npz`,
   `mini_goat_fusion_results.json`, `mini_goat_fusion_per_fold.csv`,
   `robust_goat_regate_fusion_per_fold.csv`, `per_fold_comparison.csv`,
   `headline_comparison.csv`, `mini_goat_fusion_scores.npz`.

Environment: venv python (`datasets` 2.18, `torch` 2.6+cu124,
`transformers` 4.36.2, `pytorch_lightning`), one A10 GPU, `HF_DATASETS_OFFLINE=1`
(all data/model weights already cached locally, no network access, no
package installs). Total wall-clock: ~60 s training + ~3 min scoring/fusion.
