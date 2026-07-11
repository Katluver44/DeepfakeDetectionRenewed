# Rerun 2026-07-11 — MANIFEST

Task 2: rerun the laughing-augmentation (D1) experiment with the freshly
downloaded checkpoint and look for changes in WavLM space. All commands run
from `laughsmi/` unless noted.

## 1. Staged data

### `data/eval_asv19/` — ASVspoof2019-LA test, 150 bona-fide + 150 spoof
- Script: `rerun_2026/stage_eval_asv19.py` (wraps `Bisher/ASVspoof_2019_LA`
  parquet, columns confirmed by direct inspection: `speaker_id`,
  `audio_file_name`, `audio` (bytes+path), `system_id` (`-`=bona,
  `Axx`=attack), `key` (0=bona-fide, 1=spoof, VERIFIED by cross-tab against
  system_id)).
- Command: `python rerun_2026/stage_eval_asv19.py`
- Output: `data/eval_asv19/meta.csv` (file,speaker,label), `data/eval_asv19/wavs/*.wav`
  (mono 16kHz), `data/eval_asv19/attack_ids.csv` (file,system_id).
- Seed 20260710, matches prior D1 run's sampling seed.

### `data/laugh_bank_bark/` — 60 Bark synthetic laughter clips
- Insert source used: **Bark** (the faithful reproduction of the original D1
  setup, not the real-laughter fallback). `suno-bark` (PyPI) was installed;
  the packaged Bark checkpoints are old numpy-pickled `.th` files that fail
  under torch>=2.6's default `weights_only=True` `torch.load` — worked around
  with a thin wrapper (`rerun_2026/run_generate_laugh_bank.py`) that patches
  `torch.load` to default `weights_only=False` before Bark loads anything
  (this is a benign torch-version compat shim for a package we deliberately
  installed, not arbitrary code execution).
- Command:
  ```
  export SUNO_USE_SMALL_MODELS=1
  python rerun_2026/run_generate_laugh_bank.py --n-per-preset 6
  ```
- Output: `data/laugh_bank_bark/*.wav` (60 files, 10 speaker presets x 6 laughter
  prompt variants), `data/laugh_bank_bark/manifest.csv`. Generation took ~4m45s
  on the L4 GPU. Log: `rerun_2026/logs/generate_laugh_bank.log`.

## 2. Augmentation + scoring rerun

```
python scripts/augment_laughter.py --src data/eval_asv19 --out data/eval_asv19_aug \
    --laugh-dirs data/laugh_bank_bark --frac 0.7 --seed 20260710
# -> 105/150 spoof files augmented (70%)

python scripts/score_itw_detector.py --audio_dir data/eval_asv19 \
    --meta data/eval_asv19/meta.csv --ckpt ../models/asv19-wavlm-gat-full.ckpt \
    --out rerun_2026/base_asv19.csv

python scripts/score_itw_detector.py --audio_dir data/eval_asv19_aug \
    --meta data/eval_asv19_aug/meta.csv --ckpt ../models/asv19-wavlm-gat-full.ckpt \
    --out rerun_2026/aug_asv19.csv
```
Outputs: `rerun_2026/base_asv19.csv`, `rerun_2026/aug_asv19.csv` (300 rows each,
file_id,label,speaker,score,logit). Logs: `rerun_2026/logs/score_base.log`,
`rerun_2026/logs/score_aug.log`.

## 3. Evasion table (D1 reproduction)

- Script: `rerun_2026/d1_analysis_rerun.py` (verbatim pairing/statistics logic
  of `scripts/d1_analysis.py::analyze()`, pointed at rerun paths).
- Command: `python rerun_2026/d1_analysis_rerun.py`
- Output: `rerun_2026/tables/table_d1_rerun.csv`

**Result (n=105 augmented spoof files) vs prior (RESULTS.md Table D1):**

| metric | prior (orig D1) | rerun 2026-07-11 |
|---|---|---|
| clean EER | 5.33% | 6.00% |
| base spoof score (mean) | 0.899 | 0.8886 |
| aug spoof score (mean) | 0.604 | 0.6249 |
| Δ (aug − base) | −0.295 | −0.2637 (median −0.149) |
| Wilcoxon p | ≈0 | 0.0 |
| rank-biserial | −0.78 (large) | −0.682 (large) |
| evasion rate | 0.20 | 0.257 |
| strongest position | start (−0.38) | mid (−0.323), then start (−0.242), end (−0.216) |

Effect direction, magnitude, and significance all reproduce cleanly with the
freshly downloaded checkpoint and a freshly generated Bark laughter bank
(different seed draws for both the eval subsample and the laughter clips).
The insert-position ranking differs slightly (mid strongest here vs start in
the original) — plausibly noise given n=105 and only 60 available insert
clips split ~35/pos.

## 4. WavLM-space analysis

### Embeddings
- Script: `rerun_2026/extract_detector_wavlm.py`
- Representation: the DETECTOR's own frozen WavLM-base backbone
  (`lit.model.transformer_in_phoneme_model`, a `transformers.WavLMModel`
  inside the checkpoint), hidden_states layers **0, 9, 12**, mean-pooled over
  time, 768-d, **raw/unstandardized**. Audio preprocessed identically to
  scoring: mono 16kHz, then `crop_or_tile_center` to exactly 3s (48000
  samples) BEFORE running WavLM, so the embedding matches exactly what the
  detector scored.
- Command:
  ```
  python rerun_2026/extract_detector_wavlm.py \
    --ckpt ../models/asv19-wavlm-gat-full.ckpt \
    --sets base=data/eval_asv19 aug=data/eval_asv19_aug \
    --attack-ids data/eval_asv19/attack_ids.csv \
    --manifest-aug data/eval_asv19_aug/manifest.csv \
    --out-npz rerun_2026/embeddings/wavlm_base_embeddings.npz \
    --out-meta rerun_2026/embeddings/wavlm_base_meta.csv
  ```
- Output: `rerun_2026/embeddings/wavlm_base_embeddings.npz` (keys `layer0`,
  `layer9`, `layer12`, each `(600, 768)` float32), `rerun_2026/embeddings/wavlm_base_meta.csv`
  (600 rows, row-aligned with the npz). Full schema + reuse recipe:
  `rerun_2026/embeddings/README.md`.
- Note: an initial bug in the manifest lookup (keyed by full relpath instead
  of basename) silently produced 0 `augmented==1` rows in the meta; fixed in
  `extract_detector_wavlm.py::read_manifest` and re-run before any downstream
  analysis used the file.
- Secondary WavLM-Large L9/L12 (`scripts/extract_wavlm_features.py`) was
  **not** run in this pass — the detector's own backbone embeddings were
  judged the decision-relevant representation and sufficient to answer the
  task; see embeddings/README.md for how to extend if needed.

### Movement analysis
- Script: `rerun_2026/wavlm_space_analysis.py`
- Command: `python rerun_2026/wavlm_space_analysis.py`
- Axis: `w = centroid(spoof_base) − centroid(bona_base)`, unit-normalized, in
  layer-12 embeddings **standardized per-coordinate by the bona-fide (base)
  mean/SD** — same convention as `experiments/axis_audits/audit_common.py::loso_axis_features`.
  (Raw/unstandardized geometry was checked first and found to be dominated by
  high-variance nuisance WavLM dimensions unrelated to the spoof/bona-fide
  distinction — noted in the script's docstring.)
- Outputs:
  - `rerun_2026/tables/wavlm_space_changes.csv` — per-file (n=105 paired
    augmented spoof files): L2 shift, cosine(base,aug), projections along w
    (base/aug/Δ), orthogonal-to-w magnitude (base/aug/Δ), cosine distance to
    bona centroid (base/aug/Δ), score_base, score_aug, delta_score.
  - `rerun_2026/tables/wavlm_space_summary.csv` — aggregate stats +
    Pearson/Spearman correlation of Δ(along w) with Δscore.
  - `rerun_2026/figures/wavlm_pca.png`, `rerun_2026/figures/wavlm_umap.png` —
    2D projections of bona-fide(base) / spoof(base) / spoof(aug,+laughter).
  - `rerun_2026/figures/along_w_vs_delta_score.png` — scatter of the
    correlation.

## 5. Logs
`rerun_2026/logs/generate_laugh_bank.log`, `score_base.log`, `score_aug.log`,
`extract_wavlm.log`.

## 6. Scripts written this pass
`rerun_2026/stage_eval_asv19.py`, `rerun_2026/run_generate_laugh_bank.py`,
`rerun_2026/d1_analysis_rerun.py`, `rerun_2026/extract_detector_wavlm.py`,
`rerun_2026/wavlm_space_analysis.py`.

## 7. Does the laughing augmentation affect sd_along? (paper's headline hardness axis)

- Script: `rerun_2026/sd_along_augmentation.py`
- Command: `python rerun_2026/sd_along_augmentation.py`
- Convention: identical to `experiments/axis_audits/audit_common.py::loso_axis_features`
  and `rerun_2026/wavlm_space_analysis.py` — layer-12 (also 9, 0 for robustness)
  WavLM-base detector-backbone embeddings, standardized per-coordinate by the
  BASE bona-fide mean/SD, `w = centroid(spoof_base) − centroid(bona_base)`
  (unit norm) in that standardized space. Per utterance `s_along = (z−μ_bona)·w`,
  `s_orth = ‖(z−μ_bona) − s_along·w‖`; per system/pool, `s_along` = mean (also
  report median), `sd_along` = std of `s_along` values. Analysis restricted to
  the 105 `augmented==1` aug-spoof files, paired against their exact base
  counterparts by basename (apples-to-apples).
- Outputs:
  - `rerun_2026/tables/sd_along_augmentation.csv` — per-system (13 ASVspoof19
    attack systems, layer 12, fixed base-derived w): n_utts, s_along_mean,
    s_along_median, sd_along, s_orth for base and aug, plus deltas.
  - `rerun_2026/tables/sd_along_augmentation_pooled_summary.csv` — pooled
    (whole-spoof-set) s_along/sd_along/s_orth base vs aug + bootstrap 95% CIs
    + Wilcoxon p-values (per-system fixed-w, per-system LOSO, utterance-level
    paired), one row per layer (12, 9, 0).
  - `rerun_2026/tables/sd_along_augmentation_loso.csv` — per-system LOSO
    (leave-one-system-out w reconstruction, paper's actual axis-estimation
    procedure) s_along/sd_along base vs aug, all 3 layers stacked.
  - `rerun_2026/figures/sd_along_augmentation_per_system.png` — per-system
    s_along and sd_along, base vs aug, layer 12.
  - `rerun_2026/figures/sd_along_augmentation_pooled.png` — pooled bar chart.

### Result

At layer 12 (the paper's layer), fixed base-derived w, n=13 ASVspoof19 attack
systems (5–19 utts/system; caveat: audit_common's own MIN_UTTS=8 threshold
leaves only 7/13 systems non-thin — pooled/utterance-level numbers below are
the primary read given this):

| statistic | base (pooled) | aug (pooled) | Δ | 95% CI (paired boot) | per-system Wilcoxon p (13 sys) | LOSO Wilcoxon p |
|---|---|---|---|---|---|---|
| **sd_along** | 10.39 | 13.93 | **+3.54** | [1.36, 5.66] | **0.00024** | **0.00049** |
| s_along (mean) | 20.09 | 26.22 | +6.13 | [2.78, 9.48] | 0.068 (n.s.) | 0.0081 |
| s_orth (mean) | 31.53 | 79.11 | **+47.57** | — | — | — |

**sd_along IS affected — it increases, robustly and consistently.** Per-system
delta_sd_along is positive in **13/13 of 13 systems** at layer 12 (Wilcoxon
p=0.00024, the exact floor for n=13; LOSO p=0.00049; pooled bootstrap CI
excludes 0). This direction (aug > base) holds at layer 0 too (Δ=+3.04,
p=0.00073 LOSO) but is not significant/flips small-negative at layer 9
(Δ=−1.57, CI includes 0) — so the magnitude is layer-dependent but the
layer-12 result (the paper's actual definition) is unambiguous and the most
consistent across both fixed-w and LOSO estimation.

s_along (the ASVspoof-operative statistic per the paper) is murkier: pooled/
utterance-paired and LOSO tests are significant and point the SAME direction
as sd_along — **increasing**, i.e. moving further along w into spoof territory,
not toward bona — but the naive per-system fixed-w Wilcoxon (13 system-level
medians) is not significant (p=0.068, 8/13 systems up, 5/13 down) and the sign
flips at layers 9 and 0 (large pooled decreases, −13.0 and −10.3). So s_along's
response to the augmentation is far less robust/layer-stable than sd_along's.

**Mechanism / interpretation.** The dominant effect by a wide margin is
orthogonal: Δs_orth (+47.6, layer 12) is ~8× the magnitude of Δs_along (+6.1),
matching agent A's "laughter island" finding — augmented spoof embeddings
mostly move OFF the spoof–bona axis rather than along it. sd_along increasing
is a secondary, indirect symptom of this: the laughter insert's effect is
heterogeneous (varies with position start/mid/end and per-clip acoustic
content — audit's own D1 rerun found position-dependent evasion strength,
mid > start > end), so different augmented utterances get pushed by different
amounts along whatever residual component their orthogonal shift projects
onto w, inflating the within-system scatter of s_along (= sd_along) even
though the central tendency (s_along mean/median) doesn't move consistently
toward bona. Net: **the axis geometry (w, s_along, sd_along) built the paper's
way registers the laughter perturbation only weakly and inconsistently in its
"along" component — mostly it is invisible to a 1-D projection because the
perturbation is orthogonal — even though the actual detector score drops
sharply (0.889→0.625, D1 rerun) and evasion is real.** This means sd_along/
s_along, calibrated on ASVspoof19's few, TTS/VC-based attack systems, are not
good proxies for "how effective is this specific evasion trick" here: the
laughter attack is a different kind of perturbation (a localized acoustic
insert, not a change in overall synthesis quality) that the linear spoof↔bona
axis was not built to capture, and its evasion mechanism is largely orthogonal
to that axis rather than a movement toward the bona-fide centroid.

## C3 (architecture & benchmark independence) — added 2026-07-11
- Obtained official AASIST: cloned github.com/clovaai/aasist; staged AASIST.pth + AASIST.py + AASIST.conf into baselines/aasist/{models,config}.
- MLAAD-tiny eval built: `python rerun_2026/concerns/c3_build_mlaad.py` -> data/eval_mlaad (64 spoof + 120 LibriSpeech bona). (mueller91/MLAAD is gated/inaccessible; used mueller91/MLAAD-tiny + openslr/librispeech_asr.)
- MLAAD augment/score: augment_laughter.py --src data/eval_mlaad --out data/eval_mlaad_aug --laugh-dirs data/laugh_bank_bark --frac 0.7 --seed 20260710; scored base/aug with ../models/mlaad_wavlm-gat.ckpt -> rerun_2026/concerns/{base,aug}_mlaad.csv.
- MLAAD real-speech-insert control: data/real_speech_bank (30 LibriSpeech clips) -> data/eval_mlaad_realaug -> rerun_2026/concerns/realaug_mlaad.csv.
- Analyses: c3_mlaad_analysis.py, c3_aasist_pipeline.py, d1_defense.py (asv19 ref + mlaad), c3_figure.py.
- Outputs: tables/c3_{aasist,mlaad}_insertion.csv, tables/c3_{aasist,mlaad}_defense.csv, tables/c3_asv19_defense_ref.csv, figures/c3_architecture_independence.png; RESULTS.md "## C3" section.
- Verdict: (3b) AASIST YES (evasion 0.057 -> 0.00 under worst-window; clean EER 1%). (3a) MLAAD WavLM-GAT BLOCKED — checkpoint out-of-domain on MLAAD-tiny (clean EER 47%), insertion not interpretable.
