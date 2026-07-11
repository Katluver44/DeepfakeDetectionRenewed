# WavLM embeddings — rerun 2026-07-11

Produced by `rerun_2026/extract_detector_wavlm.py`.

## Files
- `wavlm_base_embeddings.npz` — keys `layer0`, `layer9`, `layer12`, each a
  `(600, 768)` float32 array. Row order matches `wavlm_base_meta.csv` exactly
  (row i of the .npz == row i of the meta CSV).
- `wavlm_base_meta.csv` — 600 rows, one per embedded file. Columns:
  `file` (relative path, e.g. `wavs/spoof_00042.wav`), `basename`,
  `set` (`base` or `aug`), `label` (`bona-fide`/`spoof`), `speaker`,
  `system_id` (ASVspoof19 attack id, e.g. `A07`; `-`/empty for bona-fide),
  `augmented` (`1`/`0`/`` — from the aug manifest; only meaningful for `set=aug`
  spoof rows), `position` (`start`/`mid`/`end`, aug set only), `insert_file`
  (path to the Bark laughter clip spliced in, aug set only).

## Source model / representation
- Backbone: the DETECTOR's own frozen WavLM-base, i.e.
  `lit.model.transformer_in_phoneme_model` inside
  `../models/asv19-wavlm-gat-full.ckpt` (a `transformers.WavLMModel`,
  `microsoft/wavlm-base` architecture, loaded via the checkpoint's
  `phoneme_model` submodule). This is the exact backbone whose output feeds
  the model's adaptive phoneme pooling / GAT / BiLSTM / classification head —
  the representation the natural-vs-synthetic axis (`w`, sd_along) should be
  computed in.
- Layers saved: `hidden_states` indices **0** (post feature-projection, before
  any transformer block), **9**, and **12** (last transformer layer output —
  this is the layer the paper's `sd_along` uses, and matches what the
  detector itself consumes as `phoneme_feat`).
- Pooling: MEAN over time (frame axis) per file, giving one 768-d vector per
  (file, layer).
- Preprocessing: mono 16 kHz, then **center-crop/tile to exactly 3s (48000
  samples)** via `score_itw_detector.crop_or_tile_center` — the SAME crop the
  detector uses when scoring — applied BEFORE running WavLM. So each
  embedding reflects exactly the audio window the detector actually saw.
- Values are **RAW / unstandardized** (no per-coordinate normalization
  applied). Downstream consumers (e.g. the sd_along axis-hardness analysis)
  should standardize per-coordinate by the bona-fide mean/SD themselves, as
  in `experiments/axis_audits/audit_common.py::loso_axis_features` (which
  does `Z = (X - mu_bona) / sd_bona` before computing the axis).

## Datasets covered
- `set=base`: `data/eval_asv19` — 150 bona-fide + 150 spoof, ASVspoof2019-LA
  test split (seed 20260710).
- `set=aug`: `data/eval_asv19_aug` — same 300 files, but 70% of the 150 spoof
  files (105) have a Bark synthetic laughter clip spliced in at a random
  start/mid/end position (`scripts/augment_laughter.py`, seed 20260710,
  laughter bank `data/laugh_bank_bark`, 60 clips). Bona-fide rows and
  non-augmented spoof rows are byte-identical to `base`.

## Reuse recipe (e.g. for a sd_along audit)
```python
import numpy as np, pandas as pd
npz = np.load("rerun_2026/embeddings/wavlm_base_embeddings.npz")
meta = pd.read_csv("rerun_2026/embeddings/wavlm_base_meta.csv")
X12 = npz["layer12"]          # (600, 768), row-aligned with meta
base = meta["set"] == "base"  # use base (unaugmented) set for per-system axis work
labels = (meta["label"] == "spoof").astype(int).values
systems = meta["system_id"].values
```
The `rerun_2026/wavlm_space_analysis.py` script in this same directory shows
a worked example: axis `w = centroid(spoof_base) - centroid(bona_base)` in
raw layer-12 space, and per-file projection deltas for the augmented spoof
files.

## Secondary representation (not extracted here — time-permitting extra)
The brief's secondary suggestion (WavLM-Large L9/L12 via
`scripts/extract_wavlm_features.py`) was NOT run in this pass — the primary
detector-backbone embeddings above were judged sufficient to answer "how does
laughter insertion move points in the space the detector actually reads,"
which is the more decision-relevant representation. If needed later, that
script already exists and follows the same 3s-crop-then-mean-pool convention
(confirm crop policy is applied the same way if extending — as written it
extracts from full segments defined by a `segment_inventory.csv`, so it would
need a small wrapper to point at these 3s-cropped wavs / apply the crop first).
