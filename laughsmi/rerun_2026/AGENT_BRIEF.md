# Rerun 2026-07-11 — Shared brief (laughing augmentation × WavLM space × sd_along)

Prepared by the lead agent after de-risking. All facts below are VERIFIED unless marked (unverified).

## Environment (already set up — do NOT reinstall unless something is missing)
- Python: `/opt/conda/bin/python` (plain `python`). torch 2.8.0 + CUDA 12.9, **works on GPU**. GPU: 1× NVIDIA L4, 23 GB.
- Installed & working: torch, transformers 4.57, pytorch_lightning 2.6, datasets 5.0, huggingface_hub 0.36, sklearn, pandas, scipy, soundfile, librosa, umap-learn, einops, **torchaudio 2.8.0+CPU** (the CUDA torchaudio wheel is version-mismatched; the CPU wheel imports fine and phoneme_GAT only needs `torchaudio.transforms.{LFCC,Spectrogram}`, unused on the WavLM path). Do not install the CUDA torchaudio wheel.
- HF token: `cat ../secrets.txt` (run from `laughsmi/`). Export `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` before HF calls.
- Working dir for all scripts: `/home/sagemaker-user/DeepfakeDetectionRenewed/laughsmi` (scripts use paths relative to it: `data/`, `detector_out/`, `scripts/`; model is at `../models/`).
- Ignore stderr noise: onnxruntime/absl/cuFFT "already registered" warnings and the nvm/.npmrc warning are harmless.

## The model (VERIFIED loads + scores)
- `../models/asv19-wavlm-gat-full.ckpt` (531 MB), just downloaded from the user's Drive. It is a PyTorch-Lightning `Phoneme_GAT_lit` checkpoint: frozen WavLM-base → adaptive phoneme pooling → GAT → BiLSTM → linear head. Outputs one **logit per utterance**; `score = sigmoid(logit)` = P(spoof). Higher = spoof, lower = bona-fide.
- Load + score via `scripts/score_itw_detector.py` helpers: `from score_itw_detector import load_detector, score_batch, crop_or_tile_center, load_wav_mono_16k`. Smoke test passed: white-noise→0.996, 220 Hz sine→0.365. Input policy: mono 16 kHz, **center-crop/tile to exactly 3 s (48000 samples)** before scoring.

## Datasets to STAGE (laughsmi/data/ is currently EMPTY — only dummy wavs under scripts/tests/_dummy)
### ASVspoof2019-LA test → `data/eval_asv19/` (150 bona-fide + 150 spoof, seed 20260710)
- `build_eval_sets.py` expects a local arrow at `../data/asvspoof_2019_la` with a `test` split having columns `system_id` (`'-'`=bona, `'Axx'`=spoof), `speaker_id`, `audio`. That arrow does NOT exist.
- Source: HF dataset **`Bisher/ASVspoof_2019_LA`** (parquet, splits `test`/`train`/`validation`, NO dataset script). `LanceaKing/asvspoof2019` is a script-dataset and FAILS on datasets 5.0 — do not use it.
- Audio decode: datasets 5.0 needs torchcodec which we do NOT have. Bypass with `.cast_column("audio", datasets.Audio(decode=False))` then `soundfile.read(io.BytesIO(row["audio"]["bytes"]))`. OR download the parquet directly: `hf_hub_download("Bisher/ASVspoof_2019_LA","data/test-00000-of-00001.parquet",repo_type="dataset")` and read with pyarrow/pandas — faster and lets you inspect the schema. **Inspect the columns first** (I could not see them — network was slow); find the bona/spoof label column (likely `system_id`, `key`, or `label`; bona-fide is the `'-'`/`'bonafide'` value) and, importantly, the **per-attack system id** (Axx) — you need it for the sd_along per-system analysis.
- Downloads are SLOW (~GB shards, >2 min just for the header). Use long timeouts / run in background. Write to the eval_asv19 schema `meta.csv`(file,speaker,label; label in {bona-fide,spoof}) + `wavs/`. Also write a sidecar `data/eval_asv19/attack_ids.csv` (file, system_id) so the sd_along agent has per-system labels.

### Laughter inserts → `data/laugh_bank_bark/` (Bark synthetic laughter, the faithful reproduction)
- Original D1 spliced Bark `[laughter]` clips. Generator: `scripts/generate_laugh_bank.py` (read `--help`; needs the `bark` package + model download, slow).
- **Fallback if Bark is infeasible within ~15 min:** stage real laughter from VocalSound (HF, e.g. search for a VocalSound/laughter mirror) OR AudioSet-laughter; last resort reuse `scripts/tests/_dummy/vocalsound/audio/*.wav`. **Clearly record which insert source you used** — it matters for interpreting the evasion sign (Bark synthetic evaded −0.295, real VocalSound −0.133 in the prior run per RESULTS.md). Generate/collect ~60 clips, 16 kHz mono, roughly 1–3 s each.

## Rerun recipe (the "laughing augmentation experiment", from `laughsmi/`)
1. `python scripts/augment_laughter.py --src data/eval_asv19 --out data/eval_asv19_aug --laugh-dirs data/laugh_bank_bark --frac 0.7 --seed 20260710` → splices a laughter clip into a random 70% of spoof files at start/mid/end; writes paired `meta.csv` + `manifest.csv`(file,label,augmented,insert_file,position,insert_dur_s,host_dur_s).
2. Score base + aug with the model:
   `python scripts/score_itw_detector.py --audio_dir data/eval_asv19 --meta data/eval_asv19/meta.csv --ckpt ../models/asv19-wavlm-gat-full.ckpt --out rerun_2026/base_asv19.csv`
   `python scripts/score_itw_detector.py --audio_dir data/eval_asv19_aug --meta data/eval_asv19_aug/meta.csv --ckpt ../models/asv19-wavlm-gat-full.ckpt --out rerun_2026/aug_asv19.csv`
3. Reproduce the evasion table with the pairing logic in `scripts/d1_analysis.py` (`analyze()` pairs base vs aug by basename over `augmented==1` files; reports paired Δscore, Wilcoxon, rank-biserial, evasion rate at clean-EER threshold, Δ by insert position). Prior result to compare against (RESULTS.md Table D1): clean EER ~5.33%, base spoof score ~0.899 → aug ~0.604, Δ≈−0.295, evasion ~0.20.

## WavLM representation for "wavlm space" + sd_along
- **Primary (most faithful to the detector's decision geometry / sd_along):** the detector's OWN frozen WavLM-base features. Extract by running the loaded model's WavLM backbone and mean-pooling frame embeddings (768-d). This is the representation whose natural-vs-synthetic axis the head reads.
- **Secondary (continuity with prior laughsmi D3 figures):** WavLM-Large layers 9 & 12 via `scripts/extract_wavlm_features.py` (`--model microsoft/wavlm-large --layers 9 12 --fp16`).
- Save mean-pooled embeddings for EVERY file in eval_asv19 and eval_asv19_aug, tagged with: file, label, system_id (attack), augmented flag, insert position. Put them in `rerun_2026/embeddings/` (e.g. an .npz + a parquet of metadata) so the sd_along agent can reuse them without re-extracting.

## sd_along definition (from model_behavior_analysis.md / the COLM paper jathin_aaky.pdf)
- Axis `w = centroid(synthetic/spoof embeddings) − centroid(bona-fide embeddings)` in the frozen WavLM space.
- Per attack system: project that system's utterance embeddings onto unit `w`; `s_along` = mean projection, **`sd_along` = std of projections (spread along w)**. The paper's headline: `sd_along → per-system hardness`, Spearman ρ≈0.59 on MLAAD (position `s_along` matters more on few-attack ASVspoof). Confirm exact layer/normalization against the paper text and `experiments/axis_audits/audit2_sdalong_claim.py`.

## Output location
Everything under `laughsmi/rerun_2026/` (`embeddings/`, `tables/`, `figures/`, `logs/`). Keep a running `rerun_2026/MANIFEST.md` of what you produced and the exact commands used.
