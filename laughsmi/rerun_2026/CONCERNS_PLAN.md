# Plan to address reviewer concerns (concerns.md) — LaugHSMI paper

Paper: `laughsmi/paper/main.tex`. Figures: Fig 1 taxonomy (`fig_taxonomy`), Fig 2 separability (`fig_separability`: 2a controlled-AUC-vs-floor, 2b held-out transfer), Fig 3 insertion locality (`fig_insertion`). Work from `cd laughsmi`; env is already set up (see `rerun_2026/AGENT_BRIEF.md` for env + model facts). Output under `laughsmi/rerun_2026/concerns/`.

## Concern → action → feasibility

### C1. Small N (20 real vs 20 synth/family) + uncorrected tests — acoustic taxonomy (§3, §6, Fig 1)
**Action:** Add multiple-comparison correction (Benjamini–Hochberg FDR across the full descriptor×family grid) and bootstrap CIs on the effect size (Cliff's δ). Report which cells survive correction; replace "bold = uncorrected p<.05" with FDR-annotated cells + δ CIs. Add a sensitivity note on N.
**Data (all cached, no re-extraction):** `tables/laughter_literature_features.csv` (60 rows = 12 feats × 5 generators, cliffs_delta + uncorrected p), per-clip values in `tables/laughter_literature_features_clips.csv`, `tables/laughter_dynamics_effects.csv`.
**Feasibility: HIGH (pure recompute).**

### C4. No CIs for within-generator AUCs in Fig 2a
**Action:** Bootstrap (stratified, ~2000 resamples) 95% CIs for every AUC in Fig 2a — WavLM L12(+PCA20) and the content-free floor, per generator family — and for the "WavLM > floor" gaps. Redraw Fig 2a with error bars (Fig 2b already has auc_std). State where the WavLM>floor claim survives once CIs are shown.
**Data (cached):** `embeddings/d3/mean_emb_layer12.npy` (+layer9), `embeddings/d3_inventory.csv` (group labels), `tables/table_d3_fixed.csv` (point estimates to reproduce), `embeddings/d3_features.parquet`.
**Feasibility: HIGH (pure recompute).**
> C1+C4 = one agent (statistical rigor), fully tractable from cache.

### C2. Single real corpus (VocalSound) + XTTS references from same source → source leakage (§6) — the biggest validity threat
**Action:** (a) Stage a SECOND, independent real-laughter corpus (try AudioSet-laughter / an HF laughter set / AVLaughterCycle; documented fallback). (b) Extract WavLM L9/L12 features for it. (c) Re-run the separability (real-vs-synth) and taxonomy comparisons using the NEW anchor against the EXISTING cached synthetic embeddings (d3 covers Bark/Parler/XTTS/AudioLDM2). (d) Specifically re-evaluate XTTS, whose refs were VocalSound, against the non-VocalSound anchor — does its apparent authenticity survive? (e) If a second corpus cannot be staged, deliver a source-leakage quantification instead: a speaker/channel probe measuring how much VocalSound-vs-XTTS separability is carried by channel/speaker rather than laughter content, plus a scoping note.
**Data:** synthetic embeddings cached in `embeddings/d3*`; raw D2/VocalSound audio is GONE (must stage anew for the real side only).
**Feasibility: MEDIUM / uncertain (depends on corpus availability).**

### C3. Single detector architecture + single benchmark (ASVspoof2019-LA) — locality not shown architecture-independent (§6, Fig 3)
**Action:** Replicate the laughter-insertion locality vulnerability (Fig 3: laughter pulls fakes toward genuine; max/worst-window scoring repairs it) on:
- (3a) **Second benchmark, same arch:** MLAAD-trained WavLM-GAT (`models/mlaad_wavlm-gat.ckpt`, staged). Reuse/stage an MLAAD spoof eval; run the augment→score→evasion→defense pipeline.
- (3b) **Second architecture:** official **AASIST** (spectro-temporal, NOT WavLM). The repo integration is at `baselines/aasist/` + `experiments/scripts/j5_aasist_crossfamily.py`; weights `baselines/aasist/models/weights/AASIST.pth` are MISSING — download from the official clovaai/aasist repo. Score clean vs laughter-inserted ASVspoof fakes (reuse `data/eval_asv19` + `data/eval_asv19_aug` from the rerun); test whether insertion evasion + a worst-window defense replicate on a non-WavLM detector.
**Data:** eval_asv19 + eval_asv19_aug already staged; MLAAD ckpt staged; AASIST weights to download.
**Feasibility: MEDIUM (AASIST weights download + scoring-path adaptation).**

## Execution
Three parallel Sonnet agents: P1 (C1+C4, stats), P2 (C2, second corpus), P3 (C3, architecture/benchmark). Each writes tables/figures under `rerun_2026/concerns/` and appends a section to this file's companion `rerun_2026/concerns/RESULTS.md`. Data-limited items get a documented fallback rather than a stall.
