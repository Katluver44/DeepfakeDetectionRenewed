# Concerns — results log

## C1+C4 Statistical rigor

Scripts: `rerun_2026/concerns/scripts/c1_taxonomy_fdr.py`, `rerun_2026/concerns/scripts/c4_fig2a_ci.py`.
Both are pure recompute from cached tables/embeddings — no re-extraction, no WavLM forward passes.

### C1 — acoustic taxonomy (Fig 1): BH-FDR + bootstrap CIs on Cliff's δ

Data: `tables/laughter_literature_features.csv` (60-cell grid: 12 descriptors × 5 generators,
uncorrected p) cross-checked against the per-clip values in
`tables/laughter_literature_features_clips.csv` (120 clips, 20/group × 6 groups). Recomputing
Cliff's δ and the Mann–Whitney p directly from the per-clip table reproduces the published
point estimates almost exactly (max |Δδ| = 0.0027, max |Δp| = 0.011 — differences are float/tie
handling only), so the pipeline is verified before adding statistics.

**BH-FDR across the full 60-cell grid:**
- 24/60 cells were "bold" in the paper (uncorrected p<.05).
- **21/24 survive BH correction at q<.05**; 23/24 survive at the looser q<.10.
- Only 3 cells lose significance under correction: `harmonicity_acf_db × bark_laughs_inline`
  (δ=−0.41, p=.029→q=.077, still survives q<.10), `unvoiced_run_mean_s × bark_laughter_token`
  (δ=+0.42, p=.024→q=.065, still survives q<.10), and `unvoiced_run_mean_s × xtts`
  (δ=−0.37, p=.048→q=.121, does **not** survive even q<.10 — this is the one previously-bold
  cell that should now be read as non-significant).
- Net effect: the taxonomy's qualitative claims are essentially unchanged after correction —
  correction mainly strips out the borderline (|δ|≈0.35–0.42) cells, which the power analysis
  below flags as underpowered anyway. The large-effect cells (|δ|>0.5, e.g. Bark-inline's bout
  duration δ=−0.89 [CI −0.99,−0.72], AudioLDM2's spectral CoG δ=−0.84 [CI −1.00,−0.62],
  Parler-TTS's harmonicity δ=−0.83 [CI −0.96,−0.65]) all survive with tight, zero-excluding CIs.

**Bootstrap 95% CIs on Cliff's δ** (5000 resamples, resampled independently within the real
and synthetic groups): all 21 surviving-at-q<.05 cells have CIs that exclude 0; several
q<.10-only or non-surviving cells (e.g. the 3 above) have CIs that come close to or straddle 0,
consistent with the FDR call.

**Sensitivity/power note (N=20/20):** using the Hanley–McNeil asymptotic SE(AUC) and the
AUC↔Cliff's-δ identity (δ = 2·AUC−1), the minimum |δ| reliably detectable at α=.05, 80% power,
n=20/20 is **≈0.45**. Effects below that (roughly |δ|<0.45) are underpowered at this N even
before multiple-comparison correction — a non-significant or FDR-non-surviving small-N cell
should be read as "inconclusive," not "no difference." Full note:
`rerun_2026/concerns/tables/taxonomy_power_note.txt`.

Outputs: `rerun_2026/concerns/tables/taxonomy_fdr_ci.csv` (feature, generator, cliffs_delta,
delta_ci_lo/hi, boot_se, p_raw, q_bh, bold_in_paper_p05, survives_fdr_q05/q10),
`rerun_2026/concerns/figures/fig_taxonomy_fdr.png` (cells annotated `**`=q<.05, `*`=q<.10,
plain=non-surviving, with δ and its 95% bootstrap CI printed in each cell).

### C4 — Fig 2a within-generator AUCs: bootstrap CIs vs the content-free floor

**Data note (documented judgment call):** the brief pointed at `embeddings/d3/mean_emb_layer12.npy`
+ `embeddings/d3_inventory.csv`, but that pair only covers 3 groups (laugh-real, speech-real,
laugh-bark) — not the 5 generator families plotted in Fig 2a. The file that actually covers all
5 families (`bark_laughter_token`, `bark_laughs_inline`, `parler_tts`, `xtts`, `audioldm2`) plus
`laugh-real`/`speech-real` is `embeddings/d3_multi/mean_emb_layer12.npy` +
`embeddings/d3_multi_inventory.csv` (682 clips, WavLM-Large 1024-d) — used instead. These cached
embeddings are the original (audited, pre-fix) mean-pooled extraction, i.e. **without** the
length/silence/loudness confound-control normalization `scripts/d3_fixed.py` applied when it
produced `table_d3_fixed.csv`'s point estimates (that script re-extracts WavLM from raw audio
per-clip; raw audio is not present in this environment — the inventory's `file_path` column
points at a machine (`/home/ubuntu/...`) this environment doesn't have, confirmed by direct
`os.path.exists` checks). Re-running WavLM was out of scope ("pure recompute"), so recomputed
WavLM AUCs here run a bit higher for some families than table_d3_fixed (near-ceiling ~0.97–1.00
vs. the confound-controlled 0.96–0.99) — the discrepancy is disclosed in the output table's
`wavlm_l12_pca20_auc_ref_table_d3_fixed` column, and the qualitative WavLM-vs-floor comparison
is preserved. Similarly, no per-clip content-free scalar features are cached anywhere (they were
computed on-the-fly from raw audio and never written to disk in the original run), so the floor's
CI is obtained **analytically** via the Hanley–McNeil (1982) asymptotic SE for AUC, centered on
the table_d3_fixed point estimate, rather than by direct bootstrap. WavLM's CI *is* a full
nonparametric stratified bootstrap (2000 resamples) on the actual cached embeddings — the gold
standard the brief asked for. The WavLM−floor gap CI combines both via Monte Carlo (paired draws
from the WavLM bootstrap distribution and an independent Normal(floor, SE_HM) draw for the floor)
— a conservative combination since it treats the two AUCs as independent.

**Results (balanced N=30/group, matching `scripts/d3_fixed.py`'s subsampling exactly):**

| family | WavLM L12+PCA20 [95% CI] | content-free floor [95% CI] | gap [95% CI] | claim survives? |
|---|---|---|---|---|
| bark_laughter_token | 0.968 [0.918, 1.000] | 0.706 [0.574, 0.838] | +0.262 [+0.126, +0.401] | **yes** |
| bark_laughs_inline | 1.000 [1.000, 1.000] | 0.772 [0.652, 0.892] | +0.228 [+0.115, +0.347] | **yes** |
| parler_tts | 1.000 [1.000, 1.000] | 0.889 [0.803, 0.975] | +0.111 [+0.025, +0.193] | **yes** |
| xtts | 1.000 [1.000, 1.000] | 0.711 [0.580, 0.842] | +0.289 [+0.156, +0.418] | **yes** |
| audioldm2 | 0.981 [0.937, 1.000] | 0.961 [0.910, 1.000] | +0.020 [−0.038, +0.085] | **no — gap CI includes 0** |

Every family except AudioLDM2 keeps a "WavLM beats the content-free floor" claim with a gap CI
that excludes 0. AudioLDM2 is the one family the paper itself already flagged as a "tie w/ floor
(quirk)" — the CI analysis confirms that call quantitatively: its gap (+0.020) is small and its
95% CI straddles zero, so that family's separability should continue to be treated as
confound-explainable rather than a genuine authenticity signal. This matches the paper's own
verdict column in `table_d3_fixed.csv`, so the CI analysis doesn't overturn any claim — it just
makes explicit which claims (4/5 families) are backed by a zero-excluding interval and which one
(AudioLDM2) is not.

Outputs: `rerun_2026/concerns/tables/fig2a_auc_ci.csv` (per-family n, WavLM AUC + bootstrap CI,
content-free AUC + analytic CI, gap + Monte-Carlo CI, `gap_excludes_zero` flag),
`rerun_2026/concerns/figures/fig_separability_a_ci.png` (dumbbell plot redrawn with error bars;
families whose gap CI excludes 0 are marked with `*`).

### Net take for the paper
- Fig 1 taxonomy: robust to FDR correction — 21/24 previously-bold cells survive q<.05 (23/24 at
  q<.10); only the weakest single cell (`unvoiced_run_mean_s × xtts`) should be walked back to
  "not significant." The N=20/20 power ceiling (~0.45 minimum detectable δ) is the more important
  caveat going forward — it, not the correction, is what should temper claims about the smaller
  (|δ|<0.45) borderline cells.
- Fig 2a separability: the "WavLM > content-free floor" claim now has CIs and holds for 4/5
  families (Bark-token, Bark-inline, Parler-TTS, XTTS); AudioLDM2 remains the one family where
  the apparent separability is not distinguishable from the confound floor once uncertainty is
  shown, consistent with the paper's existing "quirk" framing for that cell.

## C2 Second real corpus / source leakage

**What was staged.** VocalSound is the only real-laughter corpus in D3, and XTTS's synthesis
references were themselves cloned from VocalSound clips, so any "XTTS looks close to real"
finding risks being source/channel leakage rather than a laughter-authenticity signal
(flagged already in `RESULTS.md` §D3's "XTTS: episode-topology mismatch, with a source-leakage
caveat"). A second, fully independent real-laughter corpus was staged: **ESC-50's `laughing`
category** (`ashraq/esc50` on HF), 40 clips, 5.0 s each, 44.1 kHz — real human laughter recordings
sourced from Freesound.org, entirely unrelated to VocalSound and never used as an XTTS/Bark/
Parler/AudioLDM2 reference. (An HF search also turned up `krishnakalyan3/vocal_bursts_taxonomy`
["Snicker", "Chuckle", ... 10 clips/prompt across 82 engineered prompt strings] — this was
**rejected** as a candidate: the systematic 10-per-engineered-prompt structure and its sibling
repos (`mrfakename/gemini_vocal_bursts`, `laion/synthetic_vocal_bursts`) strongly indicate it is
itself LLM/TTS-generated, i.e. synthetic, and using it as a "real" anchor would have reintroduced
exactly the contamination this task is trying to remove.) Audio saved to
`rerun_2026/concerns/data/real_laughter_2/` (+ `meta.csv`); extraction code
`rerun_2026/concerns/extract_real2_wavlm.py`.

**Embeddings extracted.** WavLM-Large L9/L12 mean embeddings, same model/layers as
`scripts/extract_wavlm_features.py`/`scripts/d3_fixed.py`, in two variants (both saved under
`rerun_2026/concerns/data/`):
- `real2_emb_layer{9,12}.npy` — **confound-controlled**: energy-trimmed silence, fixed 2.0 s
  central active segment, peak-normalized to 0.9 (exactly `d3_fixed.py`'s `load_fixed()`), plus
  `real2_contentfree.npy` (the paper's 8-scalar content-free floor features on the same fixed clip).
- `real2_raw_emb_layer{9,12}.npy` — **raw whole-clip**, no trim/peak-norm (matches
  `extract_wavlm_features.py`'s pipeline, which is what produced the *cached* synthetic-family
  embeddings in `embeddings/d3_multi/mean_emb_layer12.npy`).

**Why two variants, and the residual limitation.** Raw audio for VocalSound and for all five
synthetic families (Bark-token, Bark-inline, Parler-TTS, XTTS, AudioLDM2) no longer exists in
this environment (confirmed by `os.path.exists` on every `file_path` in
`embeddings/d3_multi_inventory.csv` — 0/682 resolve) — only their pre-computed *raw-pipeline*
mean embeddings survive. `d3_fixed.py`'s confound-controlled numbers in `table_d3_fixed.csv`
were produced by **re-extracting from raw audio per-run**; that audio is gone, so a genuinely
confound-controlled real2-vs-synth AUC (comparable in rigor to `table_d3_fixed.csv`) **cannot be
reproduced** — this is a hard data-availability limitation, not a design choice, and the C4 agent
working on the same shared cache independently hit and documented the identical constraint. The
best available comparison pairs the new real2 corpus against the cached *raw* synthetic
embeddings, in the same pipeline that generated them.

### Replication result (raw pipeline; `tables/second_anchor_separability.csv`)

WavLM L12+PCA20 AUC, real2 (ESC-50 laughing, n=40, raw whole-clip) vs each cached synthetic
family (raw whole-clip), bootstrap 95% CI (2000 resamples, stratified half-split refits):

| comparison | AUC | boot 95% CI |
|---|---|---|
| real2 vs Bark-token | 1.000 | [0.986, 1.000] |
| real2 vs Bark-inline | 1.000 | [1.000, 1.000] |
| real2 vs Parler-TTS | 1.000 | [1.000, 1.000] |
| real2 vs XTTS | 1.000 | [1.000, 1.000] |
| real2 vs AudioLDM2 | 1.000 | [0.963, 1.000] |
| **real2 vs real-VocalSound** | **0.993** | **[0.970, 1.000]** |

Controls pass: label-shuffle (real2 vs Bark-token) AUC = 0.488 (want ~0.5); real2's own random
half-split AUC = 0.487 on WavLM and 0.600 on the content-free floor (both confound-controlled
pipeline, want ~0.5 — no true label exists within one corpus). A semi-controlled variant
(real2 in the *confound-controlled* pipeline vs synth still in the raw cached pipeline,
`tables/second_anchor_separability_semicontrolled.csv`) gives the same picture: 0.994–1.000
across all five families and 0.994 for real2-vs-VocalSound.

**Does WavLM>floor separation replicate with an independent real anchor?** The raw-pipeline
comparison is **saturated** — every family, including the real-vs-real (VocalSound-vs-ESC50)
pair, sits at AUC 0.99–1.00. This is not the confound-controlled protocol the paper's headline
claim relies on, so it cannot directly confirm or refute the *specific* 0.11–0.29 WavLM-vs-floor
margins in `table_d3_fixed.csv`. What it *does* show, cleanly, is the leakage-probe result below.

### XTTS crux test (Step 4)

Because the raw-pipeline AUC saturates at ~1.0 for **every** family simultaneously (including
real-vs-real), there is no differential signal in this representation that could show XTTS
being any more or less separable from an independent real anchor than Bark/Parler/AudioLDM2 are.
**This means the specific, fine-grained question — "does XTTS's apparent closeness-to-real
survive a non-VocalSound anchor, at a confound-controlled 0.97-vs-0.71-floor level of
precision?" — cannot be answered from the surviving cached data; that requires re-synthesizing
or re-recording clips with a preserved, reprocessable raw-audio pipeline.** This is a genuine
open item, recorded as unresolved rather than papered over.

What CAN be said: the paper's own confound-controlled table already shows XTTS is not an outlier
in *raw AUC magnitude* among the four families with a "signal" verdict — real-vs-XTTS AUC (0.972)
is within the 0.956–0.994 band of Bark-inline/Bark-token/Parler-TTS, and its floor-beating margin
(+0.26) is the second-largest of the five, comparable to Bark-token's (+0.27) and larger than
Parler's (+0.11). The "XTTS looks closer to real" claim in RESULTS.md §D3 was never primarily an
AUC claim — it came from the literature-feature *acoustic-profile* analysis (voiced fraction, F0,
harmonicity), where XTTS tracked the real anchor unusually closely on several axes. That specific
claim remains **unverified against a second anchor** and should stay flagged exactly as
RESULTS.md §D3 already flags it ("XTTS must therefore be evaluated with references from a
different real corpus before calling it more authentic") — this task did not lift that caveat,
because lifting it requires raw audio (or a repeat XTTS synthesis run referencing ESC-50 clips)
that is not available here.

### Leakage probe (Step 5; `tables/leakage_probe.csv`) — the crux quantification

| probe | AUC | interpretation |
|---|---|---|
| real-VocalSound vs real2-ESC50 (raw pipeline) | **0.993** [0.970, 1.000] | two independent REAL corpora — this is the leakage floor |
| real-VocalSound vs XTTS (confound-controlled, `table_d3_fixed.csv`) | 0.972 | prior reported real-vs-XTTS separability |
| real2-ESC50 vs XTTS (raw pipeline) | 1.000 [1.000, 1.000] | independent-anchor XTTS separability |
| label-shuffle control | 0.488 | sanity (~0.5 expected) |
| real2 internal half-split, WavLM | 0.487 | sanity (~0.5 expected) |
| real2 internal half-split, content-free | 0.600 | sanity (~0.5 expected) |

**This is the central finding.** In the (raw, uncontrolled) representation that is the only one
available for a same-pipeline comparison against the surviving synthetic-family embeddings, two
*independent real* laughter corpora (VocalSound vs. ESC-50) separate at **AUC 0.993** —
statistically indistinguishable from, and numerically *higher than*, the previously reported
confound-controlled real-vs-XTTS separability of 0.972, and on par with the raw real-vs-every-
other-synthetic-family AUC of ~1.0. Controls confirm this isn't a methodology artifact: label
shuffling collapses to chance (0.488) and a real2-internal random split collapses to chance
(0.487/0.600). **Two corpora that are both unambiguously "real" laughter are exactly as
separable, in this representation, as real is from any synthetic family.** This directly
replicates and *extends* the mechanism D3_AUDIT already found within VocalSound alone (raw
speech-real vs laugh-real AUC ≈ 1.0, "fires on any two corpora from different pipelines,
laughter or not") to a genuinely independent second real corpus, and confirms the paper's
existing decision to lead with the confound-controlled protocol (`d3_fixed.py`) rather than the
raw AUC numbers: raw WavLM separability is dominated by corpus/channel/recording-chain
differences, not laughter authenticity, and a magnitude like 0.97 is by itself uninformative
about content unless it is shown (as `table_d3_fixed.csv` does for VocalSound) to beat a
content-free floor on length/silence/loudness-equalized clips.

### Verdict

- **Second real corpus:** staged successfully (ESC-50 `laughing`, n=40, independent of VocalSound
  and of every synthesis family's reference audio).
- **Does WavLM>floor separation replicate with an independent anchor?** Not directly testable at
  the confound-controlled precision of `table_d3_fixed.csv`, because the raw audio needed to
  reproduce that protocol for the synthetic families no longer exists. The raw-pipeline
  replication that IS possible saturates uninformatively (all families ~1.0).
- **Does XTTS's "closer to real" signal survive removing the shared source?** **Unresolved** for
  the acoustic-profile claim specifically (needs a fresh XTTS synthesis run or preserved raw audio
  to test properly) — this task could not lift RESULTS.md §D3's existing caveat. However, the
  **leakage probe strongly supports the caveat's premise**: independent real-vs-real separability
  (0.993) matches or exceeds real-vs-XTTS separability (0.972), so a same-anchor "XTTS looks
  close to real" AUC/similarity number is, on this evidence, at least as likely to reflect
  corpus/channel leakage as laughter content. The paper should keep the XTTS source-leakage
  caveat in place (do not read it as resolved by this work) and additionally cite this leakage
  probe as direct, independent-corpus evidence for *why* the caveat is warranted, not just a
  theoretical concern.
- **What would close this out properly:** either (a) recover/re-download the raw VocalSound and
  synthetic-family audio and rerun `d3_fixed.py`'s exact protocol with ESC-50 (or another real
  corpus) substituted for VocalSound, confound-controlled on both sides; or (b) regenerate the
  five synthesis families' laughter with ESC-50 (not VocalSound) clips as the XTTS reference
  set and any other generator-specific conditioning, then rerun D3 end-to-end. Either would let
  Step 4's fine-grained "is XTTS still closer to real" question be answered at the same rigor
  as the original VocalSound-anchor result.

Artifacts: `rerun_2026/concerns/extract_real2_wavlm.py`,
`rerun_2026/concerns/second_anchor_analysis.py`,
`rerun_2026/concerns/data/real_laughter_2/` (+ `meta.csv`),
`rerun_2026/concerns/data/real2_*.npy`, `rerun_2026/concerns/data/real2_inventory.csv`,
`rerun_2026/concerns/tables/second_anchor_separability.csv`,
`rerun_2026/concerns/tables/second_anchor_separability_semicontrolled.csv`,
`rerun_2026/concerns/tables/leakage_probe.csv`,
`rerun_2026/concerns/figures/second_anchor_auc.png`.

## C3 Architecture & benchmark independence

**Concern (Fig 3):** the laughter-insertion "detector-locality" finding — inserting a
laughter clip pulls ASVspoof2019-LA fakes toward the genuine side and evades detection, while
worst-window / local scoring repairs it — is shown on ONE detector (WavLM-GAT) and ONE
benchmark (ASVspoof2019-LA), so it is not yet established as architecture-independent. Goal:
replicate on **(3b)** a second, non-WavLM architecture and **(3a)** a second benchmark.

All numbers reuse the exact rerun pipeline: `scripts/augment_laughter.py` (splice a Bark
`[laughter]` clip from `data/laugh_bank_bark` into a random 70% of spoof files, seed 20260710),
`scripts/d1_analysis.py::analyze` (paired base→aug Δscore, Wilcoxon, rank-biserial, evasion at
clean-EER threshold), and `scripts/d1_defense.py` (slide the model's native window over the full
waveform, hop 1 s; score by the **most-spoof** window instead of a single center crop).

Verdict: **(3b) AASIST — YES, replicates. (3a) MLAAD WavLM-GAT — BLOCKED / not interpretable**
(detector is out-of-domain on the only accessible MLAAD eval; documented below).

| detector | benchmark | dim tested | clean EER | insertion Δ (dir) | rank-biserial | evasion single-crop | evasion worst-window (max) |
|---|---|---|---|---|---|---|---|
| WavLM-GAT (asv19) | ASVspoof19-LA | reference (paper Fig 3) | 6.0% | −0.264 → toward bona | −0.68 | **0.257** | **0.04** |
| **AASIST (official)** | ASVspoof19-LA | **2nd architecture** | 1.0% | −1.89 → toward bona | −0.49 | **0.057** | **0.00** |
| MLAAD WavLM-GAT | MLAAD-tiny | 2nd benchmark | **47.2%** (degenerate) | +0.21 (wrong sign) | +0.65 | 0.244 (uninterpretable) | 0.09 |

Tables: `tables/c3_aasist_insertion.csv`, `tables/c3_aasist_defense.csv`,
`tables/c3_mlaad_insertion.csv`, `tables/c3_mlaad_defense.csv`,
`tables/c3_asv19_defense_ref.csv`. Figure: `figures/c3_architecture_independence.png`.
Scripts: `c3_aasist_pipeline.py`, `c3_build_mlaad.py`, `c3_mlaad_analysis.py`, `c3_figure.py`.

### (3b) Second architecture — official AASIST (spectro-temporal, no WavLM): **YES**
- **Weights obtained.** The repo's `baselines/aasist/` was empty. Cloned the official
  `github.com/clovaai/aasist`; its `models/weights/AASIST.pth` (1.28 MB), `models/AASIST.py`,
  and `config/AASIST.conf` are real binaries (not LFS pointers). Copied into
  `baselines/aasist/{models,config}` exactly where `experiments/scripts/j5_aasist_crossfamily.py`
  expects them. Loads and scores (2-way head; input = raw 16 kHz, `nb_samp=64600` ≈ 4.04 s).
  Score orientation determined empirically on the clean labeled set (median-split), matching j5's
  `orient` convention, so the sign is not hardcoded.
- Scored the **same** clean vs laughter-inserted ASVspoof fakes (`data/eval_asv19`,
  `data/eval_asv19_aug`) used for the WavLM-GAT result. Clean EER **1.0%** (AASIST works in-domain).
- **Insertion evades in the same direction:** Bark laughter shifts the paired spoof score toward
  bona-fide (Δ_mean = −1.89 raw units, Wilcoxon p = 1.4e-5, rank-biserial −0.49), strongest for
  **start** inserts (Δ = −2.25), exactly mirroring the WavLM-GAT position law. Aug-fake catch
  drops 0.99 → 0.94; evasion at the clean-EER threshold = **5.7%**.
- **The worst-window defense repairs it on AASIST too:** switching from single-center-crop to
  the most-spoof sliding window drops evasion 0.057 → **0.00** (max) / 0.01 (p90) while keeping
  clean EER ≤ 2% and restoring aug-fake catch to 1.00.
- Magnitude of the evasion is smaller than on WavLM-GAT (5.7% vs 25.7%) because AASIST is far more
  confident in-domain (base spoof scores sit well above threshold, so a −1.9 shift crosses the
  boundary less often) — but the **vulnerability (significant toward-bona shift) and the fix
  (local max-scoring removes it) are both clearly present on a non-WavLM architecture.** The
  locality phenomenon is therefore not an artifact of the WavLM SSL front-end.

### (3a) Second benchmark — MLAAD WavLM-GAT: **BLOCKED (out-of-domain), documented**
- **Detector/benchmark loads and smoke-tests fine** (`../models/mlaad_wavlm-gat.ckpt` via the same
  `score_itw_detector.load_detector` path: white-noise→P(spoof)=0.09, 220 Hz tone→0.99).
- **Data-access deviation (documented):** `mueller91/MLAAD` — the source `build_eval_sets.py`
  expects — is now a **gated** HF repo; our token has `canReadGatedRepos` but is **not on the
  dataset's authorized-user allowlist** (403 GatedRepoError on every file; web approval not
  obtainable in an agent session). Fallback used = **`mueller91/MLAAD-tiny`**, the same MLAAD
  corpus/lab and already the MLAAD source elsewhere in this repo
  (`experiments/results/mlaad/baseline_eval/test_in_distribution.json` points at MLAAD-tiny
  snapshot paths). Real anchor: LibriSpeech test-clean via the `openslr/librispeech_asr` parquet
  mirror (the `kresnik/...` script-dataset the builder expects is unsupported on `datasets` 5.0).
  Built 64 EN-spoof (64 modern TTS systems) + 120 LibriSpeech bona = `data/eval_mlaad`.
- **Why the replication is not interpretable here:** the provided `mlaad_wavlm-gat` checkpoint is
  **out-of-domain on MLAAD-tiny** (whose generators are 2024–2025: Cartesia, Chatterbox, FishTTS,
  f5-tts, kokoro, …). **Clean EER = 47.2%, i.e. near chance** — the detector does not separate
  these fakes from LibriSpeech. (Not a fluke of MLAAD-tiny: the repo's own MLAAD baseline eval
  already records this family at EER 24–32% even in-distribution; MLAAD WavLM-GAT detectors are
  inherently weak, and the gated newer full MLAAD needed for an in-domain match is inaccessible.
  Only one MLAAD checkpoint is staged.) At a near-chance operating point the clean-EER threshold
  is meaningless, so neither the insertion evasion rate nor the worst-window defense on this
  detector is interpretable.
- Consistent with that, **the insertion effect does not reproduce**, and the reason is diagnostic
  rather than contradictory: (i) with **Bark laughter**, Δ is **positive** (+0.21, rank-biserial
  +0.65) — synthetic Bark laughter reads as *spoof* to a MLAAD-trained detector, so it pushes
  scores up, the opposite of dilution; (ii) with a **real-speech insert control** (real
  LibriSpeech clips spliced in with the same seed/positions — the `d1_explain` speech-insert
  control that gave −0.31 on the working ASV19 detector), Δ ≈ **0** (−0.02, Wilcoxon p = 0.94,
  n.s.). So even authentic audio produces **no pooling-dilution** on this detector — precisely
  because a near-chance detector isn't reading the frame-level real/synthetic evidence that
  pooling could dilute. The locality vulnerability is **contingent on the detector actually
  working in-domain** (a low clean EER); where it does (ASV19 WavLM-GAT, AASIST) the effect and
  its fix both appear, where it doesn't (this MLAAD checkpoint on MLAAD-tiny) there is nothing to
  dilute or repair. (Real-speech-insert scores: `rerun_2026/concerns/realaug_mlaad.csv`.)

### Net take for the paper
- **Architecture-independence (the primary reviewer ask) is supported:** the insertion-evasion
  vulnerability and the worst-window-scoring fix both reproduce on **official AASIST**, a
  spectro-temporal detector with no SSL/WavLM component (evasion 0.057 → 0.00 under the defense,
  same toward-bona direction and start-insert position law as WavLM-GAT). The phenomenon is a
  property of **single-crop pooled scoring**, not of the WavLM backbone.
- **Benchmark-independence via MLAAD could not be established** because the only accessible
  MLAAD eval is out-of-domain for the only staged MLAAD checkpoint (clean EER 47%). This is a
  data/model-access limitation, not evidence against the finding; the control experiments show
  the null is exactly what a non-functional (near-chance) detector should produce. Recommended
  paper framing: lead the independence claim on the AASIST result; report MLAAD as a scoped
  negative that additionally demonstrates the **precondition** of the vulnerability (the detector
  must be a working, in-domain classifier for laughter/real-audio insertion to dilute its score).
