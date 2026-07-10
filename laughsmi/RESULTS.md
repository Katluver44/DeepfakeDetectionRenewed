# LaugHSMI — Results Summary (pivot: laughter × deepfake detectors)

All numbers reproducible from `laughsmi/scripts/*` on the CSVs in
`detector_out/`, tables in `tables/`, figures in `figures/`.

## D1 — Inserting synthetic laughter into fakes evades detectors (MUST-SHIP)

Two tiny balanced eval sets, scored with the matching checkpoint via
`score_itw_detector.py` (score=sigmoid(logit), higher=spoof):
- **eval_asv19**: 150 bona-fide + 150 spoof from ASVspoof2019-LA test →
  `asv19-wavlm-gat-full.ckpt`. **Clean EER 5.33%** (detector works in-domain).
- **eval_mlaad**: 156 MLAAD spoof (39 EN TTS systems, freshly downloaded) + 150
  LibriSpeech test-clean bona-fide anchor → `mlaad_robust_goat.ckpt`. Clean EER
  37% — this "robust" model is **saturated** (scores ~0.95 for real and fake
  alike) and barely separates modern MLAAD from LibriSpeech (a finding in itself).

Augmentation (`augment_laughter.py`): splice a Bark `[laughter]` clip into a
random **70% of spoof files** at a varied position (start/mid/end); paired
file-ids so base/aug scores compare 1:1.

**Table D1** (`tables/table_d1.csv|.tex`):

| Eval (detector) | n | clean | +laugh | Δ | unmask | silence | catch→ / evade |
|---|---|---|---|---|---|---|---|
| ASVspoof19 (asv19) | 105 | 0.899 | 0.604 | −0.295 | 0.891 | 0.363 | 0.97→0.80 / **0.20** |
| MLAAD (robust) | 109 | 0.963 | 0.962 | −0.001 | 0.949 | 0.934 | 0.61→0.38 / 0.62 |

- ASV19: Wilcoxon p≈0, rank-biserial −0.78 (large). Strongest for start inserts (−0.38).
- **Causal control** (`d1_masking_control.py`): *unmask* (splice laughter back
  out) recovers the clean score (0.891) → the drop is the inserted region, not a
  seam/crop artifact. *silence* (zero the region, keep timeline) drops further
  (0.363).
- Fig `figures/figure_d1.png`: per-file score shift + Δ histogram.

### Why it happens (`d1_explain.py`) — the mechanism is pooling dilution, not laughter acoustics
- **Speech-insert control:** splice a random *real LibriSpeech speech* clip into
  the same fakes → score drops **−0.306**, essentially identical to laughter's
  **−0.295**. So the effect is **not laughter-specific**: inserting *any*
  authentic non-synthetic audio pulls the score toward "real."
- **Dose-response is flat:** Spearman ρ≈0 between Δscore and the laughter
  fraction of the 3s crop; Δ is already ≈−0.24 even when laughter is <25% of the
  crop (saturates fast).
- **Interpretation:** the detector mean-pools frame-level evidence, so a minority
  of genuinely-real frames dilutes the pooled spoof score. **Laughter is an
  evasion vector *of opportunity*** — a socially plausible, unsuspicious segment
  to splice into synthetic speech (unlike a stranger's sentence) — rather than an
  acoustically special one. The blind spot is real; the cause is pooling.
- **Fix implied by D3:** WavLM *can* separate real from synthetic laughter
  (AUC≈1.0), so a detector that scored laughter/non-speech regions specifically,
  instead of mean-pooling them away, could both resist this evasion and flag
  synthetic laughter. That is the training-time motivation for the D2 dataset.

### Laughter vs. words in WavLM space (`d1_laughter_vs_words.py`, `tables/table_d1_laughter_vs_words.csv`)
Why laughter frames read as non-spoof: they sit **off the speech manifold** the
detector's discrimination was learned on.
- Centroid cosine distance laughter↔speech = **0.22 (L9), 0.28 (L12)**; laughter
  vs. speech linear probe **AUC = 1.00** at both layers (trivially separable).
- Laughter is **~3× more internally dispersed** than speech (within-group cosine
  spread 0.13–0.15 vs. 0.04–0.05) — laughter is acoustically heterogeneous.
- Real and synthetic (Bark) laughter are far closer to *each other* (cos
  0.04–0.05) than either is to speech (0.20–0.28): laughter (real or synthetic)
  is one island, speech another.
- **Takeaway:** inserted laughter frames are effectively out-of-distribution for
  a spoof classifier trained on synthetic-speech artifacts; mean-pooling then
  reads them as non-spoof and dilutes the score — matching the speech-insert
  control (any authentic off-manifold frames do it).

### Test-time defense — no retraining (`d1_defense.py`, `tables/table_defense_asv19.csv`)
The mechanism (mean-pooling dilution + center-crop displacement) implies a fix:
slide 3s windows over the **full** waveform and score by the **most-spoof
window** instead of the whole-crop mean — genuine synthetic speech still scores
high even when a laughter window would dilute the average.

| scoring | clean EER | evasion (aug fakes) | aug-fake catch |
|---|---|---|---|
| original single-center-crop (the attack) | 5.33% | **0.20** | — |
| full-waveform sliding + mean-pool | 3.67% | 0.03 | 0.97 |
| full-waveform + **max (worst-window)** | 4.33% | **0.01** | 0.99 |
| full-waveform + p90 | 4.33% | 0.02 | 0.98 |

- Sliding over the full waveform (vs. a single center crop) already cuts evasion
  20%→3% — the synthetic speech can no longer be cropped away.
- Worst-window aggregation cuts it further to **~1%** and *raises* augmented-fake
  catch to 99%, at a negligible clean-EER cost (3.7%→4.3%).
- **No retraining** — a pure inference-time change that follows directly from the
  diagnosed mechanism.

### Replication on a properly-calibrated MLAAD model — the effect's DIRECTION is contingent on the detector's bona-fide distribution
The stock MLAAD checkpoints were saturated (EER 37–42%). We trained a
domain-matched one, `models/mlaad_libri_calibrated.ckpt` (MLAAD-English spoof +
LibriSpeech train-clean bona, speaker-disjoint from the test-clean eval), reaching
**clean EER 16.7%** (bona 0.49 vs spoof 0.92). Re-running the identical laughter
augmentation on it:

Δ spoof-score from inserting laughter into the same fakes, by insert **source**
(Bark synthetic vs VocalSound real), per model:

| model | +Bark(synth) Δ | +Real(VocalSound) Δ | evasion (Bark) | catch (Bark) |
|---|---|---|---|---|
| ASVspoof19 (asv19) | **−0.295** | **−0.133** | 0.20 | 0.97→0.80 |
| MLAAD (calibrated, EER 16.7%) | **+0.034** | **+0.026** (n.s. vs Bark, p=0.31) | 0.06 | 0.83→0.94 |

**The sign differs by model, and it is NOT about synthetic-vs-real laughter.**
- On ASV19, laughter *evades* (score drops, fakes look more real); Bark evades
  *more* than real laughter (−0.30 vs −0.13).
- On the calibrated MLAAD model, laughter mildly *raises* the spoof score, and
  **real and Bark laughter do so equally** (p=0.31) — so the +shift is **not** a
  synthetic-laughter tell (tested: hypothesis that MLAAD flags Bark-as-generated —
  REJECTED). The masking control shows it is the inserted region (unmask → base),
  and **silence raises the score too (+0.030)**, so it is generic "inserted
  non-read-speech segment," not laughter authenticity.

**What is established:** inserting a laughter (or silence) segment perturbs both
detectors' mean-pooled scores; the perturbation is large and evasive for ASV19
(−0.13 to −0.30) but small and slightly anti-evasive for calibrated MLAAD
(+0.03), and this holds for **both real and synthetic** laughter. So the
direction/magnitude reflects each model's learned **decision geometry**, not a
property of laughter authenticity.

**What is NOT established (open):** *why* the two models move in opposite
directions. It is not the insert being synthetic (tested above) and not obviously
the bona-fide class (both ASV19 and LibriSpeech are laughter-free read speech —
an earlier "bona-fide distribution sets the direction" explanation was WRONG and
is retracted). Candidate untested causes: different spoof-class cues learned by
each model, or MLAAD-cal operating near score saturation (~0.9, little downward
room). Report the asymmetry as an empirical finding, not a mechanism.

## D3 — Real vs. synthetic laughter in WavLM (SSL) space

Inventory `embeddings/d3_inventory.csv`: 300 VocalSound (real), 100 Bark
(synthetic), 150 LibriSpeech speech (anchor). Features via
`extract_wavlm_features.py` (WavLM-Large L9/L12).

> **⚠ AUDITED — the raw AUC≈1.0 claim does NOT hold (see `D3_AUDIT.md`, `scripts/audit_d3.py`).**
> The near-perfect separation is driven by **recording-pipeline confounds** (clip
> duration, silence padding, loudness, spectral tilt), not a demonstrated
> laughter-authenticity representation. Evidence:
> - A **content-free 8-scalar probe** (duration, lead/trail silence, RMS,
>   spectral centroid/rolloff, ZCR, HF-energy — *no* laughter content) already
>   gets **AUC 0.83–0.98** real-vs-synth (and 0.98 speech-vs-laughter), no WavLM.
> - **speech-real vs laugh-real is also AUC 1.0** and survives every fix → AUC 1.0
>   fires on any two corpora from different pipelines, laughter or not.
> - After a fix-stack (2 s center-crop + peak-normalize + PCA), WavLM AUC falls to
>   **0.87–1.00** and **ties or loses to the content-free probe for 4 of 5
>   methods** (only Bark-token modestly ahead, 0.87 vs 0.83, n=30 — within noise).
> - Controls pass: label-shuffle → 0.50, real-vs-real split → 0.53, no NaNs/dupes,
>   layers distinct. So the *classifier* is sound; the *labels* are confounded with
>   corpus statistics.

### D3 CONFOUND-CONTROLLED RERUN (`d3_fixed.py`, `tables/table_d3_fixed.csv`, `figures/figure_d3_fixed.png`) — THIS IS THE CURRENT D3 RESULT
Every clip energy-trimmed (remove lead/trail silence) → fixed 2 s central active
segment → peak-normalized, so **length, silence, and loudness are equalized before
WavLM**. Balanced N = 30/group. Controls pass: label-shuffle 0.48, real-vs-real
0.45 (≈chance). WavLM L12 (+PCA-20) vs. the content-free scalar floor **on the
same fixed clips**:

| comparison | WavLM L12+PCA20 | content-free floor | verdict |
|---|---|---|---|
| real vs Bark-token | 0.978 | 0.706 | **WavLM > floor (+0.27)** |
| real vs Bark-inline | 0.956 | 0.772 | **WavLM > floor (+0.18)** |
| real vs Parler-TTS | 0.994 | 0.889 | **WavLM > floor (+0.11)** |
| real vs XTTS | 0.972 | 0.711 | **WavLM > floor (+0.26)** |
| real vs AudioLDM2 | 0.967 | 0.961 | tie — confound-explainable |
| speech vs laughter (sanity) | 1.000 | 0.950 | partly confound |

**Corrected, defensible claim:** after equalizing length/silence/loudness, WavLM
**still separates real from synthetic laughter at 0.96–0.99 for 4 of 5 generators,
exceeding a content-free baseline by 0.11–0.27** — evidence of a representational
difference *beyond* recording artifacts. The exception is **AudioLDM2** (0.97 vs
0.96 floor), whose separation remains confound-explainable, so no content claim
there. Caveat: N=30/group (limited by the smallest synthetic sets), so these are
indicative, not tight. The raw uncontrolled AUC≈1.0 headline is retracted; the
controlled version above replaces it. The confound-controlled UMAP shows real
laughter now *overlapping* Bark-token/XTTS rather than forming clean islands —
honestly reflecting partial, method-dependent separability.

**Effect on the thesis:** the per-generator source-controlled evidence is
promising but is not a generic authenticity result. The stricter
generator-held-out analysis below is the current D3 headline. D1 (evasion) and
the test-time defense are independent of D3 and unaffected.

### D3 generalization and acoustic properties (CURRENT HEADLINE)

`analyze_laughter_dynamics.py` uses manually verified laughter clips with at
least 2 s of active audio. It rejects short clips rather than tiling them,
peak-normalizes every retained clip, and balances the six groups at N=20. It
then trains on real laughter plus four synthesis methods and tests against a
fifth, held-out generator and unseen real clips. This is still a single-source
real-anchor study, not a cross-corpus real-laughter claim.

- **This is not a test of the established spontaneous-versus-volitional
  laughter literature.** That literature reports differences in F0, percentage
  of unvoiced material, harmonicity, and temporal regularity. The present
  exploratory set pools heterogeneous generators, removes full-bout duration
  and onset/offset structure, and uses an HPSS energy ratio rather than a
  standard harmonicity/HNR measurement. Its non-significant pooled effects
  therefore cannot refute those established properties.
- **What the exploratory stress test does show:** this particular small proxy
  set does not yield a *single cross-generator direction*. Spectral flux has
  the largest pooled difference (real 0.500 vs synthetic 0.447, Cliff's delta
  0.357), but it does not survive FDR correction (q=0.109). Descriptor-only
  held-generator AUC is 0.71--0.73 for Bark/Parler, 0.51 for XTTS, and 0.31 for
  AudioLDM2.
- **Literature-aligned full-bout analysis recovers the expected pattern for
  several speech-derived generators.** Per-generator results from
  `analyze_laughter_literature_features.py` show that, relative to Bark-token,
  Bark-inline, and Parler, real laughter has markedly lower voiced fraction
  (0.24 vs 0.52--0.62), longer unvoiced runs (0.93 s vs 0.28--0.67 s), and lower
  autocorrelation harmonicity (for Bark-token/Parler, -2.1 dB vs +2.0/+4.0 dB).
  These are consistent with established spontaneous/authentic-laughter cues.
  AudioLDM2 and XTTS do not share this full profile: AudioLDM2 is primarily
  higher-pitched and brighter, while XTTS has weak or opposite effects. The
  tests are exploratory (N=20/group; uncorrected per-generator p-values), but
  the heterogeneity is the finding: generators miss different parts of the
  genuine laughter-production repertoire.

#### Where AudioLDM2 and XTTS differ

The two apparent exceptions do not invalidate the established acoustic
properties; they fail on different axes and arise from different synthesis
setups.

- **AudioLDM2: excitation/timbre mismatch rather than a voicing mismatch.** It
  is a free-form diffusion audio model prompted with descriptions such as
  "hearty human laughter" and "loud boisterous laughter," not a speech-TTS
  model. Its voiced fraction (0.235 vs 0.239), unvoiced-run duration (0.795 s
  vs 0.933 s), and harmonicity proxy (-2.32 vs -2.06 dB) are close to the real
  anchor. Instead, it is much brighter (spectral centre of gravity 2598 vs
  1916 Hz, p<0.001) and higher-pitched (F0 432 vs 244 Hz, p<0.001). Thus the
  current evidence is that it approximates the broad voiced/unvoiced
  organisation while producing an exaggerated, high-register acoustic texture.
- **XTTS: episode-topology mismatch, with a source-leakage caveat.** XTTS-v2 is
  reference-conditioned on VocalSound laughter, then synthesizes textual
  laughter prompts. It is consequently much closer to the VocalSound real
  anchor on most measured features. Its remaining tendency is toward longer
  unvoiced runs (1.76 vs 0.93 s, p=0.048) and fewer voicing transitions (1.10
  vs 1.72/s, p=0.085): it appears to produce separated, text-like laugh units
  rather than the anchor's denser alternation of calls and gaps. This is
  provisional because only 14/20 XTTS clips yielded usable F0 and because the
  VocalSound reference conditioning can transfer source/channel or speaker
  characteristics into the synthetic clips. XTTS must therefore be evaluated
  with references from a different real corpus before calling it more authentic.

**Mechanistic interpretation:** a useful paper claim is not that every
generator is "too smooth." Speech-derived Bark/Parler outputs are too
continuously voiced and too harmonic; AudioLDM2 is primarily high-pitched and
spectrally bright; reference-cloned XTTS is closer on basic acoustics but has
different laugh-episode topology. This generator-family taxonomy is both more
accurate and more useful for a detector: a laughter-aware system should model
multiple authenticity axes rather than a single synthetic-laughter prototype.
- **WavLM contains a stronger but still method-dependent signal.** With the
  identical held-generator protocol, layer 3 AUC is 0.93 for Bark-inline, 0.90
  for Bark-token, and 0.81 for Parler; layer 12 is strongest for Parler (0.91)
  and XTTS (0.75). AudioLDM2 remains unstable (0.69 +/- 0.19 at layer 3).
  The layer that transfers best changes by generator, so the result is not
  evidence for one universal low-level "synthetic laughter" direction.

**Defensible D3 claim:** several synthesis families produce laughter with
acoustic patterns that are separable from the current verified real anchor in
WavLM, including when the synthesis family is held out. The transferable
signature is generator- and layer-dependent. The nine exploratory proxies do
not yet explain it; they neither establish nor refute the well-supported
spontaneous/volitional laughter mechanisms. The next necessary validation is a
standard Praat-HNR replication of the current harmonicity proxy plus a second
real-laughter corpus.

Artifacts: `tables/laughter_dynamics_effects.csv`,
`tables/laughter_dynamics_held_generator.csv`,
`tables/laughter_dynamics_wavlm_held_generator.csv`, and
`tables/laughter_literature_features.csv`.

## D2 — Synthetic-laughter deepfake dataset (LaughFake)

`data/laughter_dataset/` (README + manifest): **232 clips (~19.8 min) across 4
synthesis engines / 5 method categories**:
| method | clips | type |
|---|---|---|
| bark_laughter_token | 100 | audio-token TTS, true laughter |
| bark_laughs_inline | 40 | inline `[laughs]` in speech |
| parler_tts | 32 | description-conditioned TTS, laugh-like |
| audioldm2 | 30 | diffusion text-to-audio, true laughter |
| xtts | 30 | reference-cloned TTS, spoken-laughter |

Generators span the spectrum from true non-linguistic laughter (AudioLDM2, Bark)
to articulated laugh-speech (XTTS, Parler) — useful diversity for the SSL study.
Scripts: `generate_laugh_bank.py` (Bark), `generate_laugh_oss.py` (Parler),
`generate_laugh_audioldm2.py`, `generate_laugh_xtts.py`; consolidated by
`consolidate_dataset.py`.

- **Measured Bark throughput: ~13 clips/min (~780/hr, ~80 min audio/hr).**
- Real anchor recommendation: VocalSound + AudioSet laughter (MLAAD-style,
  fake-only corpus + external real set).
- Now: release the 140-clip slice + D3 probe benchmark + D1 augmentation
  protocol. Future: scale to ≥4 techniques (2–5k clips); laughter-robust
  detector benchmark; training-time augmentation to close the D1 blind spot.

## Non-goals honored
No retraining; no ITW/VoxCeleb2/Gillick; no ElevenLabs.
