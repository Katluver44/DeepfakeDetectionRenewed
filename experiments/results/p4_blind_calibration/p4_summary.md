# P4 — Blind channel-aware logit correction (no retraining, no representation access)

## Verdict: **PARTIAL / NOT supported**

ĝ maps 10 blind waveform descriptors → channel-induced genuine-logit inflation, fit on MLAAD clean/reverb+MP3 bona only; logit_corr = logit − ĝ(d). ITW/labels never used to fit ĝ or pick the threshold.

### P4.1 In-domain neutralization (held-out MLAAD degraded bona)
median logit inflation (degraded − clean): **+2.01 → -0.02** after correction (101% removed).

### P4.2 / P4.3 ITW transfer + shuffle control
| pipeline | MLAAD-EER | ITW-EER | ITW-FPR | ITW-FNR | ITW-acc | ITW-bal |
|---|---|---|---|---|---|---|
| baseline (no correction) | 0.263 | 0.372 | 0.664 | 0.149 | 0.594 | 0.594 |
| blind correction ĝ(d) | 0.286 | 0.410 | 0.600 | 0.234 | 0.583 | 0.583 |
| shuffled-descriptor control | 0.286 | 0.387 | 0.588 | 0.208 | 0.602 | 0.602 |

ITW false-positive change from blind correction: **-0.064**; the shuffled-descriptor control gives ITW-FPR 0.588 (no real gain) — confirming the correction uses per-utterance channel information, not a global shift.

### Most informative blind descriptors
| descriptor | importance |
|---|---|
| flatness | 0.425 |
| mod_depth | 0.151 |
| reverb_proxy | 0.108 |
| hf_ratio | 0.057 |
| zcr | 0.054 |
| crest | 0.048 |

## Reading
- **In-domain, the blind descriptors work**: 10 cheap signal features (led by spectral flatness,
  modulation depth, reverb proxy) predict the channel-induced genuine-logit inflation well enough
  to cancel it almost perfectly (101% removed) on held-out MLAAD degraded bona. The channel's
  spurious "spoof evidence" is real, blind-measurable, and removable — corroborating E12/E13.
- **But it does NOT transfer to ITW as a per-utterance fix.** On ITW the correction only lowers
  FPR by shifting all scores down (FP↓ 0.064, FN↑ 0.085, EER worse), and the SHUFFLED-descriptor
  control matches/beats it (FPR 0.588, bal 0.602 vs 0.600/0.583). That null-matching is the tell:
  per-utterance descriptor information adds nothing on ITW beyond a global offset.
- **Why:** identical to P1's finding — channel and spoof evidence are entangled, so a *scalar*
  correction (like a *linear* projection) can only slide the operating point, not disentangle.
  The in-domain success + ITW shuffle-null together pinpoint that the gap is disentanglement,
  not measurement — the job of the LEARNED methods (P2 augmentation, P3 adapter).

## Files: p4_pipelines.csv, p4_feature_importance.csv, p4_blind_calibration.png