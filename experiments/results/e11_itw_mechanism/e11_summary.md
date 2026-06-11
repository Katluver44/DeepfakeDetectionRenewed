# E11 — Why is In-The-Wild harder than MLAAD? (mechanistic decomposition)

## Verdict (mechanisms adjudicated)
- **M1 bona-fide domain shift: SUPPORTED** — FP share of ITW error = 1.00 (FN share 0.00); bona logit shift vs MLAAD = +6.84 (Cliff +0.955).
- **M2 spoof realism: ITW spoofs MORE bona-like** — spoof-vs-bona AUC ITW=0.619 vs MLAAD=0.783; spoof logit shift=+4.37.
- **User's 'less-advanced TTS in some cases': NOT supported** — latent crude/easy tier present=False; ITW-spoof logits multimodal=yes.
- **M3 off-manifold shift: weak** — ITW kNN dist 29.55 vs MLAAD-self 26.65.
- **M4 recording-quality (bona): SUPPORTED** — ρ(bona logit, SNR)=-0.313.

**Bottom line:** ITW difficulty is dominated by FALSE POSITIVES on real audio (bona-fide domain shift), not by spoof quality — the 'less-advanced TTS' framing targets the wrong class.

## Part A — Error decomposition (FP = real flagged fake; FN = spoof passed as real)
AUC: ITW(mlaad_det)=0.619  ITW(robust_goat)=0.779  MLAAD-indomain=0.783
EER: ITW(mlaad_det)=0.415  ITW(robust_goat)=0.288  MLAAD-indomain=0.272

| scenario | thr | FPR (bona→spoof) | FNR (spoof→bona) | acc | FP share | FN share |
|---|---|---|---|---|---|---|
| MLAAD_indomain@EERthr | 1.64 | 0.272 | 0.273 | 0.728 | 0.49 | 0.51 |
| ITW@MLAAD_EERthr(transfer) | 1.64 | 0.987 | 0.002 | 0.506 | 1.00 | 0.00 |
| ITW@MLAAD_Youden(transfer) | 1.67 | 0.986 | 0.002 | 0.506 | 1.00 | 0.00 |
| ITW@own_EERthr(MLAADdet) | 7.50 | 0.415 | 0.415 | 0.585 | 0.50 | 0.50 |
| ITW@own_EERthr(robust_goat) | 6.23 | 0.288 | 0.288 | 0.712 | 0.50 | 0.50 |

## Part B — Which class moved? (logit space, MLAAD detector; thr=1.64)
| comparison | ITW med | MLAAD med | shift | Cliff δ | Cohen d | KS | ITW>thr | MLAAD>thr |
|---|---|---|---|---|---|---|---|---|
| bona: ITW vs MLAAD | 7.08 | 0.24 | +6.84 | +0.955 | +3.154 | 0.865 | 0.987 | 0.272 |
| spoof: ITW vs MLAAD | 7.87 | 3.50 | +4.37 | +0.793 | +2.079 | 0.639 | 0.998 | 0.727 |

## Part C — Latent spoof heterogeneity (no source labels ⇒ inferred tiers)
k-means k=4 on WavLM L12+L9 (GMM-BIC suggested 6). within-EER = spoof-cluster vs the ITW bona pool (lower = easier to detect).
| cluster | n | med logit (mlaad) | med logit (rg) | within-EER | mean C | mean T | mean SNR |
|---|---|---|---|---|---|---|---|
| 2 | 1602 | +8.43 | +7.94 | 0.345 | -10.85 | 2.724 | 24 |
| 1 | 584 | +7.46 | +7.74 | 0.451 | -10.80 | 2.760 | 77 |
| 3 | 163 | +7.58 | +7.36 | 0.466 | -9.90 | 2.682 | 43 |
| 0 | 651 | +6.65 | +7.47 | 0.551 | -10.61 | 2.736 | 46 |

- ITW-spoof logit modality: mlaad GMM-BIC n_modes=2 (bimod.coeff=0.371); robust_goat n_modes=2 (bc=0.497).  (>0.555 ⇒ multimodal)
- spoof-vs-bona separability (AUC): ITW(mlaad)=0.619 ITW(robust_goat)=0.779 vs MLAAD-indomain=0.783.  Lower ITW AUC ⇒ ITW spoofs are MORE bona-like (more realistic).
- crude/easy tier present? **False** (easiest cluster within-EER=0.345, n=1602).

## Part D — Off-manifold gap (kNN dist to MLAAD-spoof reference cloud, WavLM-L12)
- mean kNN dist: MLAAD-spoof(self)=26.65  ITW-spoof=29.55  ITW-bona=32.14  ⇒ both ITW classes sit near the reference manifold.
- ρ(dist-to-ref, spoof logit) = -0.298 (more off-manifold ⇒ harder).

## Part E — Recording-quality confound on bona-fide (FP = real flagged fake)
- ρ(ITW-bona logit, SNR) = -0.313 (negative ⇒ noisier real audio scored MORE spoof-like).
| SNR quintile | med dB | FP rate | n |
|---|---|---|---|
| Q1 | 12 | 0.971 | 616 |
| Q2 | 17 | 0.996 | 703 |
| Q3 | 23 | 0.994 | 485 |
| Q4 | 30 | 0.995 | 618 |
| Q5 | 54 | 0.978 | 578 |
- bona FP-rate concentrated in 21/21 speakers (>50% FP); top speaker FP=1.00 (Winston Churchill).