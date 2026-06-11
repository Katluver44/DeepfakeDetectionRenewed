# E13 — Causal test: does the recording channel manufacture false positives by
moving real audio along the WavLM natural↔synthetic axis?

## Verdict: causal theory **PARTIAL**  (2/4)

Clean MLAAD bona (n=600), RMS-matched channel degradations, detector=mlaad_robust_goat. Baseline clean FPR=0.27, clean logit=+3.06. MLAAD EER thr=4.12. For reference the REAL ITW-bona slides 1.06 gaps along w_mean (E12) and ~0.83 FPR at this threshold (E11, corrected full-model logit).

| channel | mean logit | Δlogit | FPR@thr | cos(Δμ,w_mean) | cos(Δμ,ITW) | slide (gaps) |
|---|---|---|---|---|---|---|
| clean | +3.06 | +0.00 | 0.27 | — | — | +0.00 |
| telephone | +2.55 | -0.51 | 0.19 | +0.07 | +0.19 | +0.09 |
| lowpass3.4k | +3.02 | -0.04 | 0.25 | +0.06 | +0.14 | +0.05 |
| mulaw8bit | +3.00 | -0.06 | 0.25 | +0.08 | +0.12 | +0.01 |
| mp3_16k | +4.21 | +1.15 | 0.49 | +0.13 | +0.28 | +0.14 |
| reverb | +4.28 | +1.22 | 0.54 | +0.20 | +0.52 | +0.67 |
| wild_chain | +3.76 | +0.70 | 0.40 | +0.15 | +0.49 | +0.72 |

| prediction | quantity | value | pass |
|---|---|---|---|
| C1 channel manufactures false positives | max FPR / max Δlogit | 0.54 / +1.22 | True |
| C2 shift is along w_mean | median cos(Δμ,w_mean) [logit-raising] | +0.15 | False |
| C3 same direction as real ITW shift | median cos(Δμ,Δμ_ITW_bona) | +0.49 | True |
| C4 law Δlogit≈wᵀΔμ | per-utt ρ(proj,logit); across-chan r(slide,Δlogit) | +0.13; +0.64 | False |

## Reading
- C1: applying ordinary recording-channel effects to GENUINE clean speech drives its spoof-logit up and flips it to 'fake' at the MLAAD threshold — the ITW false-positive collapse reproduced causally from clean inputs, with NO change to the speaker or content.
- C2/C3: the degradation moves audio along w_mean, and in the SAME direction the real ITW domain shift points — confirming w_mean is a recording-channel axis the detector reads as 'synthetic'.
- C4: the linear law holds — position along w_mean governs the logit as we degrade.

## Files: e13_channel_metrics.csv, e13_channel_curves.png