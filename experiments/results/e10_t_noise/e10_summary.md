# E10 (Test 1) — Is T's ITW failure a background-noise artifact?

Per-sample WADA-SNR added to E5's 6000 ITW clips (detector: logit_robust_goat). Observational test; no re-synthesis. SNR rank carrier = WADA v3; dB is display-scale.

## Verdict: **NOT supported**
- (1) T noise-driven (|rho(T,SNR)|>=0.20): **False**  (rho=+0.105)
- (2) T recovers in clean half (strength 0.096 > noisy 0.103+0.03): **False**
- (3) C comparatively noise-robust (gap smaller than T's): **False**

## (a) Is T just measuring noise?  Spearman vs WADA-SNR
| factor | all samples | spoof only |
|---|---|---|
| **T** | +0.105 (p=4.0e-16) | +0.139 (p=1.7e-14) |
| C (control) | -0.004 (p=7.6e-01) | -0.007 (p=7.2e-01) |

## (b) Stratified discriminativity — clean (hi-SNR) vs noisy (lo-SNR) half
| stratum | n_sp | med dB | rho(T,logit) | Cliff d T(sp>bo) | rho(C,logit) | Cliff d C(sp>bo) |
|---|---|---|---|---|---|---|
| ALL | 3000 | 25 | +0.004 (p=8.4e-01) | -0.077 (p=2.2e-07) | -0.156 (p=9.5e-18) | -0.236 (p=1.4e-56) |
| CLEAN (hi-SNR) | 1755 | 39 | +0.050 (p=3.5e-02) | -0.096 (p=7.7e-06) | -0.139 (p=5.0e-09) | -0.302 (p=3.4e-45) |
| NOISY (lo-SNR) | 1245 | 17 | -0.024 (p=4.1e-01) | -0.103 (p=1.6e-06) | -0.195 (p=3.9e-12) | -0.179 (p=5.2e-17) |

## (c) Partial correlation — does signal survive controlling for SNR?
| relation | raw Spearman | partial | Spearman (control SNR) |
|---|---|---|---|
| T -> logit (spoof) | +0.004 | +0.022 (p=2.2e-01) | |
| C -> logit (spoof) | -0.156 | -0.158 (p=3.2e-18) | |
| T -> label (all)   | -0.067 | -0.091 (p=1.6e-12) | |
| C -> label (all)   | -0.205 | -0.208 (p=7.3e-60) | |

## (d) Per-quintile trend (see e10_snr_quintile_trend.csv / .png)
| quintile | med dB | n_sp | rho(T,logit) | Cliff d T | Cliff d C | mean T |
|---|---|---|---|---|---|---|
| Q1 | 12 | 495 | -0.024 | -0.113 | -0.221 | 2.715 |
| Q2 | 18 | 489 | +0.027 | -0.082 | -0.107 | 2.732 |
| Q3 | 25 | 528 | -0.056 | -0.134 | -0.255 | 2.746 |
| Q4 | 35 | 672 | +0.051 | -0.136 | -0.335 | 2.740 |
| Q5 | 100 | 816 | +0.066 | -0.018 | -0.360 | 2.745 |

## Reading
- If T is noise-driven AND only discriminates in the clean half (while C holds), the hypothesis is supported: ITW's T-failure is a background-noise/artifact effect, not a statement that temporal dynamics are irrelevant to spoofing.
- If T is dead in BOTH halves and barely tracks SNR, noise is NOT the explanation — T's ITW failure is intrinsic (domain shift / representation mismatch), echoing E9 where T was not an isolable causal lever.
- This is the cheap observational test; the causal confirmation is Test 2 (inject controlled noise into clean MLAAD and watch T's signal collapse).

## Files
- `e10_itw_ct_snr.csv` — per-sample C/T/logits + WADA SNR
- `e10_snr_quintile_trend.csv`, `e10_ct_vs_snr.png`, `e10_snr_quintile_trend.png`
---

# E10 (Test 2) — Causal noise injection into clean MLAAD

Clean source: spoof=500 bona=500. Noise: babble (sum of 5 bona clips) + gaussian, SNR grid [20, 15, 10, 5, 0] dB. Factors on standalone wavlm-base. Detector=mlaad_robust_goat.ckpt.

## Verdict: NOISE HYPOTHESIS NOT SUPPORTED (causal side): T has ~no intrinsic spoof/bona signal even on clean MLAAD (AUC=0.534), AND noise does NOT inflate T toward ITW (mean T moves -0.070). Any rho(T,logit) drop is generic detector collapse (EER +0.206, C-corr degrades too), not a T-specific noise effect.

- Clean baseline: mean T=2.768 (ITW spoof ref 2.731), AUC_T=0.534 (~chance), Cliff_T=+0.067, rho(T,logit)=-0.191, EER=0.356
- Strongest noise (0 dB): mean T moves -0.070 (babble=2.656, gauss=2.698) — i.e. AWAY from ITW, not toward it; T AUC stays 0.509/0.501 (~chance)
- Detector collapses generically: EER 0.356 -> 0.562/0.508; rho(C,logit) degrades alongside rho(T,logit), so the rho(T,logit) drop is not T-specific.

## Full sweep

| noise | SNR | mean T | AUC T | Cliff T | rho(T,logit) | AUC C | Cliff C | EER |
|---|---|---|---|---|---|---|---|---|
| clean | clean | 2.768 | 0.534 | +0.067 | -0.191 | 0.500 | -0.001 | 0.356 |
| babble | 20 | 2.752 | 0.508 | -0.016 | -0.043 | 0.517 | +0.034 | 0.414 |
| babble | 15 | 2.739 | 0.509 | -0.019 | -0.024 | 0.516 | -0.032 | 0.438 |
| babble | 10 | 2.719 | 0.506 | +0.012 | -0.045 | 0.533 | -0.066 | 0.490 |
| babble | 5 | 2.694 | 0.511 | +0.022 | +0.011 | 0.502 | -0.004 | 0.524 |
| babble | 0 | 2.656 | 0.509 | -0.017 | +0.161 | 0.552 | +0.103 | 0.562 |
| gaussian | 20 | 2.771 | 0.553 | -0.106 | -0.133 | 0.544 | -0.088 | 0.418 |
| gaussian | 15 | 2.762 | 0.546 | -0.092 | -0.152 | 0.552 | -0.104 | 0.428 |
| gaussian | 10 | 2.752 | 0.500 | -0.001 | -0.088 | 0.560 | -0.119 | 0.426 |
| gaussian | 5 | 2.735 | 0.519 | -0.038 | -0.081 | 0.570 | -0.140 | 0.436 |
| gaussian | 0 | 2.698 | 0.501 | +0.002 | +0.111 | 0.565 | -0.129 | 0.508 |

## Reading
- The proposed mechanism is 'noise inflates/scrambles velocity entropy T'. It fails twice: (i) T barely separates spoof/bona even on CLEAN MLAAD (AUC~0.53), so there is essentially no T signal for noise to destroy; (ii) adding noise LOWERS mean T (broadband noise makes frame-to-frame velocities uniformly large -> a more peaked velocity histogram -> lower entropy), moving T AWAY from the ITW value, not toward it.
- The one quantity that shrinks with noise, rho(T,logit), is confounded: the detector itself collapses to chance (EER -> ~0.5) and rho(C,logit) degrades in step, so this is generic noise-degradation of the detector, not a T-specific effect.
- Combined with Test 1 (T does not track SNR within ITW; T dead even in clean ITW clips), background noise is NOT the mechanism behind T's ITW failure from either the observational or the causal side.

## Files
- `e10_inject_metrics.csv`, `e10_inject_curves.png`