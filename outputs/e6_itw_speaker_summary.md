# E6 ITW Speaker — Does C/T predict ITW difficulty at the speaker level?

Source: `outputs/e5_itw_utterance_ct.csv` (no re-extraction). Speakers with n_spoof≥5: **45**. Primary checkpoint: `mlaad_robust_goat` (MLAAD-analogue comparator).
Sign convention: C/T higher = harder ⇒ a **positive** rho with EER means the mechanism transfers (this is EER-space; E5 reported logit-space, opposite sign).

## Verdict: speaker-level mechanism **SUPPORTED** (C only)

## 1. Per-speaker C/T → EER (Spearman across speakers)
| checkpoint | rho(meanC, EER) | rho(meanT, EER) |
|---|---|---|
| mlaad_robust_goat | +0.437 (p=0.0027, n=45) | -0.021 (p=0.89) |
| robust_goat | +0.384 (p=0.0093, n=45) | -0.184 (p=0.23) |
| **MLAAD (per-system ref, n=63)** | +0.368 (p=0.003) | +0.362 (p=0.0035) |

## 2. Quartile stratification (primary ckpt; Q4 = highest C/T = predicted hardest)
| axis | Q1 | Q2 | Q3 | Q4 | Q4>Q1? |
|---|---|---|---|---|---|
| C (ITW speakers) | 0.327 | 0.411 | 0.404 | 0.501 | yes |
| T (ITW speakers) | 0.437 | 0.380 | 0.422 | 0.395 | no |
| C (MLAAD systems) | 0.222 | 0.216 | 0.272 | 0.337 | yes |
| T (MLAAD systems) | 0.199 | 0.266 | 0.232 | 0.347 | yes |

## 3. Between- vs within-speaker decomposition of E5's per-sample rho(C, logit)
| checkpoint | pooled (per-sample) | between-speaker | within-speaker |
|---|---|---|---|
| mlaad_robust_goat | -0.177 | -0.465 (p=0.0013) | -0.155 (p=1.7e-17) |
| robust_goat | -0.155 | -0.364 (p=0.014) | -0.186 (p=1.3e-24) |

- C variance that is between-speaker (ICC-like): **0.104**  |  T: 0.124
- Dominant component (primary ckpt): **between-speaker**

_RESCUE: at speaker granularity, C/T **does** predict ITW detection difficulty (C only), with the same sign as MLAAD's per-system effect. E5's per-sample dissociation was partly a **granularity artifact** — the C/T mechanism operates between speakers even though the ITW spoof population does not overlap MLAAD's absolute hard region._

## Figures
- `figures/e6_itw_speaker_ct_eer.png` — per-speaker C/T vs EER scatter (+trend, +rho)
- `figures/e6_itw_speaker_quartile_eer.png` — EER by C/T quartile, ITW vs MLAAD
- `figures/e6_itw_within_between.png` — within- vs between-speaker decomposition

## Outputs
- `e6_itw_speaker_ct.csv` — per-speaker mean C/T, n, EER (both ckpts, shared-pool + within)