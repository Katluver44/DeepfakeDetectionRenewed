# E8 ASVspoof — Does C/T predict per-attack detection hardness?

ASVspoof 2019 LA **eval split A07–A19** (13 attacks), 300/attack spoof + 3000 bona-fide.
C/T from frozen WavLM (microsoft/wavlm-base, L12/L9). Per-attack EER scored vs the shared
bona-fide pool. Logit convention (verified): **higher logit = more spoof-like = correctly
flagged = easier**; bonafide mean logit ≈ −5.5, spoof mean ≈ +3.5.

**Model framing (corrected):** robust_GOAT was trained on ASVspoof train+test (A01–A19), so
A07–A19 are **in-distribution held-out utterances** for it → this is the true analogue of
the MLAAD in-distribution C/T result. GOAT (standard A01–A06 protocol) sees A07–A19 as
**unseen** → a generalization probe. They are reported side-by-side but measure different
constructs; the in-distribution one (robust_GOAT) is the apples-to-apples MLAAD comparison.

## Verdict: **C predicts hardness in ASVspoof — but with the OPPOSITE polarity to MLAAD. T does not transfer.**

- In **MLAAD**, higher C (more compact) → **harder** (ρ(C,logit) ≈ −0.15 on spoof).
- In **ASVspoof**, higher C → **easier** (ρ(C,logit) = **+0.16/+0.18**, p≈1e-24/1e-31; between-attack ρ ≈ +0.50).
- So C is a real, high-power difficulty axis in *both* corpora, but its sign is **corpus-dependent**, not universal.
- **T** is essentially inert in ASVspoof (per-utterance ρ ≈ −0.04; per-attack ρ ≈ 0) — same fragility seen for ITW.

## 1. Per-attack correlations (Spearman ρ, 95% bootstrap CI, permutation p; n=13)
| model | metric | axis | ρ | CI | perm p | Pearson r | R²(C+T) | LOO |
|---|---|---|---|---|---|---|---|---|
| GOAT (unseen) | EER | C | -0.38 | [-0.85,+0.31] | 0.20 | -0.38 | 0.19 | -0.44 |
| GOAT (unseen) | EER | T | +0.02 | [-0.57,+0.67] | 0.97 | +0.03 | 0.19 | -0.44 |
| GOAT (unseen) | AUC | C | +0.40 | [-0.28,+0.83] | 0.17 | +0.42 | 0.18 | -0.40 |
| GOAT (unseen) | AUC | T | -0.12 | [-0.76,+0.51] | 0.71 | -0.16 | 0.18 | -0.40 |
| GOAT (unseen) | Acc | C | +0.41 | [-0.24,+0.82] | 0.16 | +0.45 | 0.22 | -0.18 |
| GOAT (unseen) | Acc | T | -0.01 | [-0.62,+0.56] | 0.98 | -0.14 | 0.22 | -0.18 |
| robust_GOAT (in-dist) | EER | C | -0.43 | [-0.82,+0.24] | 0.14 | -0.43 | 0.21 | -0.37 |
| robust_GOAT (in-dist) | EER | T | -0.21 | [-0.80,+0.44] | 0.50 | +0.10 | 0.21 | -0.37 |
| robust_GOAT (in-dist) | AUC | C | +0.43 | [-0.22,+0.84] | 0.14 | +0.41 | 0.17 | -0.44 |
| robust_GOAT (in-dist) | AUC | T | +0.19 | [-0.46,+0.76] | 0.55 | -0.25 | 0.17 | -0.44 |
| robust_GOAT (in-dist) | Acc | C | +0.55 | [-0.05,+0.89] | 0.05 | +0.47 | 0.23 | -0.32 |
| robust_GOAT (in-dist) | Acc | T | +0.08 | [-0.54,+0.71] | 0.81 | -0.16 | 0.23 | -0.32 |

Per-attack ρ(C, EER) is **negative** (−0.38/−0.43) — i.e. **more compact attacks are EASIER**
(reversed vs MLAAD's +0.29). At n=13 the CIs span 0 (perm p 0.14–0.20), so the per-attack
test alone is underpowered; the per-utterance analysis below is the decisive one.

_MLAAD per-system reference (n=63): ρ(C,resid)=+0.29, ρ(T,resid)=+0.33 — higher C/T → harder._

## 2. Quartile stratification (headline EER; Q4 = highest C = most compact)
| axis | Q1 | Q2 | Q3 | Q4 | direction |
|---|---|---|---|---|---|
| C | 0.140 | 0.104 | 0.090 | 0.072 | **monotone: compact (Q4) = EASIEST** (reversed vs MLAAD) |
| T | 0.101 | 0.100 | 0.122 | 0.096 | flat (no effect) |

## 3. Per-utterance (high power, n=3900 spoof) C/T → logit
Higher logit = more spoof-like = easier. **ρ(C,logit) > 0 ⇒ compact spoofs easier to flag.**
| model | axis | ρ(C/T, logit) | CI | p | n |
|---|---|---|---|---|---|
| GOAT | C | **+0.161** | [+0.129,+0.192] | 3.5e-24 | 3900 |
| GOAT | T | -0.044 | [-0.077,-0.013] | 5.6e-03 | 3900 |
| robust_GOAT | C | **+0.184** | [+0.151,+0.215] | 5.3e-31 | 3900 |
| robust_GOAT | T | -0.042 | [-0.073,-0.009] | 8.6e-03 | 3900 |

C is strongly, significantly predictive at the utterance level (opposite sign to MLAAD's
≈−0.15). T's effect is negligible (|ρ|<0.05) despite a small p inflated by n=3900.

## 4. Between- vs within-attack decomposition (C → logit)
| model | pooled | between-attack | within-attack |
|---|---|---|---|
| GOAT | +0.161 | **+0.511** | +0.092 |
| robust_GOAT | +0.184 | **+0.495** | +0.122 |

- C variance between-attack (ICC-like): **0.127**.
- Like MLAAD/ITW, the C effect is **mostly between-system** (between-attack ρ ≈ 0.50, ~5×
  the within-attack ρ): compactness is an attack-level property, not a clip-level one.

## Interpretation — why the polarity flips
C measures WavLM representational **compactness**, not hardness per se. What "compact" *means*
differs by corpus:
- **MLAAD (modern multilingual TTS):** compact systems are the smooth, natural neural-TTS
  ones that closely mimic real speech → **hard**.
- **ASVspoof 2019 (legacy A07–A19: vocoder/VC/Griffin-Lim/WORLD era):** compact systems are
  the crude, artifact-heavy ones with monotonous representations → **easy** (their artifacts
  are obvious to the detector).

So C is a **descriptor whose hardness-polarity depends on the attack population**, not a
universal "harder" axis. T does not generalize to ASVspoof at all. This refines the roadmap
claim: the C/T → hardness mapping established on MLAAD is **corpus-specific in sign**, and
should not be assumed to point the same way on a corpus with a different generation-tech mix.

## Files
- `asvspoof_per_attack.csv`, `asvspoof_correlations.csv`, `asvspoof_utterance_ct.csv`
- `asvspoof_utterance_corr.csv`, `asvspoof_decomposition.csv`
- figures: `asvspoof_per_attack_ct_eer.png`, `asvspoof_per_utterance_ct_logit.png`
