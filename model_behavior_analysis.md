# Model Behavior Analysis — Companion to *Low-Cost Geometric Directions in Frozen Speech Models*

**Scope.** This document is a behavioral analysis of the three detectors studied in the COLM submission
([final_submission/main.tex](final_submission/main.tex)) and of the training-free *axis fusion*
intervention applied to them:

1. **WavLM-GAT on ASVspoof 2019 LA** — the primary in-domain detector, evaluated at two capacities
   (a near-ceiling "strong" checkpoint and a deliberately undertrained 500-utterance "weak" one).
2. **WavLM-GAT on MLAAD** — the same architecture retrained on 61-system multilingual TTS, sitting
   far from ceiling; this is where the geometry→hardness *association* is established.
3. **AASIST evaluated cross-domain** — a spectro-temporal detector *not* built on WavLM, used as a
   naturally-weak, architecture-independent probe.

Everything below is grounded in the committed training logs
([training_logs/](training_logs/)) and the fusion artifacts
([experiments/results/i7_axis_fusion/](experiments/results/i7_axis_fusion/)); no numbers are recomputed
here that are not already in those files or in `main.tex`.

**Why this framing (COLM alignment).** COLM rewards work that *understands, improves, and critiques* the
behavior of models built on frozen foundation representations. Anti-spoofing detectors are a clean
instance: they are decision heads bolted onto a frozen ~95M-parameter SSL encoder (WavLM), and their two
notorious failure modes — per-system evasion and cross-domain collapse — are exactly the kind of behavior
that is usually reported as an aggregate EER and left unexplained. This analysis follows the venue's
preferred shape: (i) characterize the behavior mechanistically, in the geometry of the frozen
representation; (ii) state falsification conditions and report the tests that failed; (iii) apply a
low-cost intervention (axis fusion) and show *where it works and where it does not* rather than only where
it wins. The compute envelope (one GPU, largest run ≈10¹⁶ FLOPs, under the 10²⁰ / 3B-parameter caps) is
itself part of the contribution: the claims are small-scale by construction.

---

## 1. The behavioral question

All three detectors report a headline EER, but that number hides two structured behaviors:

- **Per-system inconsistency.** Against a *fixed* detector, some synthesis systems evade far more often
  than others. Per-system hardness (`1 − AUC` of the detector's scores for that system against a shared
  bona-fide pool) spans nearly the whole `[0,1]` range on MLAAD.
- **Cross-domain inconsistency.** A detector tuned on one recording domain often collapses on another
  (WavLM-GAT trained on ASVspoof degrades on MLAAD/ITW; AASIST is near-ceiling on ASVspoof-2021 but weak
  on MLAAD and ITW).

The claim under analysis is that **both behaviors are legible in a single frozen-representation direction**
— the *natural-versus-synthetic axis* `w = centroid(synthetic) − centroid(bona)` — and that a system's
**spread along `w`** (`sd_along`), not its distance from the boundary, is what predicts evasion. This
places the work in the linear-representation-hypothesis program (Bolukbasi 2016; Arditi 2024): a
security-relevant behavior recoverable as one direction in a frozen encoder.

---

## 2. Model-by-model behavior

### 2.1 WavLM-GAT on ASVspoof 2019 LA

**Architecture.** Frozen WavLM-base (94.7M params) → adaptive phoneme pooling → graph-attention network →
BiLSTM → linear head. Only the ~13M head trains. This is the phoneme-GAT detector of Zhang et al. (2025).

**Capacity is the dominant behavioral variable.** The same architecture behaves completely differently at
two training capacities, and this dissociation is the cleanest single fact in the paper:

| Checkpoint | Train set | EER (in-domain) | Behavioral regime |
|---|---|---|---|
| Strong | full | **0.078** | near-ceiling; almost no exploitable headroom |
| Weak | 500 utts | **0.144** | undertrained; systematically under-reads the axis |

The strong detector has effectively *already internalized* the natural-vs-synthetic direction in its head
weights; the weak one has not. This is the substrate for the headroom law in §3.

**Hardness is carried by *position*, not spread.** ASVspoof 2019 LA has only ~13 attacks, but per-attack
*position* along `w` varies over a ~3.5× wider range than on MLAAD. Consequently the operative statistic
flips: `s_along` (mean projection) `ρ = −0.71` (`p = 0.006`, 13 attacks). This is the same axis-rotation
phenomenon seen "from the hardness side" — few-system corpora expose position; many-system corpora expose
spread.

**Provenance caveat.** The strong ASVspoof checkpoint (`models/robust_goat.ckpt`) has no surviving wandb
run folder; its training log is reconstructed from `robust.ipynb` cell outputs
([training_logs/asv19_wavlm_gat_robust/README.txt](training_logs/asv19_wavlm_gat_robust/README.txt)). A
seed-7 replication ([...seed7_replication/](training_logs/asv19_wavlm_gat_robust_seed7_replication/))
exists for the hardness-ranking stability claim.

### 2.2 WavLM-GAT on MLAAD

**Same architecture, harder regime.** Retrained on MLAAD-tiny (61 TTS systems, multilingual, ≥8
utterances/system) with reverberation augmentation. Deviations from the ASVspoof recipe are logged in
[mlaad_wavlm_gat_hyperparameters.json](training_logs/mlaad_wavlm_gat/mlaad_wavlm_gat_hyperparameters.json):
center-crop instead of random-crop, no train-sample cap, CSV/JSON logging instead of wandb.

**This model lives far from ceiling — which is what makes it useful.** From
[mlaad_wavlm_gat_final_metrics.json](training_logs/mlaad_wavlm_gat/mlaad_wavlm_gat_final_metrics.json):

| Variant | Best val-EER | Best epoch | Train time |
|---|---|---|---|
| `mlaad_goat` (regular) | **0.2795** | 5 | 22m19s |
| `mlaad_robust_goat` | **0.2858** | 5 | 22m16s |

A behavioral red flag worth naming: the last-epoch confusion stats show `val-tpr = 1.0`, `val-tnr = 0.0`
— the raw head at threshold-0 is degenerate (predicting one class), and the usable signal is entirely in
the *ranking* (val-AUC ≈ 0.74), which is why EER (threshold-swept) is the honest metric and why there is
so much headroom for a corrective signal to exploit.

**Hardness is carried by *spread*.** This is the central association:
`sd_along → hardness`, Spearman **ρ = 0.59** (`p < 10⁻⁶`), LOSO **R² = 0.277** (permutation `p = 2.5×10⁻⁴`,
bootstrap CI [0.41, 0.73]). It is the **strongest of 83 quantities** and the **only** survivor of a single
global Benjamini–Hochberg correction (`q ≈ 3×10⁻⁵`; [main.tex Table 1](final_submission/main.tex)).

**The behavior is directional, not "diverse systems are just hard."** The decisive control:

- Isotropic (axis-free) total variance — radius of gyration — predicts *poorly and with the wrong sign*
  (`ρ = −0.31` at L12, `−0.38` at L0; fails FDR).
- Mean position is weak (`s_along ρ = −0.26`; centroid norm `+0.29`).
- Best non-axis runner-up (trajectory `vel_entropy`) `ρ = +0.35`, dies under correction (`q = 0.072`).

So the behavior is specifically: **hard systems straddle the boundary along `w`**, leaking mass into the
bona-fide region despite sometimes sitting *farther* out. The `R² = 0.277` is ≈32% of the
reliability-corrected ceiling (`R²_max = 0.863`), not 32% of raw variance — an important framing for
avoiding an overclaim.

**Robustness of the behavioral claim** (from [main.tex App. D](final_submission/main.tex)): survives
leave-one-system jackknife (0/61 folds lose significance), loudness partialling (`ρ|RMS = 0.58`),
utterance-count partialling, and estimator choice; shows the *attenuation signature* of a real effect
(`ρ = 0.59 / 0.64 / 0.69` at ≥8 / 12 / 16 utts/system — the effect strengthens as measurement noise
falls).

### 2.3 AASIST evaluated cross-domain

**Why it is in the study.** AASIST is a spectro-temporal detector with *no WavLM front-end*. It serves two
behavioral roles: (i) an architecture-independent test of the hardness ranking, and (ii) a *naturally*
weak detector, to answer the obvious objection that the ASVspoof "weak" detector is only weak because it
was crippled on purpose.

**Behavior: strong in-family, collapses cross-domain.**

| Domain | AASIST EER | Regime |
|---|---|---|
| ASVspoof-2021 (in-family) | 0.073 | near-ceiling |
| MLAAD (cross-domain) | 0.376 | naturally weak |
| ITW (cross-domain) | 0.486 | ≈ chance |

**The ranking replicates across architecture — but weakly, and only as rank.** WavLM-derived `sd_along`
*ranks* AASIST per-system hardness (`ρ = +0.35`, `p = 0.006`) on a near-independent target (AASIST hardness
correlates only ≈0.55 with WavLM-GAT's), but the out-of-sample predictive form is **null**
(`LOSO R² = −0.032`, `p = 0.13`). The honest reading, stated in the paper: *rank replication across
detector families, not transfer of the predictive law.* This is exactly the kind of split (`ρ` positive,
`R²` null) that reviewers flag as confusing, and the paper pre-empts it in Q8 rather than hiding it.

---

## 3. How fusion alters model behavior

The intervention is a training-free score fusion:

```
s_fused = z(logit) + λ · z(axis_projection)
```

No detector retraining, no per-system difficulty labels. To prevent leakage, the `z`-statistics and `λ`
are fit on each **system-disjoint fold's training partition only**. `λ` is remarkably stable — it takes
only the values `{1.5, 2.0}` across all 5 folds × 3 seeds
([i7_stats.json](experiments/results/i7_axis_fusion/i7_stats.json)), so the effect is not a
tuning artifact.

The behavioral question is not "does fusion lower EER" (it usually does) but **what does it change about
the model, and on which inputs**. Three findings.

### 3.1 Headroom decides — fusion is a corrective, not additive, signal

Fusion moves a detector *toward* a behavior it was failing to express, and does nothing to a detector
that already expresses it:

| Detector | Baseline EER | Fused EER | Δ | Verdict |
|---|---|---|---|---|
| WavLM-GAT / ASVspoof, **weak** (500 utts) | 0.144 | 0.114 | −0.030 (≈21%), CI [−0.046, −0.013], p<10⁻³ | **helps** (all 5 folds) |
| WavLM-GAT / ASVspoof, **strong** | 0.078 | ~0.078 | −0.0004 to +0.004, p≈0.56–0.80 | **no change** |
| WavLM-GAT / MLAAD (in-domain, far from ceiling) | 0.266 | 0.157 | −0.109, CI [−0.129, −0.091], p≈0 | **helps** (15/15 fold×seed cells) |
| AASIST / MLAAD (naturally weak) | 0.376 | 0.116 | −0.26 | **helps** |
| AASIST / ITW (naturally weak) | 0.486 | 0.161 | −0.33 | **helps** |
| AASIST / ASVspoof-2021 (strong) | 0.073 | 0.076 | +0.003 | **no change** |

MLAAD numbers from [i7_headline.csv](experiments/results/i7_axis_fusion/i7_headline.csv) /
[i7_stats.json](experiments/results/i7_axis_fusion/i7_stats.json) (main + s42 + s1024 seeds); ASVspoof and
AASIST from `main.tex` §5. **The behavioral law is monotone in headroom, across two architectures and four
domains.** The AASIST cross-domain result is what defeats the "you only fixed a detector you broke
yourself" objection: AASIST is weak for real reasons and fusion still repairs it by 26–33 EER points.

### 3.2 The gain lands exactly on the inputs the mechanism predicts

If the story is "hard systems spread along a direction the detector under-weights," then re-injecting that
direction should help *precisely the hardest systems* and barely touch the easy ones. It does
([i7_quartiles.csv](experiments/results/i7_axis_fusion/i7_quartiles.csv), by MLAAD detector-difficulty
quartile):

| Quartile | AUC detector-alone | AUC fused | Δ AUC |
|---|---|---|---|
| Q1 (easiest) | 0.950 | 0.985 | **+0.036** |
| Q2 | 0.870 | 0.952 | +0.082 |
| Q3 | 0.769 | 0.905 | +0.136 |
| Q4 (hardest) | 0.617 | 0.837 | **+0.220** |

The gain grows **monotonically** with difficulty — a factor of ~6× from easiest to hardest quartile. This
is the closest thing to a *causal* test in the study: the intervention recovers the systems the mechanism
says it should, and only those. It is also the paper's stated falsification condition — flat gains across
quartiles would have falsified the directional mechanism.

### 3.3 Where fusion does *not* alter behavior — the negative results

A COLM-style analysis is incomplete without the failures, and here they are load-bearing:

- **Cross-domain transfer of the axis fails.** The MLAAD-estimated axis, applied to ITW, is *worse than
  the ITW baseline* — ITW EER 0.363 → 0.518 for axis-alone, and fusion only degrades it
  (0.363 → 0.373–0.423 as λ rises;
  [itw_fusion_test.json](experiments/results/i7_axis_fusion/itw_fusion_test.json)). The reason is
  geometric: `cos(w_MLAAD, w_ITW) = 0.05`, i.e. the axes are **near-orthogonal**. The axis rotates across
  corpora (cross-corpus cosines sit in the ±0.07 chance band), so a transferred direction carries no
  usable information. This *is* the geometric correlate of cross-domain detector collapse.
- **The position law is domain-local, but it does hold in-domain everywhere.** An ITW-*internal* axis
  (speaker-disjoint) alone beats the fine-tuned ITW detector by ~7 EER points (0.363 → 0.292). So the
  mechanism is not MLAAD-specific — it recurs within each domain — but the *direction* must be
  re-estimated per domain. Fusion with the internal axis still slightly *hurts* here (0.292 → 0.312),
  because the ITW detector is already reading its own domain axis.
- **The AASIST cross-domain "repair" is not a free lunch.** The operative cross-domain axis there is a
  *supervised in-corpus LDA probe* on the frozen embeddings, which *alone* matches or beats the fused
  score (MLAAD 0.100, ITW 0.101). So the honest claim is "in-domain axis information helps a weak
  detector," not "a zero-label direction transfers for free."
- **Voice conversion breaks the whole predictor.** On VC attacks all predictors give `ρ ≈ 0` — converted
  speech inherits content and spread from its source, so the natural-vs-synthetic geometry does not apply.

**Net behavioral characterization of fusion:** it is a *headroom-conditional, difficulty-targeted,
domain-local corrective*. It nudges a weak detector toward a direction the frozen encoder already
represents but the head under-weights; it cannot manufacture information that is not in the target-domain
representation, and it cannot import a direction across a domain boundary.

---

## 4. Falsification conditions and honest negatives (COLM rigor apparatus)

The analysis is structured so that each behavioral claim has a stated way it could have failed:

| Claim | Would be falsified if… | Status |
|---|---|---|
| Directional (spread) mechanism | isotropic variance predicted hardness as well as `sd_along` | **passed** — rog `ρ = −0.31`, wrong sign, fails FDR |
| Not a channel/loudness artifact | partial `ρ` given RMS energy collapsed | **passed** — `ρ|RMS = 0.58`, `p = 8×10⁻⁷` |
| Fusion is difficulty-targeted | gains flat across difficulty quartiles | **passed** — monotone +0.036 → +0.220 |
| Real effect, not noise-mining | `ρ` fell as utts/system rose | **passed** — attenuation signature, `ρ` rises 0.59→0.69 |
| Cross-corpus transfer | pre-registered ASVspoof-2021 test | **FAILED / inconclusive** — `ρ = +0.25`, `p = 0.41`, `n = 13` |
| Axis ≠ detector's logit direction relabeled | axis aligned with fitted probe | **passed** — `cos ≈ 0.4` only |

The failures are reported, not buried: the pre-registered cross-corpus test was null; the exploratory
ASVspoof-2021 position result (`ρ = +0.60`) does **not** survive family correction and is downgraded to
exploratory; a residual causal-compactness effect flipped sign across seeds and was dropped; the ITW
speaker count was corrected 58 → 29. A red-team pass recomputed every number from cached artifacts (not
the original analysis code), tallying 8 verified / 6 mislabeled / 5 misleading / 2 contradicted claims.

**The single most important behavioral honesty note:** the headline effect explains ≈32% of *explainable*
variance (`R² = 0.277` against a reliability ceiling of 0.863), and its most quotable applied win (the
≈21% ASVspoof fusion cut) is on a *deliberately undertrained* detector. Both are stated as such. The
naturally-weak AASIST result exists precisely to keep the applied claim from resting on the crippled
detector alone.

---

## 5. Reading guide — what behavior each artifact demonstrates

| Behavior | Primary evidence |
|---|---|
| MLAAD detector far from ceiling (headroom source) | [training_logs/mlaad_wavlm_gat/*_final_metrics.json](training_logs/mlaad_wavlm_gat/) |
| Capacity dissociation (weak vs strong ASVspoof) | `main.tex` §5, Fig. 3a |
| `sd_along → hardness` association (the core behavior) | `main.tex` §4.2, Table 1 (App. A) |
| Directional > isotropic control | `main.tex` Table 1; App. F (Q1) |
| Fusion headroom law | [i7_stats.json](experiments/results/i7_axis_fusion/i7_stats.json), [i7_headline.csv](experiments/results/i7_axis_fusion/i7_headline.csv) |
| Difficulty-targeted gain | [i7_quartiles.csv](experiments/results/i7_axis_fusion/i7_quartiles.csv) |
| Cross-domain fusion failure (axis rotation) | [itw_fusion_test.json](experiments/results/i7_axis_fusion/itw_fusion_test.json) |
| AASIST architecture-independent replication | `main.tex` §4.2, §5, App. F (Q2, Q8) |

---

### One-paragraph summary

Three detectors, one behavioral law. A frozen WavLM natural-vs-synthetic direction organizes both of the
failure modes these anti-spoofing models are known for: per-system evasion (systems that *spread* along
the axis leak across the boundary and evade — `ρ = 0.59`, the sole survivor of an 83-quantity FDR screen
on MLAAD) and cross-domain collapse (the axis *rotates* to near-orthogonality between corpora,
`cos ≈ 0.05` MLAAD↔ITW). A training-free fusion of that direction into the score is a *corrective*
signal: it repairs detectors with headroom (undertrained WavLM-GAT −21% EER; naturally-weak cross-domain
AASIST −26 to −33 EER points; in-domain MLAAD 0.266→0.157) in exact proportion to system difficulty
(+0.036 → +0.220 AUC across quartiles), does nothing to already-strong detectors, and cannot cross a
domain boundary because the direction it depends on does not transfer. The whole account runs on one GPU,
states its falsification conditions, and reports the tests that failed — which is the analysis posture COLM
is built to reward.

---

*Sources on venue framing:* [COLM 2026](https://colmweb.org/), [COLM 2025](https://colmweb.org/2025/).
