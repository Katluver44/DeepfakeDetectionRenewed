# Is C = −rog@L12 causal? An adversarial causal-inference review and discovery program

*Role: adversarial NeurIPS reviewer / causal representation learning / speech SSL.*
*Scope: Parts 1–4 of the C-causality audit. Part 5 (executed interventions I1, I2) is reported
in `i1_geometry_causal_decomp/` and `i2_geometry_battery/` and synthesized at the end.*

---

## 0. The evidence as it stands, read adversarially

What the repo has actually established (with file-level provenance):

| Claim | Evidence | Adversarial reading |
|---|---|---|
| C predicts residual MLAAD hardness | r=−0.329 (p=.009), LOO R²=+0.063, **N=63 systems × 2 utts** (`outputs/final_report.md`) | 2 utts/system is close to anecdote; feature measured on the *same* utterances the hardness is computed from → finite-sample coupling |
| C survives PCA/shuffle/noise | E2/E3 | These rule out *measurement* artifacts of rog itself, not confounding. A proxy of a deeper quantity survives all three too |
| C in WavLM & wav2vec2, absent in HuBERT | E1 | Cuts both ways: a *physical* property of the audio should appear in any good SSL encoder. Encoder-dependence is evidence C is a property of (signal × representation), i.e. possibly epiphenomenal to the training objective |
| Injection improves detector (P1/P3/P5) | ΔEER −0.005 to −0.023 | Predictive usefulness ≠ causality. Any correlated summary improves a detector when injected |
| Shapley unique variance | orthogonalization report | Unique variance is relative to the *measured* covariate set; an unmeasured common cause is untouched |
| E9 intervention: compaction → harder, 3 seeds, p<.001 | `e9_causal_seeds` | The strongest item — but see §1.1: the isotropic intervention does not isolate rog, and the hook leaked into the phoneme-ID path |
| ASVspoof A01–A06: sign flips (r=−0.38 → reversed) | E4 | Dismissed as "homogeneous", but a real mechanism shouldn't flip sign; "insufficient power" explains a null, not a reversal |
| ITW (E10–E13): hardness is a *channel-direction* effect | E12/E13 | A second, *separate* hardness mechanism exists and is directional, not geometric-dispersion. Any C-causality claim must state its domain of validity (clean synthesis-system hardness, not channel-shift hardness) |

**Prior verdict before new work:** C is somewhere between (E) partial mechanism and (B) proxy.
It is almost certainly **not** a pure measurement artifact (C) — the E2/E3 invariances and the
cross-encoder replication preclude that. It is not obviously causal (A), because the one true
intervention (E9) is confounded in three specific ways (§1.1).

---

## Part 1 — Causal analysis: C → hardness vs. hidden factor → {C, hardness}

### 1.0 The causal estimand, stated precisely

Unit = synthesis system *s* (not utterance). Treatment = the within-utterance dispersion of
frames in deep SSL space, C(s). Outcome = hardness H(s) = 1 − AUC(s) under a fixed detector.
The DAG against which everything must be tested:

```
                 G (generator family: vocoder class, AR vs flow, training data)
               /   |   \
              v    v    v
   smoothness S    C    A (other acoustic signatures: spectral tilt, phase coherence…)
              \    |    /
               v   v   v
                H (hardness)
```

The threat is that G (or S, or A) causes both C and H, and C carries no arrow into H at all.
Note C cannot be intervened on at the *waveform* level without changing other descendants of G
— this is the fundamental identifiability problem. Representation-level interventions sidestep
it but change the question to "does the detector causally *use* the dispersion coordinate?",
which is the only version of (A) that is actually testable. Be explicit about this in any paper:
**"C is causal" can only mean "the detector's decision function is causally sensitive to the
dispersion coordinate of its input representation, and systems differ on that coordinate."**
That two-part claim (mechanism inside the model + naturally occurring variation outside it) is
the publishable form; "compact audio is intrinsically harder" is not identifiable and should
never be written.

### 1.1 Why E9 does not yet license claim (A) — three confounds

1. **The isotropy confound.** h′ = μ + α(h−μ) multiplies rog, *every frame velocity*, and every
   covariance eigenvalue by α simultaneously. T (temporal smoothness) is the project's other
   established hardness factor — and this intervention moves it in lockstep. The shift control
   matches energy but holds *both* C and T fixed, so ΔAUC(scale−shift) is "C-or-T-or-spectrum
   specific", not C-specific. E9's headline is literally consistent with "velocity magnitude is
   the causal coordinate and rog is its epiphenomenon."
2. **The phoneme-path leak.** `modules.py:417` sets `self.encoder = transformer_in_phoneme_model.encoder`
   (same object), and E9's hook is ungated, so the intervention also corrupted the phoneme IDs
   that drive pooling/graph construction. Part of the measured harm may be "wrong graph", not
   "compact representation".
3. **The asymmetry problem.** Only the compaction arm is significant and seed-robust; expansion
   is not. A causal dose–response should be locally monotone through the operating point. The
   honest reading: compaction *destroys information* (any contraction toward the mean does),
   and information destruction harms any classifier. The shift control does not control for
   information loss — a translation is invertible, a contraction toward μ is (numerically)
   contractive. So E9's compaction harm has a rival explanation: **generic contraction harm**,
   which would occur for any direction-uniform shrinkage regardless of rog's semantic role.

These three are exactly what intervention I1 (executed, Part 5) is built to break: gated hooks
(kills #2), velocity/spectrum-matched arms at *fixed* rog and rog-changing arms at fixed
velocity/shape (kills #1), and subspace-resolved compaction + shuffle controls (addresses #3 —
if "any contraction harms", contracting the residual subspace should harm as much as the
dominant one at matched rog).

### 1.2 The analysis battery, ranked by evidential strength

**(i) Geometry-resolved representation intervention (I1 — executed).**
*Assumptions:* the hook site is the full mediator of upstream information (true by architecture:
everything reaching the GAT passes through this tensor); transformed representations stay in the
region where the downstream network behaves smoothly (checked via the moderate-regime restriction
and matched-energy controls).
*Expected outcomes:* table of (Δrog, Δvel, Δeff-rank) → ΔAUC across arms. C is causal iff AUC
tracks the rog column and is flat along matched-rog arms.
*Failure cases:* off-manifold artifacts — a spectral reshape may create representations no real
utterance produces; mitigate by moderate γ and reporting the dose–response, not one point. Also
interactions: if harm requires *joint* movement of rog and velocity, all single-coordinate arms
look flat and C is "partial mechanism" — that is itself the finding.
*Strength:* **high** — this is the only design that can separate (A) from (B)/(D) at the
representation level. It is the experiment a NeurIPS reviewer would ask for.

**(ii) Waveform-level instrumental manipulation (the strongest *external-validity* design, not yet run).**
Find audio operations that move C@L12 *differentially* while controlled for low-level smoothness:
e.g., re-synthesis through a neutral vocoder (Griffin-Lim / HiFi-GAN copy-synthesis) applied to
*bonafide* audio, prosody flattening (pitch/energy monotonization), and speaking-rate warping.
Measure (ΔC, ΔT, Δlogit) per operation; fit the mediation model Δlogit ~ ΔC + ΔT + op dummies.
*Assumptions:* operations form a valid "instrument set" — they affect H only through measurable
representation changes (exclusion restriction is *not* fully testable; partially checkable by
showing op dummies carry no residual effect).
*Expected:* if C is causal, waveform ops that compact L12 should raise spoof-logit on bonafide
audio in proportion to ΔC, across heterogeneous ops.
*Failure:* every audio op moves many representation coordinates; with ~10 ops you cannot
deconfound 14 geometry features. This design *complements* (i); alone it is weak.
*Strength:* medium-high as a robustness companion; high audience value ("you can make real
speech look fake by compacting it").

**(iii) Natural experiment: system families as instruments.**
MLAAD contains the *same* TTS architecture with different vocoders, and the same vocoder under
different acoustic models (e.g., XTTS/VITS/Tacotron2 families across languages). Within-family
contrasts hold G partially fixed: regress ΔH on ΔC *within architecture family*. If C→H, the
within-family slope matches the across-family slope; if a family-level confounder drives both,
the within-family slope collapses toward 0.
*Assumptions:* family labels capture the confounder's level; enough within-family variation in C.
*Failure:* families are few; within-family C spread may be small (the ASVspoof problem again).
*Strength:* medium; cheap; the right response to "G causes both".

**(iv) Cross-detector transportability.**
A causal-for-the-task property should make systems hard for *any* detector trained on the same
distribution; a detector-specific epiphenomenon will not transfer. Compute per-system H under:
(a) the GAT detector seeds, (b) a linear probe on frozen WavLM, (c) an AASIST/LCNN-style baseline
(no WavLM at all). Rank-correlate per-system hardness and test whether C predicts the *shared*
hardness component (first PC of the H-matrix) vs detector-specific residuals.
*Assumptions:* detectors differ enough architecturally.
*Expected:* C predicting shared hardness ⇒ property of the data-representation interface (supports A/E);
C predicting only WavLM-detector hardness ⇒ representation-specific (D-leaning, still publishable but different paper).
*Failure:* all detectors implicitly use SSL-like features → shared component is itself
representation-bound. Note the HuBERT null (E1) already *hints* at detector-relativity.
*Strength:* medium-high; this is the cleanest test of (D) epiphenomenon.

**(v) Within-system, within-utterance panel design.**
Move below the system level: for each system, does the *utterance-level* C predict the
utterance's logit margin, with system fixed effects? Fixed effects absorb every system-level
confounder (G, training data, vocoder). The prior analyses never ran this with adequate N.
I2 (executed) provides the data; the FE regression is the single cheapest high-value test in
the whole program.
*Failure:* range restriction within system; measurement noise dominates single utterances.
*Strength:* medium per se, **high in combination** with (iii) — confounders must then operate
both within and across systems to survive.

### 1.3 What pattern of results maps to which verdict

| Result pattern | Verdict |
|---|---|
| I1: AUC tracks rog arms, flat on matched-rog arms; (v) FE slope ≠ 0; (iv) predicts shared hardness | **(A) causal** (as the two-part claim of §1.0) |
| I1: matched-rog velocity/spectral arms reproduce the harm | **(B) proxy** for smoothness / spectral shape |
| I1: only joint movements harm; battery (I2) shows rog subsumed by a multi-feature factor | **(E) partial mechanism / low-dim summary** |
| (iv): no transfer beyond WavLM-family detectors; HuBERT null persists | **(D) epiphenomenon** of the SSL objective (masked-prediction encoders allocate dispersion to predictability; detection rides the same axis) |
| I2 full-power battery: rog signal evaporates at 15 utts/system | **(C) measurement artifact** of the 2-utt sampling (unlikely given E2/E3, but I2 is the direct test) |

---

## Part 2 — Geometry discovery: what actually governs synthesis-system hardness in WavLM space?

Treat C as one coordinate of the within-utterance Gram structure. The full object is the
centered frame matrix S ∈ R^{T×768}; everything below is a functional of S (and its time order):

- **Size:** rog = √(tr Σ / 1) — what C measures.
- **Shape:** eigen-spectrum of Σ — effective rank, participation ratio, top-1 fraction.
- **Orientation:** where the utterance's principal axes sit relative to the detector's
  decision-relevant subspace (E12's w_mean axis) — *none of the current features measure this.*
- **Dynamics:** velocity stats (T lives here), curvature, tortuosity, recurrence, attractor dim (TwoNN).
- **Position:** centroid norm, distance to bona centroid — decision-boundary geometry, not shape.

Hypotheses for what C may be standing in for:

1. **H-spectrum:** hard systems don't have *small* clouds, they have *low-rank* clouds (oversmoothed
   TTS collapses fine phonetic variation onto fewer directions); rog is correlated with total
   variance but the operative quantity is eff-rank/top1-frac. → I2 tests: does eff_rank subsume
   rog in LOSO prediction? I1 tests: does spec(γ) at fixed rog move AUC?
2. **H-dynamics:** the operative quantity is velocity magnitude (over-smooth transitions); rog is
   its time-integral shadow. → I1 smooth_rescale arm; I2 partial-Spearman rog | vel_mean.
3. **H-position:** hard systems sit *close to the bona manifold along the detector's axis*
   (decision-boundary distance), and compact clouds are simply less likely to cross the boundary
   — i.e., C matters only multiplied by margin. → I2's dist_bona_centroid feature; interaction
   margin × rog; this is also E12's directional lesson imported from ITW to MLAAD.
4. **H-occupancy:** hard systems under-occupy the phonetic manifold (fewer effective phonetic
   states); measurable as recurrence rate / TwoNN ID. rog is a crude occupancy proxy.

Scale matters: all of these exist at (a) frame level, (b) phoneme-segment level (pool first,
then geometry — closer to what the GAT consumes), (c) utterance level. The current C is (a);
the detector operates at (b). A discrepancy between frame-level and phoneme-pooled rog would
itself explain residual variance.

Priority analyses (NeurIPS-survivable): the executed I2 battery (14 features × {L9, L12} ×
63 system-language units, ~15 utts each, LOSO + permutation + unique-variance), then the
orientation/margin features (3), then phoneme-pooled geometry (b).

---

## Part 3 — Minimum convincing experiment set for a skeptical reviewer

1. **I1** (geometry-resolved intervention with gated hooks, matched controls, 3 seeds, paired
   bootstrap) — establishes *which coordinate the detector causally uses*. (Executed.)
2. **I2** (full-power battery + LOSO + unique variance + smoothness/language residualization)
   — establishes that the coordinate *naturally varies across systems* and is not subsumed.
   (Executed.)
3. **Within-system fixed-effects panel** (from I2's per-utterance table) — kills system-level
   confounders. (Executed as part of I2 synthesis.)
4. **Cross-detector transportability** (1 GPU-day): per-system hardness under ≥3 detector
   families; C must predict the shared component. This is the remaining gap for (D).
5. **Waveform-level demonstration** (copy-synthesis compaction of bonafide audio raises spoof
   logit in proportion to ΔC) — external validity, figure-1 material.
6. **Pre-registered sign prediction on a held-out corpus** (e.g., MLAAD v5 new systems or
   ASVspoof5): predict hardness ranking from C before scoring. One clean prospective
   replication outweighs any amount of retrospective bootstrap.

Items 1–3 are done as of this audit; 4–6 are the residual program. If 1–3 hold and 4 holds,
the claim "deep-representation dispersion is a causal interface coordinate of detection
hardness" survives review. E10–E13 are not ambiguity to be resolved but a *boundary condition*:
ITW hardness is a different (directional channel-shift) mechanism, and the paper should say so.

---

## Part 4 — Discovery mode: candidate mechanisms for the unexplained ~65%

(Ranked; signature → measurement → validation → novelty.)

1. **Decision-margin geometry (classifier-level).** Hardness = f(distance of system centroid to
   the detector boundary along its logit axis, × within-system spread along that axis).
   Signature: per-system mean logit + *logit variance* predict H almost tautologically; the
   science is whether *pre-detector* geometry (centroid position along w_mean ⊕ spread) predicts
   the margin. Measure: project utterances on w_mean (E12 axis) at L12; H ~ margin + spread×margin.
   Validate: replicate across detector seeds; intervention = move along w_mean (axis projection
   already implemented in px_common.AxisProjector). Novelty: medium-high; unifies E12 (ITW) with
   MLAAD hardness under one directional law. *This is the single most promising direction.*
2. **Orientation overlap (representation-level).** Hardness depends on cos²-overlap between the
   utterance's top-k within-utterance PCs and the detector-discriminative subspace: a compact
   cloud hurts only if the *discriminative* directions are the compacted ones. Signature:
   interaction term overlap × rog dominates either alone. Measure from I2's stored embeddings +
   detector logit axis. Novelty: high — "subspace alignment" theories exist for transfer
   learning, not for deepfake hardness.
3. **Phoneme-pooled geometry mismatch (graph-level).** The GAT sees phoneme-pooled nodes;
   per-phoneme cluster collapse (low between-phoneme/within-phoneme variance ratio) should
   matter more than raw frame rog. Signature: Fisher-style separability of phoneme clusters
   predicts H beyond rog. Measure: pool frames by the model's own phoneme IDs, compute
   between/within ratio. Validate: targeted intervention collapsing only between-phoneme
   structure at fixed rog. Novelty: high, architecture-specific.
4. **SSL predictability (SSL-level).** WavLM dispersion at L12 tracks masked-prediction
   *uncertainty*; synthetic speech is more predictable → lower dispersion → less informative
   features. Signature: per-utterance WavLM masked-prediction loss correlates with C and with H;
   HuBERT's discrete targets break the relation (explaining the E1 null!). Measure: run the
   pretrained WavLM pretext loss per utterance. Validate: encoder-family comparison (wav2vec2
   contrastive vs HuBERT cluster-prediction vs data2vec regression). Novelty: very high — gives
   the *why* behind C and the HuBERT failure; it converts encoder-dependence from a weakness
   into the mechanism.
5. **Calibration/threshold mismatch (decision-level).** Part of per-system "hardness" is just
   score-distribution shift relative to the global threshold (E11/E13 logic in-domain).
   Signature: per-system EER vs per-system AUC disagree; H computed threshold-free (AUC) vs
   threshold-bound decompose differently. Low novelty, must be controlled (I2 uses AUC-based H
   for this reason).

---

## Part 5 — Executed interventions and verdict

Two interventions were selected as highest-value and executed (2026-06-11):

**I1 — Geometry-resolved causal decomposition** (`i1_geometry_causal_decomp/`):
ASVspoof A07–A19, 800+800, robust_GOAT × 3 seeds, 22 conditions/seed, paired utterance
bootstrap B=2000. All arms gated to the detection path; one ungated arm audits E9.

**I2 — Full-power geometry battery** (`i2_geometry_battery/`):
MLAAD test split, 61 system×language units (≥8 spoof utts each; ~15× the per-system data of
the prior analysis), hardness = 1−AUC under the real detector, 30 features at L0/L9/L12,
LOSO-CV + permutation nulls + unique variance + within-system fixed effects (`i2b`).

### Headline results

1. **E9's "C is causal" effect was ~75–80% an experimental artifact.** The E9 hook was
   ungated and `modules.py:417` aliases the trainable encoder to the frozen phoneme-ID
   encoder, so E9's intervention also corrupted phoneme pooling. Gated vs ungated at the same
   dose (iso 0.7): ΔAUC = +0.0101*** [3/3 seeds]. The gated compaction-specific effect
   (iso_0.7 − shift_0.7) is only **−0.0036** (p=0.006) — real, but an order of magnitude
   smaller than E9's published regime effects (e.g. −0.185 at α=0.5).
2. **rog is not the causal coordinate.** At *identical* rog reduction (0.70×):
   compacting the dominant-PC subspace → ΔAUC −0.0320***; compacting the residual subspace →
   **+0.0039\*\*\*** (improvement). Contrast sub_top − sub_res = −0.0359*** [3/3 seeds].
   The same ΔC spans harm to benefit depending on *which directions* carry it.
3. **Shape and order dominate size at fixed C.** With rog exactly fixed: spectral whitening
   −0.0059***, spectral sharpening +0.0051***, temporal smoothing +0.0035***, frame shuffling
   −0.0170***. Every one of these exceeds or rivals the C-specific effect while C is constant.
4. **Observationally, C has no unique content at full power.** rog_L12 replicates in sign
   (ρ=−0.31, p=0.016; language-residualized ρ=−0.31) but LOSO R² = −0.015 (perm p=0.09),
   unique variance over the battery = +0.003, and adding rog to vel_entropy *hurts* prediction
   (0.083→0.070). Partial Spearman: vel_entropy|rog = +0.33 (p=0.009); rog|vel_entropy = −0.20
   (p=0.12). Within-system fixed effects: rog alone β=+0.16 (perm p=0.016) but collapses to
   +0.01 jointly with velocity (which keeps β=+0.21).
5. **What actually predicts hardness at scale:** velocity *entropy* at L9/L12 (LOO 0.083),
   centroid position mu_norm_L12 (0.035), input-level rog_L0 (0.053); best pair
   vel_entropy_L12 + mu_norm_L12 = **0.155** — nearly double any previously reported value.
   Within-system, distance-to-bona-centroid is the strongest predictor (β=−0.25, p=0.001),
   importing E12's directional law into the clean domain.

### Verdict on C = −rog@L12

**(B)/(E): a proxy with a small genuine causal residue — not the mechanism.**

- Not (C) measurement artifact: sign-replicates at 15×$ data, survives language residualization,
  within-system FE significant alone.
- Not (A) causal: the operative causal coordinates are dominant-subspace content, spectrum
  shape, and temporal order; equal-C interventions produce opposite effects.
- Not purely (D): a small gated, energy-matched compaction-specific effect (−0.0036**) and a
  within-system marginal association survive every control — but both are subsumed by
  dynamics/direction variables.
- C is best described as a **low-dimensional shadow of two deeper quantities**: (i) the
  detector's causal sensitivity to its dominant-PC subspace and frame ordering (I1), and
  (ii) trajectory velocity-entropy + manifold position, which carry the real cross-system
  hardness signal (I2). The HuBERT null and the ASVspoof sign-flip stop being anomalies under
  this reading: a shadow flips and vanishes where its sources decouple.

### The publishable story this licenses

"Synthesis-system hardness lives in the *dynamics and orientation* of deep SSL trajectories,
not their size: (1) a forensic audit showing a published causal claim (compactness) was an
intervention-design artifact; (2) subspace-resolved interventions identifying dominant-subspace
content and temporal order as the detector's true causal interface; (3) a full-power
observational battery where velocity entropy × manifold position doubles hardness-prediction
R² and subsumes compactness." Items 4–6 of Part 3 (cross-detector transport, waveform-level
demonstration, prospective replication) remain the residual program.
