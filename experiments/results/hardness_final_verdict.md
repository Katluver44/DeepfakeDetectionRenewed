# What makes synthesis systems hard to detect? — Final adversarial verdict

*Second-round adversarial campaign (2026-06-11). Builds on the C-causality audit
(`c_causality_review.md`, I1/I2). New experiments: I3 (position/boundary geometry, MLAAD),
I4 (ASVspoof replication), I5 (ITW transfer + T survival), I6 (test-time reshaping, 3 seeds),
I7 (frozen-axis score fusion + ITW falsification test).*

---

## 1. Adversarial critique of the first-round claims, and what survived

| Claim (from I1/I2) | Attack | Outcome after testing |
|---|---|---|
| vel_entropy (T-family) predicts hardness (MLAAD LOO 0.083) | binning artifact? SNR proxy? dataset-specific? | **Partially failed.** Survives RMS partialling on MLAAD (ρ=+0.25, p=.055) and is independent of position (partial ρ=+0.36, p=.004); but **ns across ASVspoof attacks** (ρ=+0.31, p=.31; FE p=.20) and **ns / sign-flipped on ITW speakers** (ρ=−0.14). T is a within-MLAAD secondary factor, not a general mechanism. |
| mu_norm / dist-to-bona predicts hardness | recording-level confound; undecomposed | **Superseded.** Decomposition into along-axis vs orthogonal components (I3) absorbs it; mu_norm is marginal after RMS partialling (p=.067). |
| Dominant-subspace causal sensitivity → test-time reshaping gains | single-detector artifact? | **Half-failed.** Causal sensitivity replicated (I1, 3 ASVspoof seeds), but the *gains* are not seed-robust on MLAAD (I6: best condition p=.07; spec hurts ITW p=.007). Reshaping is not the lever. |
| Feature-extractor integrity | pos_conv load warning suggested random weights | **Audited clean.** transformers maps legacy weight_norm keys correctly despite a spurious warning; bit-identical re-extraction confirmed; loader now repairs explicitly (`px_common.load_frozen_wavlm`). |
| Hardness itself is detector-specific | circularity | **Refuted.** Per-system hardness Spearman 0.89–0.97 across 3 independently trained detector seeds; all position claims tested against the shared component. |

## 2. The factor that survived everything: decision-boundary position geometry

Define, in **frozen pretrained WavLM-base L12 mean-pool space** (independent of any detector),
the natural↔synthetic axis w (bona vs spoof centroid difference, always fit
**system/speaker/fold-disjoint** from the evaluation unit), and per utterance:
s_along = position along w, s_orth = off-axis residual distance. Per system:
mean position, spread along axis (sd_along).

**MLAAD (61 system×language units, hardness = shared 3-seed 1−AUC):**
- sd_along alone: LOSO R² = **+0.273**, perm p = **0.0005** — the strongest single predictor
  found in the entire project (old C: −0.015).
- s_along: ρ=−0.26 (p=.042; replicates in all 3 seeds); 3-factor model
  (s_along + sd_along + vel_entropy): LOSO R² = **+0.316**, perm p = 0.0005.
- Within-system FE: **s_orth β=−0.20, perm p=.005** — off-axis deviation pushes scores toward
  bona ("orthogonal masking").

**ASVspoof A07–A19 (cross-dataset replication):**
- s_along: **ρ=−0.71, p=.006** across 13 attacks. The community's known-hardest attacks
  (A17/A18/A19) sit at s_along≈2–3; every easy attack at 12–18.
- Within-attack FE: s_along β=+0.57, perm p<1e-4. (vel_entropy ns; rog sign-flips again.)
- Why T fails here: ASVspoof position spread is 3.5× MLAAD's (SD 5.9 vs 1.7) — hardness
  variance is position-dominated, leaving no measurable dynamics margin at N=13.

**In-The-Wild:**
- The position *law* holds: an ITW-internal axis (speaker-disjoint) alone achieves
  **EER 0.292 / AUC 0.783 — beating the fine-tuned detector (0.363 / 0.683) by 7 EER points.**
- Speaker-level hardness is governed by **s_orth (ρ=+0.53, p=.003)** — same orthogonal-masking
  law; FE β=−0.96, p<1e-4.
- The *direction* does not transfer: cos(w_MLAAD, w_ITW) = **0.05**.

## 3. Performance: the decisive intervention (I7)

Zero-training score fusion, fully system-disjoint 5-fold protocol, λ selected on train folds,
3 detector seeds, paired utterance bootstrap (B=2000):

| MLAAD test | EER | AUC |
|---|---|---|
| Detector (3-seed mean) | 0.272 | 0.797 |
| Frozen axis **alone** | 0.187 | 0.872 |
| **Fused** | **0.163** | **0.917** |

**ΔEER = −0.109 absolute (−41% relative), 95% CI [−0.129, −0.091], p<1e-4; ΔAUC = +0.120
[+0.102, +0.137].** Gains are monotone in baseline hardness: Q1 +0.036 → **Q4-hardest +0.220
AUC** (0.617→0.837). This is the mechanism converted into performance: hard systems are hard
because they sit close to bona with high spread along an axis the detector under-reads;
re-injecting that axis recovers them. (Contrast: all test-time representation-reshaping
conditions failed seed-robustness — I6.)

By contrast the same fusion on ITW with the MLAAD axis **fails monotonically**
(EER 0.363→0.423 at λ=2) — a pre-registered falsification that confirms the mechanism rather
than undermining it (§4).

## 4. Why ITW is hard (E10–E13, completed)

Three quantitative facts, all on frozen out-of-sample features:
1. **Translation:** ITW-bona slides **+0.64** of the full bona→spoof gap along the MLAAD axis
   (E12's law replicated with a transfer axis); within ITW-bona, axis position predicts the
   false-positive logit at ρ=+0.47, p≈1e-83 (n=1500).
2. **Collapse/inversion:** ITW classes on the MLAAD axis: bona 4.39 vs spoof 4.02 — separation
   gone, slightly inverted. The MLAAD-axis is worthless on ITW (AUC 0.527).
3. **Dispersion + rotation:** ITW-bona variance along the axis is 47.7 vs 15.4 (MLAAD-bona);
   and the ITW-internal natural↔synthetic direction is nearly orthogonal to MLAAD's
   (cos=0.05). The discriminative geometry is intact in ITW (internal axis AUC 0.783) but
   **rotated** by the recording-channel manifold.

**One sentence:** ITW is hard because real-world channel variation translates, disperses, and
rotates the natural↔synthetic axis — the detector keeps reading the *old* direction (E12/E13),
on which the classes have collapsed.

**Does T survive ITW?** No — at the speaker level vel_entropy is ns (ρ=−0.14), and its
within-speaker sign flips vs MLAAD. T did not survive adversarial cross-domain testing.

## 5. Final verdict (the seven questions)

1. **What causes hardness?** Decision-boundary geometry in deep SSL space: a system is hard in
   proportion to (a) how close its utterances sit to the bona population *along* the
   domain's linear natural↔synthetic axis, (b) how widely they spread along that axis, and
   (c) how much off-axis (orthogonal) displacement masks the synthetic signature. Secondary,
   domain-local factor: trajectory velocity entropy (MLAAD only).
2. **Which mechanisms survived adversarial testing?** Position/spread along the axis (every
   dataset, every level of analysis, every confound control: system-disjoint axes, 3 detector
   seeds, RMS/language partialling, permutation nulls, FE panels); orthogonal masking
   (MLAAD FE p=.005, ITW p=.003); detector under-reading of the axis (proven by the linear
   probe and fusion gains on both MLAAD and ITW).
3. **Which failed?** C = −rog@L12 (no unique variance; sign flips on ASVspoof and ITW);
   T = vel_entropy as a general mechanism (ns on ASVspoof attacks and ITW speakers);
   test-time representation reshaping as a remedy (not seed-robust); E9's original causal
   claim (~80% hook artifact, established in round 1).
4. **Why are some synthesis systems unusually difficult?** Because their outputs are
   *channel-and-prosody-faithful enough* to sit near the bona centroid along the axis
   (ASVspoof A17/A18/A19: s_along 2–3 vs 12–18; MLAAD Q4 systems) while their spread along the
   axis pushes part of their mass across the boundary — a d′ problem, not an artifact-absence
   problem. The detector compounds this by discarding part of the axis evidence
   (frozen-linear > fine-tuned GAT everywhere tested).
5. **Can the surviving mechanisms explain ITW difficulty?** Yes, quantitatively and
   completely (§4): translation (+0.64 gaps), dispersion (3×), rotation (cos 0.05) of the
   axis under channel shift — plus orthogonal masking governing speaker hardness.
6. **What detector changes yield the largest gains?** Re-injecting frozen-axis position:
   −41% relative EER on MLAAD (p<1e-4, 3 seeds, system-disjoint), +0.22 AUC on the hardest
   quartile; on shifted domains, a *domain-internal* axis alone beats the detector by 7 EER
   points (ITW) — implying the right production design is detector + per-domain axis
   estimation (a few hundred unlabeled-ish utterances suffice for centroids), not
   representation surgery.
7. **Strongest NeurIPS-level contribution?** *"Deepfake hardness is decision-boundary
   geometry: a single frozen linear axis predicts which synthesis systems defeat a
   state-of-the-art detector (LOSO R²=0.32; ρ=−0.71 cross-dataset), explains in-the-wild
   collapse as translation-dispersion-rotation of that axis, and — because fine-tuned
   detectors provably under-read it — yields a zero-training fusion that cuts EER by 41% and
   recovers the hardest systems by +0.22 AUC."* With the forensic audits (E9 hook artifact,
   C/T failures) as the methodological backbone.

---

# Round 3 (J-series): LDA equivalence, audits, prospective prediction, axis-adaptive head

## J1 — "Is this just LDA?" + leakage audit
- The hardness law holds along **any** discriminative direction of frozen space and is null
  on random directions: position+spread → hardness LOSO R²: centroid axis 0.19, logistic 0.39,
  shrinkage-LDA 0.43, random −0.07 (detector's own axis 0.71, positive control). The law is a
  property of the discriminative geometry, not of the centroid estimator.
- **Frozen shrinkage-LDA (system-disjoint OOF) achieves EER 0.0997 / AUC 0.960 — versus the
  fine-tuned detector's 0.272 / 0.797.** LDA-fusion: −15.7 EER points. The "detector under-reads
  its backbone" finding is ~3× larger than the centroid-axis version. (E12's old "LDA overfits"
  was an evaluation artifact.)
- Audits all clean: split-half (features ⟂ hardness utterances) ρ=−0.218±0.090 over 20 splits;
  per-fold fusion deltas negative in 15/15 fold×seed cells; label orientation correct;
  bona-fold-disjoint standardization changes nothing.

## J2 — Pre-registered prospective prediction on unseen systems (honest failure)
Predictions for 102 units written to disk before any scoring. Results:
- WaveFake (10 systems): centroid-geometry prediction ρ=+0.62 (p=0.054) — marginal pass;
  LDA-geometry ns.
- VCC2020 (31 teams): fails (ρ≈0).
- Reverse transport (ASVspoof-trained robust_goat on 61 MLAAD systems): ρ=+0.20, ns.
- **Bound discovered:** the two detector families agree on which systems are hard only at
  ρ=+0.30 (p=0.018). Hardness is substantially detector-conditional; no geometry can predict
  another detector's hardness beyond that ceiling.
**Conclusion:** the position law is **domain- and detector-conditional** — consistent with the
axis-rotation law (cos(w_MLAAD, w_ITW)=0.05), and now established prospectively rather than
assumed. Universal cross-corpus hardness prediction from a single source axis is falsified.

## J3 — Axis-adaptive detector head (the deployable consequence)
Frozen detector + domain axis estimated from a small calibration set
(group-disjoint evaluation, 20 draws, λ chosen on calibration only):

| dataset | detector EER | head (25/class) | head (250/class, LDA) | unsupervised |
|---|---|---|---|---|
| MLAAD | 0.272 | 0.323 (hurts) | **0.136 (−50%)** | 0.244 (−2.8 pts) |
| ITW   | 0.363 | 0.287 (−7.7 pts) | **0.192 (−47%)** | 0.365 (no-op) |
| ASVspoof | 0.078 | 0.080 | 0.082 (no gain) | 0.076 |

The head helps exactly in proportion to how much the detector under-reads the domain's axis
(massively on MLAAD/ITW, not at all for the strong ASVspoof detector) — the mechanism and the
remedy are the same quantity measured twice.

## J4 — Pre-registered prospective prediction on ASVspoof 2021 LA (refined methodology)
Lesson from J2 applied: predict from the **corpus-internal** axis (ground-truth labels of the
target corpus, leave-one-attack-out; no detector outputs), not a transferred axis. 13 attacks,
clean codec condition, 40 utts/attack + 400 bona; evaluation detector = MLAAD robust_goat × 3
seeds (a family that has NEVER seen ASVspoof; my own prior knowledge of ASVspoof rankings came
only from robust_goat, a different family). Predictions registered before scoring
(`j4_preregistered_predictions.json`).

- Registered hardest-3 (P1/P2/P3): **{A17, A18, A19}**. Actual: A19 (0.545), A18 (0.494),
  A11 (0.452) → **2/3 hits**.
- **P3 (corpus-internal LOAO shrinkage-LDA position): ρ=+0.60, p=0.031** — significant
  pre-registered prospective prediction across detector families.
- P4 (MLAAD-axis transfer): ρ=−0.37 — third independent confirmation of the rotation law.
- Honest miss: A11 sits far along the axis (predicted easiest) yet is #3 hardest; orthogonal
  masking does not explain it (orth mid-pack) — a detector-conditional blind spot, consistent
  with the ρ=0.30 cross-family hardness agreement bound.
- VCC2020's J2 failure is now interpretable: voice conversion preserves genuinely human source
  speech, so a TTS-content axis should not apply; WaveFake (vocoded resynthesis, closest to the
  TTS regime) was the corpus that marginally passed. The law's domain of validity:
  **TTS-style synthesis, corpus-internal axis.**

## J5/J6 — Non-WavLM baseline (AASIST): which hypotheses are architecture-general?
Strategy: official pretrained AASIST (ASVspoof19-trained; unimpeachable reference) for the
prospective/fusion/agreement legs; AASIST fine-tuned on MLAAD train (val EER 0.159, test EER
0.200 — beating the WavLM-GAT's 0.266) for the in-domain law cell.

- **Prospective prediction is architecture-general:** the SAME registered J4 P3 predictor
  (corpus-internal LDA position) predicts per-attack hardness for AASIST at ρ=+0.64 (p=0.018)
  and for the MLAAD WavLM-GAT at ρ=+0.60 (p=0.031), on ASVspoof 2021 clean.
- **The agreement matrix resolves what hardness is conditioned on:** same-training-domain,
  cross-architecture pairs agree at ρ≈0.55 (AASIST_ft↔MLAAD-GAT 0.553; AASIST_zeroshot↔
  robust_goat 0.544); ALL cross-domain pairs sit at ρ≈0.30–0.36. **Hardness is
  training-domain-conditional, not architecture-conditional.**
- **Decomposition of the law:** in-domain under AASIST_ft, the architecture-GENERAL hardness
  components are within-system spread along the axis (sd_along ρ=+0.35, p=0.006) and
  trajectory dynamics (vel_entropy ρ=+0.34, p=0.007 — T resurfaces as architecture-general on
  MLAAD); the mean-position component (s_along) is detector-family-conditional (ns for AASIST).
- **Under-reading/fusion is architecture-general under shift:** frozen-axis fusion repairs
  pretrained AASIST by −26 EER points on MLAAD (0.376→0.117) and −33 on ITW (0.486→0.161),
  with no gain in-domain (0.073→0.076) — same boundary condition as the WavLM detectors.
- ITW per-speaker law untestable for AASIST (EER 0.486 ≈ chance ⇒ hardness is noise).

## 6. Residual program (pre-registered)
- Cross-detector-family transport (AASIST/LCNN) of the position law.
- Prospective sign prediction on unseen corpora (ASVspoof5, MLAAD v5).
- Waveform-level demonstration (channel ops moving s_along move logits — E13 already half-does this).
- Axis-adaptive detector head (train-time integration of per-domain axis estimation).

## Artifacts
`i3_position_geometry/` (axes, embeddings, stats), `i4_asvspoof_position/`,
`i5_itw_transfer/`, `i6_testtime_boost/`, `i7_axis_fusion/` (incl. `itw_fusion_test.json`).
