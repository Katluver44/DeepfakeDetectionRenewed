# P1 — Test-time channel-axis projection (TAP): training-free ITW false-positive fix

## Verdict: **PARTIAL — causal claim confirmed, linear fix insufficient**

Two findings must be stated together (and honestly):
1. **The channel axis specifically controls ITW false positives** (causal/specificity claim:
   SUPPORTED). Projecting it out lowers ITW-FPR 0.664→0.531 (Δ−0.133); 12 random unit
   directions of equal norm do nothing (−0.001±0.020; empirical p=0.000). MLAAD EER preserved
   (0.267→0.270). So the reverb/MP3 direction estimated from unlabeled paired audio is
   genuinely the knob behind the genuine-audio collapse.
2. **A *linear* test-time projection is NOT a net fix** (the "free lunch" is FALSIFIED).
   Removing the axis trades false positives for false negatives (ITW-FNR 0.143→0.361), so ITW
   EER *worsens* (0.376→0.437) and balanced accuracy *drops* (0.597→0.554). No α improves
   separability — at every α, lowering FPR costs ≥ as much FNR.

**Why (and why this matters):** this is the direct empirical signature of the E12 geometry —
the channel axis is *partially collinear* with genuine synthetic evidence, so a rank-1 linear
removal cannot separate "channel" from "spoof". It confirms the mechanism AND proves a linear
operating-point shift is the wrong tool, motivating the *learned* disentanglement in P2
(causal augmentation) and P3 (paired-invariance adapter), which can bend the boundary
nonlinearly rather than slide along one axis.

TAP estimates a reverb/MP3 channel direction `v` from PAIRED clean/degraded MLAAD bona (n=400, disjoint from eval; 3 degradation seeds; axis stability mean pairwise cos=+0.999) in the detector's trainable-encoder frame space, then subtracts α·(h·v)v from every frame at inference. Detector = mlaad_robust_goat, full-model logit.

**Leakage status (audited):** the channel direction `v` is estimated WITHOUT any ITW data or
labels — that is the scientific claim and it is leak-free. The scalar strength α, however, is
selected here as an ORACLE (largest ITW-FPR drop with MLAAD-EER preserved), i.e. α *does* see
ITW labels. We report it this way deliberately: even granting an oracle α tuned directly on the
target domain, a rank-1 linear removal still cannot produce a net accuracy gain (below) — which
is the whole point. The full α-sweep is shown so no α is cherry-picked silently.

| α | MLAAD-EER | ITW-EER | ITW-FPR | ITW-FNR | ITW-acc | ITW-bal |
|---|---|---|---|---|---|---|
| 0.00 | 0.267 | 0.376 | 0.664 | 0.143 | 0.597 | 0.597 |
| 0.25 | 0.257 | 0.386 | 0.635 | 0.169 | 0.598 | 0.598 |
| 0.50 | 0.261 | 0.394 | 0.617 | 0.198 | 0.593 | 0.593 |
| 0.75 | 0.265 | 0.396 | 0.604 | 0.225 | 0.586 | 0.586 |
| 1.00 | 0.261 | 0.408 | 0.588 | 0.262 | 0.575 | 0.575 |
| 1.50 | 0.255 | 0.423 | 0.549 | 0.317 | 0.567 | 0.567 |
| 2.00 | 0.270 | 0.437 | 0.531 | 0.361 | 0.554 | 0.554 |

**Chosen α = 2.00** (largest ITW-FPR drop with MLAAD-EER preserved ≤ +0.01).

| quantity | baseline (α=0) | TAP (chosen α) | Δ |
|---|---|---|---|
| ITW false-positive rate | 0.664 | 0.531 | -0.133 |
| ITW EER | 0.376 | 0.437 | +0.061 |
| ITW balanced acc | 0.597 | 0.554 | -0.043 |
| MLAAD EER (control) | 0.267 | 0.270 | +0.003 |

## Specificity audit — random-axis null
At α=2.00, 12 random unit directions give ITW-FPR change -0.001 ± 0.020 (mean±sd); the TRUE channel axis gives -0.133. Empirical p (random ≤ true) = 0.000. ITW-EER change: true +0.061 vs null +0.000±0.006.

## Reading
- TAP removes a single estimated channel direction at test time and **specifically** lowers the
  rate at which GENUINE ITW audio is flagged synthetic (random axis does nothing, p=0.000),
  with in-domain MLAAD EER preserved — a clean causal confirmation of the E12/E13 channel axis.
- **But it is not a net win:** the same projection raises the spoof-miss rate by ~as much, so
  overall ITW EER/balanced-accuracy do not improve. The channel and spoof directions are
  entangled; a linear axis-removal can only slide the operating point, not disentangle.
- **Use as deployment knob, not accuracy fix:** at a fixed budget where false alarms on real
  audio are costlier than missed spoofs, TAP is a label-free, retraining-free way to trade FPR
  down (−0.13) along the *correct* axis. For an accuracy gain, a learned method is required (P2/P3).

## Files: p1_true_axis_sweep.csv, p1_random_null_sweep.csv, p1_axis_projection.png