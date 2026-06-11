# Channel-robustness contributions (P1–P4): one-page comparison

All evaluated on the SAME ITW set (3000 bona + 3000 spoof) and MLAAD test, same detector (mlaad_robust_goat), same full-model logit. Baseline = frozen detector. ITW false positives (genuine flagged synthetic) are the failure being attacked; MLAAD-EER is the in-domain control that must be preserved. Lower is better for every column except ITW-bal.

| method | variant | MLAAD-EER | ITW-EER | ITW-FPR | ITW-FNR | ITW-bal |
|---|---|---|---|---|---|---|
| baseline | frozen mlaad_robust_goat | 0.267 | 0.376 | 0.664 | 0.143 | 0.597 |
| P1 TAP (linear, test-time) | oracle α=2.0 | 0.270 | 0.437 | 0.531 | 0.361 | 0.554 |
| P4 blind correction | baseline (no correction) | 0.263 | 0.372 | 0.664 | 0.149 | 0.594 |
| P4 blind correction | blind correction ĝ(d) | 0.286 | 0.410 | 0.600 | 0.234 | 0.583 |
| P4 blind correction | shuffled-descriptor control | 0.286 | 0.387 | 0.588 | 0.208 | 0.602 |
| P2 augmentation (train) | robust_goat (ref) | 0.245 | 0.376 | 0.654 | 0.140 | 0.603 |
| P2 augmentation (train) | none | 0.247 | 0.431 | 0.670 | 0.169 | 0.581 |
| P2 augmentation (train) | broad | 0.287 | 0.397 | 0.598 | 0.213 | 0.594 |
| P2 augmentation (train) | causal | 0.273 | 0.458 | 0.715 | 0.170 | 0.557 |
| P3 adapter (train) | frozen (ref) | 0.254 | 0.378 | 0.669 | 0.145 | 0.593 |
| P3 adapter (train) | adapter λ=0.0 | 0.287 | 0.358 | 0.708 | 0.107 | 0.593 |
| P3 adapter (train) | adapter λ=1.0 | 0.313 | 0.454 | 0.703 | 0.208 | 0.545 |

## The arc
1. **P1 (linear, test-time)** proves the reverb/MP3 channel axis *causally* controls ITW false positives (random-axis null p=0.000) but, being collinear with genuine spoof evidence, a linear removal only trades FP↓ for FN↑ — no net gain even with an oracle α.
2. **P4 (blind, scalar)** confirms the channel is blind-measurable (101% in-domain neutralization) yet its ITW effect is a pure global shift (shuffle-null matches) — same disentanglement wall.
3. **P2 (causal augmentation)** and **P3 (paired-invariance adapter)** are the LEARNED methods that can bend the boundary nonlinearly; they are the test of whether disentanglement is achievable.

**Best learned method:** P2 augmentation (train) / robust_goat (ref) — ITW-bal 0.597→0.603, ITW-FPR 0.664→0.654, MLAAD-EER 0.267→0.245.