# E9 — Are C and T causal for detection, or just correlates?

Direct intervention on the detector's **frozen** WavLM representation, applied equally to
spoof + bona (no label leakage). C is manipulated at the L12 encoder output (which feeds the
GAT directly); T at L9 (`encoder.layers[8]`). ASVspoof eval (A07–A19), 800 spoof + 800 bona.
Headline detector: **robust_GOAT** (in-distribution); GOAT shows the same pattern.

**Knobs.** C-scale: `h' = μ + α(h−μ)` → rog scales by exactly α (α<1 ⇒ smaller rog ⇒ **more
compact ⇒ higher C**; α>1 ⇒ **less compact ⇒ lower C**, since C = −rog). The **shift control**
translates the representation by an equal ‖Δh‖ while leaving rog and velocities (hence C and
T) mathematically unchanged — isolating "is detection sensitive to C *specifically*" from
"is it fragile to any perturbation of this magnitude."

> Note on comparability: E9 intervenes on the *detector's own* phoneme-fine-tuned L12
> (baseline rog ≈ 14.9), which is a **different representation** from the standalone
> `microsoft/wavlm-base` L12 (rog ≈ 10.8) used for the E1–E8 *correlations*. So E9 establishes
> causality **within the detector's representation**; its direction is not directly
> comparable to E8's per-corpus correlational polarity.

## Verdict
- **C is CAUSAL.** In the clean moderate regime, matched-energy interventions move EER/AUC/Acc
  *through* baseline and **beat the equal-energy C-preserving shift control in both
  directions**: making the representation more compact worsens detection, making it less
  compact (mildly) improves it. Direction (within the detector): **more compact → harder.**
- **T is NOT cleanly testable here / shows no isolable causal effect.** The velocity
  interventions failed to move vel-entropy (stayed 2.60–2.71 vs 2.82 baseline) and instead
  moved L9 *compactness* (rog 2.5→13). So the degradation they cause is attributable to
  compactness, not T. T is not independently manipulable — consistent with its fragility /
  non-transfer in the ITW and ASVspoof correlation studies.

## C — matched-energy comparison (robust_GOAT; baseline α=1: EER 0.067, AUC 0.983)
| α | rog (↓=more compact) | EER (C-scale) | AUC (C-scale) | EER (shift, C fixed) | AUC (shift) |
|---|---|---|---|---|---|
| 0.25 (much more compact) | 3.7 | 0.284 | 0.778 | 0.094 | 0.968 |
| 0.50 (more compact) | 7.4 | **0.298** | **0.773** | 0.074 | 0.977 |
| **0.75 (mildly compact)** | 11.2 | **0.090** | 0.973 | 0.068 | 0.981 |
| 1.00 baseline | 14.9 | 0.067 | 0.983 | 0.071 | 0.980 |
| **1.50 (mildly expanded)** | 22.2 | **0.056** | 0.985 | 0.080 | 0.976 |
| 2.00 | 29.8 | 0.079 | 0.976 | 0.104 | 0.959 |
| 3.00 | 44.8 | 0.106 | 0.961 | 0.205 | 0.862 |
| 4.00 (much less compact) | 59.6 | 0.134 | 0.942 | 0.273 | 0.803 |

**Reading.**
- **Moderate regime (the clean test, α=0.75 / 1.5):** compaction specifically *worsens* (EER
  0.067→0.090 while matched shift stays 0.068); expansion specifically *improves* (EER
  0.067→0.056 while matched shift *worsens* to 0.080). The C-scale beats its matched-energy
  shift in **both** directions ⇒ the effect is specific to compactness, not magnitude.
- **Strong compaction (α≤0.5):** large degradation (EER → ~0.29). Partly genuine C-effect,
  partly the frame-cloud collapsing toward its centroid (information loss) — so the extreme
  end overstates the pure-C effect; treat the moderate regime as the clean causal evidence.
- **Strong expansion (α≥3):** degrades, but here the matched **shift control degrades *more***
  (EER 0.205/0.273 vs 0.106/0.134) ⇒ at high energy the loss is generic fragility, not a
  C-specific effect. So lowering C is *not* specifically harmful.

GOAT shows the same shape (α=0.75: scale 0.127 vs shift 0.110; α=1.5: scale 0.117 vs shift
0.131; strong compaction α=0.5: scale 0.244 vs shift 0.126), confirming C-specificity.

## T — interventions did not isolate T (robust_GOAT)
| intervention | measured T | rog@L9 | EER | AUC |
|---|---|---|---|---|
| smooth k=3 | 2.626 | 4.12 | 0.116 | 0.956 |
| smooth k=9 | 2.661 | 3.00 | 0.406 | 0.661 |
| smooth k=15 | 2.621 | 2.48 | 0.412 | 0.634 |
| jitter β=1 | 2.644 | 5.76 | 0.086 | 0.973 |
| jitter β=4 | 2.606 | 13.25 | 0.348 | 0.717 |

T (vel-entropy) barely moved (range 2.60–2.71) under every knob, while L9 **compactness**
(rog@L9) swung 2.5→13 and tracked the EER changes. So the apparent "T effect" is really the
**C-at-L9 effect** again — there is no clean causal evidence for T, because T could not be
manipulated without manipulating compactness. This re-confirms C as the operative geometry.

## Conclusion
C (representational compactness) is a **causal** factor in detection — manipulating it
specifically, beyond matched-energy controls, changes EER/AUC/Accuracy, with *more compact →
harder* inside the detector's own representation. T is **not** an independent causal lever:
it cannot be moved without moving compactness, and on its own it carries no isolable effect.
Across E5–E9 the consistent story is that **C is the robust geometric difficulty axis (its
hardness *polarity* is corpus-dependent, E8) while T is fragile and entangled with C.**

## Files
- `e9_causal_C.csv` (C-scale + matched shift control), `e9_causal_T.csv` (smooth/jitter)
- figures: `e9_causal_C_<model>.png`, `e9_causal_T_<model>.png`
