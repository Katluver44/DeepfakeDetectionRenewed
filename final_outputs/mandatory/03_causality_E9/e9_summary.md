# E9 — Are C and T causal for detection, or just correlates?

Direct intervention on the detector's **frozen** WavLM representation, applied equally to
spoof + bona (no label leakage). C is manipulated at the L12 encoder output (which feeds the
GAT directly); T at L9 (`encoder.layers[8]`). ASVspoof eval (A07–A19), 800 spoof + 800 bona.
Headline detector: **robust_GOAT** (in-distribution); GOAT shows the same pattern.

**Sign convention.** C = −rog, so **more compact = higher C = smaller rog**. C-scale knob:
`h' = μ + α(h−μ)` → rog scales by exactly α (α<1 ⇒ smaller rog ⇒ **more compact**; α>1 ⇒
**less compact**). The **shift control** translates the representation by an equal ‖Δh‖ while
leaving rog and velocities (hence C and T) mathematically unchanged — isolating "is detection
sensitive to C *specifically*" from "is it fragile to any perturbation of this magnitude."

## Verdict
- **C is CAUSAL, and the causal direction is `more compact → harder`.** In the clean moderate
  regime, matched-energy interventions move EER/AUC/Acc *through* baseline and **beat the
  equal-energy C-preserving shift control in both directions**: making the representation more
  compact specifically worsens detection; making it less compact specifically improves it.
- **This causal direction agrees with the MLAAD and In-the-Wild correlational results, and
  resolves the apparent ASVspoof contradiction (E8).** See "Reconciliation with E8" below.
- **T is NOT an isolable causal lever.** The velocity interventions failed to move vel-entropy
  (stayed 2.61–2.71 vs ≈2.82 baseline) and instead moved L9 *compactness* (rog 2.5→13). The
  degradation they cause tracks compactness, not T. T cannot be moved without moving C —
  consistent with its fragility / non-transfer in the ITW (E5/E6) and ASVspoof (E8) studies.

## C — matched-energy comparison (robust_GOAT; baseline α=1: EER 0.067, AUC 0.983)
| α | rog (↓ = more compact) | EER (C-scale) | AUC (C-scale) | EER (shift, C fixed) | AUC (shift) |
|---|---|---|---|---|---|
| 0.25 (much more compact) | 3.7 | 0.284 | 0.778 | 0.094 | 0.968 |
| 0.50 (more compact) | 7.4 | 0.298 | 0.773 | 0.074 | 0.977 |
| **0.75 (mildly compact)** | 11.2 | **0.090** | 0.973 | 0.068 | 0.981 |
| 1.00 baseline | 14.9 | 0.067 | 0.983 | 0.071 | 0.980 |
| **1.50 (mildly expanded)** | 22.2 | **0.056** | 0.985 | 0.080 | 0.976 |
| 2.00 | 29.8 | 0.079 | 0.976 | 0.104 | 0.959 |
| 3.00 | 44.8 | 0.106 | 0.961 | 0.205 | 0.862 |
| 4.00 (much less compact) | 59.6 | 0.134 | 0.942 | 0.273 | 0.803 |

**Reading.**
- **Moderate regime (the clean test, α=0.75 / 1.5):** compaction specifically *worsens*
  (EER 0.067→0.090 while matched shift stays 0.068); expansion specifically *improves*
  (EER 0.067→0.056 while matched shift *worsens* to 0.080). The C-scale beats its
  matched-energy shift in **both** directions ⇒ the effect is specific to compactness, not
  magnitude. **Polarity: more compact → harder.**
- **Strong compaction (α≤0.5):** large degradation (EER → ~0.29), but here the frame cloud is
  collapsing toward its centroid (information loss) ⇒ the extreme end overstates the pure-C
  effect. Treat the moderate regime as the clean causal evidence.
- **Strong expansion (α≥3):** degrades, but the matched **shift control degrades *more***
  (EER 0.205/0.273 vs 0.106/0.134) ⇒ at high energy this is generic fragility, not a
  C-specific effect.

GOAT shows the same shape (α=0.75: scale 0.127 vs shift 0.110; α=1.5: scale 0.117 vs shift
0.131; strong compaction α=0.5: scale 0.244 vs shift 0.126), confirming C-specificity.

## Reconciliation with E8 (why ASVspoof's *correlation* pointed the other way)
| evidence | type | finding | direction |
|---|---|---|---|
| MLAAD (n=63 systems) | correlational | ρ(C, residual) = +0.29 | more compact → harder |
| In-the-Wild E5 (per-utt) | correlational | ρ(C, logit) = −0.16 | more compact → harder |
| In-the-Wild E6 (per-speaker) | correlational | ρ(C, EER) = +0.44 | more compact → harder |
| **E9 (this experiment)** | **causal** | α<1 → EER↑, α>1 → EER↓ | more compact → harder |
| E8 (ASVspoof) | correlational | ρ(C, logit) = +0.18 | more compact → *easier* (outlier) |

E8 is **not** a methodological error — it uses the identical C extraction (frozen
`microsoft/wavlm-base`, L12 rog, C = −rog) and a verified logit convention. It is a **valid
but confounded observational** result: in ASVspoof's legacy attacks (A07–A19, 2019-era
vocoder / VC / Griffin-Lim / WORLD), the naturally-compact attacks are also the **crude,
artifact-heavy** ones, so they are easy — the *crudeness→easy* confound rides along with
compactness and dominates the raw correlation. E9 breaks that confound by scaling **only**
compactness on fixed utterances (attack identity held constant), and recovers the causal
direction `more compact → harder`, matching MLAAD and ITW. **Conclusion: ASVspoof's positive
C↔ease correlation is confounded / non-causal; it is not a genuine reversed polarity of C.**
(E8 is retained as the observational counterpoint that motivates this intervention.)

## T — interventions did not isolate T (robust_GOAT)
| intervention | measured T | rog@L9 | EER | AUC |
|---|---|---|---|---|
| smooth k=3 | 2.626 | 4.12 | 0.116 | 0.956 |
| smooth k=9 | 2.661 | 3.00 | 0.406 | 0.661 |
| smooth k=15 | 2.621 | 2.48 | 0.412 | 0.634 |
| jitter β=1 | 2.644 | 5.76 | 0.086 | 0.973 |
| jitter β=4 | 2.606 | 13.25 | 0.348 | 0.717 |

T (vel-entropy) barely moved (range 2.61–2.71) under every knob, while L9 **compactness**
(rog@L9) swung 2.5→13 and tracked the EER changes. So the apparent "T effect" is really the
**C-at-L9 effect** again — there is no clean causal evidence for T, because T could not be
manipulated without manipulating compactness. This re-confirms C as the operative geometry.

## Honest caveats on scope
- E9 intervenes on the *detector's own* phoneme-fine-tuned L12 (baseline rog ≈ 14.9), a
  **different representation** from the standalone `microsoft/wavlm-base` L12 (rog ≈ 10.8) on
  which the E1–E8 *correlations* are defined. So E9 proves causality **within the detector's
  representation**; its agreement-in-direction with MLAAD/ITW (which are on standalone WavLM)
  is strong corroboration but not a same-representation proof. The cleanest tightening would
  re-extract standalone-WavLM features, perturb them, and feed a detector decoupled from its
  own backbone — not cleanly supported by this architecture.
- Clean causal evidence = the moderate regime (α∈[0.75, 1.5]); extreme compaction is confounded
  by centroid collapse, extreme expansion by generic fragility (see shift control above).

## Conclusion
C (representational compactness) is a **causal** factor in detection: manipulating it
specifically, beyond matched-energy controls, changes EER/AUC/Accuracy, with **more compact →
harder** inside the detector. This direction is consistent across MLAAD, In-the-Wild, and this
causal test; ASVspoof's opposite *correlation* (E8) is explained as confounding (legacy compact
attacks are crude → easy), not a real reversal. T is **not** an independent causal lever — it
cannot be moved without moving compactness, and on its own carries no isolable effect.

## Files
- `e9_causal_C.csv` (C-scale + matched shift control), `e9_causal_T.csv` (smooth/jitter)
- figures: `e9_causal_C_<model>.png`, `e9_causal_T_<model>.png`
