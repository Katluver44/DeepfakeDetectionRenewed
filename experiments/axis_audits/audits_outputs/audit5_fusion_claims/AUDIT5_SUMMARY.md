# Audit 5 — Fusion Claims: Leakage, Mislabeling, and the "Zero-Training" Framing

**Claims under test.**
1. "Axis fusion reduces EER 0.272→0.163 on MLAAD" (I7).
2. "ITW-internal axis fusion: EER 0.363 → 0.292 (WavLM-GAT)" (§3.3).
3. "A zero-training-cost fusion ... reduces EER by up to 33 percentage points under domain shift" (AASIST-ZS on ITW 0.486→0.161; on MLAAD 0.376→0.116) (abstract, §4).

**Method.** Recomputed I7 EERs and per-fold ΔEER from `i7_scores.npz` with the exact fold RNG; recomputed the ITW speaker-disjoint internal axis and fusion from scratch (`i5` features + logits); decomposed J5-H4 by mirroring its exact protocol and additionally scoring the **LDA axis alone** (the component the protocol never reports), plus a strict variant where ITW *bona* utterances are speaker-disjoint across folds (j5 splits bona randomly, so bona speakers appear in both LDA-train and test).

**Results** (`audit5_results.json`, `i7_per_fold_dEER.csv`, `audit5_fusion_decomposition.png`).

1. **I7 VERIFIED.** Baseline 0.272 → fused 0.163; ΔEER negative in all 5 folds × 3 seeds (range −0.074 to −0.185). Axis alone = 0.187, so fusion genuinely adds value over both components. The protocol (system-disjoint axis, λ chosen on train folds, unsupervised centroid axis) is sound; "zero-training" is fair *here*.
2. **ITW claim MISLABELED.** The artifact itself shows 0.292 is the **axis alone**; the actual fusion is **0.312** — i.e., on ITW fusion *hurts* relative to the axis. Recomputation: axis 0.301, fused 0.306, baseline 0.363. Additionally, no committed script produced `itw_fusion_test.json`.
3. **The 26–33-point "fusion" gains are MISLEADING.** Decomposition (identical protocol):

| corpus | AASIST alone | LDA axis alone | "fused" (reported) |
|---|---|---|---|
| MLAAD | 0.376 | **0.100** | 0.116 |
| ITW | 0.486 | **0.101** | 0.161 |
| ASVspoof21 | 0.073 | 0.115 | 0.076 |

Under domain shift the supervised LDA probe **alone beats the fused score**; fusing in the broken detector makes things *worse*. The headline gains therefore demonstrate "a supervised linear probe on frozen WavLM features trained on labeled eval-corpus data is much better than a shifted detector" — a true but very different (and well-known-shaped) claim. "Zero-training-cost" is inaccurate: the LDA axis is trained, with labels, on the evaluation corpus (group-disjoint, but in-corpus).
   - **Leakage quantified:** with strictly speaker-disjoint bona folds on ITW, LDA-alone goes 0.101→0.143 and fused 0.161→0.193. So ~3–4 EER points of the reported ITW numbers come from bona-speaker leakage across folds.

**Verdict.** I7 (MLAAD, unsupervised centroid axis, system-disjoint) stands. The domain-shift fusion story must be reframed: the actionable finding is *supervised lightweight adaptation in frozen-SSL space* (consistent with J3's calibration curves), not zero-cost fusion; the ITW WavLM "fusion" number is a mislabeled axis-alone result; and the j5 protocol needs speaker-disjoint bona splits.
