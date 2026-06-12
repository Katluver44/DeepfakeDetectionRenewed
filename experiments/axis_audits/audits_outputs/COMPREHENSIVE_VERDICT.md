# Red-Team Audit of the Geometric-Axis Hardness Claims (final_outputs2)

**Scope.** Independent recomputation and adversarial stress-testing of every quantitative claim in `final_outputs2` (experiments I1–I7, J1–J6): the sd_along/s_along/s_orth/vel_entropy hardness predictors on MLAAD, the ASVspoof 2021 prospective predictions, the In-The-Wild transfer and per-speaker results, the AASIST replications, the axis-rotation law, the fusion/adaptation results, and the causal intervention battery. All audits run from the cached raw artifacts (embeddings, logits, scores) — none rely on the original analysis code paths. Ten audits; code in `experiments/axis_audits/`, outputs in `experiments/axis_audits/audits_outputs/audit{1..10}_*/`.

---

## 1. What is proven beyond reasonable scrutiny

**P1. sd_along → MLAAD hardness (the core law).** Independently recomputed: LOSO R² = 0.277 (claimed 0.273), permutation p = 0.0002, Spearman ρ = +0.597 (p < 1e-6). It is the **only** test among all 83 recorded in the project that survives *global* BH-FDR across everything ever tested (q < 0.001). It survives: leave-one-system jackknife (0/61 folds lose significance), removal of the 5 most influential systems (ρ rises to 0.66), partialling of n_utts/RMS/language/vel_entropy, estimator substitutions (ridge α, IQR spread, log-hardness), all three detector seeds, and an explicit winner's-curse simulation (wins 76.7% of discovery halves; 99.2% confirmation rate at p<0.05). Raising the min-utterance threshold *strengthens* it (ρ = 0.60→0.64→0.69 at ≥8/12/16 utts) — the attenuation signature of a true effect measured noisily. [Audits 1, 2, 9]

**P2. The hardness metric itself is sound.** Split-half reliability 0.86; cross-seed agreement 0.89–0.97; shared-bona-pool dependence negligible (rank stability ≈ 1.0 under bona bootstrap). The R² ceiling is ≈0.86, so sd_along captures ~32% of *explainable* variance. [Audit 9]

**P3. Axis rotation across recording domains is real — and stronger than claimed.** Each corpus's internal axis is estimated almost noiselessly (split-half cos 0.93–0.98), yet cross-corpus cosines are at/near the 768-d chance floor (E|cos| = 0.029): MLAAD↔ITW = 0.03–0.10 depending on frame, and MLAAD↔ASVspoof21 is actually *negative* (−0.14 to −0.21), consistent with the transferred-axis predictor P4 failing with the wrong sign. Corpus-internal axis estimation is therefore necessary, exactly as claimed. [Audit 4]

**P4. Cross-detector agreement is domain-conditional.** Verified matrix, plus a finding the paper missed: there are **two** same-domain cross-architecture pairs (AASIST-FT~WavLM-GAT = 0.553; AASIST-ZS~RobustGoat = 0.544, both ASVspoof-trained), and both sit above all four cross-domain pairs (0.30–0.36), including the same-architecture cross-domain pair (0.308). Caveat: no individual same-vs-cross difference is significant at n=61 (bootstrap p = 0.06–0.22); claim the pattern, not a significant gap. [Audit 7]

**P5. I7 MLAAD axis fusion (0.272 → 0.163) is clean.** System-disjoint folds, λ from train folds, unsupervised centroid axis; ΔEER negative in all 5 folds × 3 seeds; fusion beats both components (axis alone 0.187). "Zero-training-cost" is fair for this experiment specifically. [Audit 5]

**P6. The ungated-hook artifact (E9 correction) and the direction-content asymmetry.** Gated-vs-ungated is seed-consistent (+0.010, all seeds, seed-level p = 0.04); artifact share recomputed at 81% ("75–80%" claim ✓). sub_top (−0.032) vs sub_res (+0.004) is sign-consistent across all seeds with a 10× effect separation — the directional-not-scalar mechanism story is qualitatively supported. [Audit 8]

## 2. What is suggestive but NOT proven

**S1. ASVspoof 2021 prospective prediction (P3, ρ = 0.60/0.64).** The registered *primary* predictor P1 failed; P3 is a promoted secondary among 4 registered predictors; the earlier prospective attempt (J2) failed 9/9 tests; n = 13 attacks. Exact permutation p = 0.034/0.020 uncorrected → Holm 0.13/0.08 within-family, BH q = 0.134; nothing survives. The "pre-registration" was written 63 s before scoring by the same script (no external commitment). In its favor: the two detector replications are quasi-independent (their hardness profiles correlate only ρ = 0.21), measurement-noise bootstrap never crosses zero, and leave-one-attack-out stays in [0.49, 0.75]. **Honest framing: a promising exploratory result requiring one genuinely held-out confirmation (e.g., ASVspoof21 DF, or the codec conditions) before any "prospective/pre-registered" language is used.** [Audit 3]

**S2. ITW per-speaker geometry (s_orth ρ = +0.534).** Survives the 7-test family correction (Holm p = 0.017) and confound partials — but plain radius-of-gyration (rog12, ρ = −0.561) is *stronger*, unreported, collinear with s_orth (ρ = −0.687), and neither survives partialling the other. The robust claim is "a single spread/compactness factor predicts ITW speaker hardness"; the specific *axis* interpretation (off-axis residual) is not separable on current evidence, and a layer-0 velocity feature also predicting hardness hints at a channel confound. n = 29 speakers (not 58). [Audit 6]

**S3. sd_along on AASIST-FT (ρ = +0.349, p = 0.006).** The Spearman form replicates and survives within-family BH, but the LOSO-R² form of the law *fails* on AASIST-FT (r² = −0.03, perm p = 0.13) and this is unreported; the paper switches metrics silently between detectors. Claim "rank-order replication," not "the law replicates." [Audits 1, 10]

**S4. The I1 fine-grained causal effects.** With seed as the inference unit (n = 3), no contrast survives BH; the utterance-level `***` p-values are pseudo-replicated. The "small residual causal C effect" (iso_0.7, p = 0.024) is **contradicted** — seeds disagree in sign. Keep the artifact correction and the sub_top/sub_res asymmetry (sign-consistent); drop or re-power everything else (3–5 more seeds would suffice for the headline contrasts). [Audit 8]

## 3. What is wrong and must be fixed before submission

1. **"cos(w_MLAAD, w_ASVspoof21) ≈ 0.36" is false** (true value negative; the 0.362 is J1's within-MLAAD cos(w_mean, w_lda) transplanted). [Audit 4]
2. **The ITW "fusion" 0.363→0.292 is mislabeled** — 0.292 is the axis alone; actual fusion is 0.312 (fusion *hurts* on ITW for WavLM-GAT). No committed script generates `itw_fusion_test.json`. [Audit 5]
3. **The 26–33-point "zero-training-cost fusion" gains under domain shift are misattributed.** The supervised eval-corpus LDA probe *alone* beats the fused score (ITW: 0.101 vs 0.161; MLAAD: 0.100 vs 0.116); fusing the shifted detector in makes it worse, and ~3–4 EER points of the ITW number come from bona-speaker leakage across folds (strict speaker-disjoint: 0.193). The actionable story is *supervised lightweight adaptation in frozen-SSL space* (consistent with J3), not zero-cost fusion. [Audit 5]
4. **Numerical/labeling errors:** table2's "+0.316" is the triplet R² presented as sd_along; "true gated C-effect −0.0036" is the wrong contrast (−0.0023); "58 speakers" → 29 analyzed; "13 interventions" → 22 conditions; Figure 1's left panel prints s_along's stats (R² = 0.020) under the sd_along scatter, and its middle panel mixes I4's ρ = −0.714 with J4's P3 axis label. Tally across the report: 8 verified, 6 mislabeled, 5 misleading, 2 contradicted. [Audit 10]
5. **The I2 feature battery has zero FDR survivors** (best q = 0.078) and should not be presented as supporting evidence without that disclosure. [Audit 1]

## 4. Failure scenarios tested and excluded

- *Winner's curse / garden of forking paths on the core law* — excluded by split-half discovery/confirmation simulation. [A1]
- *sd_along as an n_utts, RMS, or language artifact* — excluded by partials and within-English analysis. [A2]
- *Hardness metric artifacts (shared bona pool, small-n systems, seed idiosyncrasy)* — excluded; sensitivity analyses point the other way. [A9]
- *Axis rotation as estimation noise* — excluded by split-half axis reliability (0.93–0.98). [A4]
- *Cross-detector agreement driven by one architecture family* — excluded; both same-domain pairs are cross-architecture. [A7]
- *I7 fusion as a fold-concatenation artifact* — excluded by per-fold ΔEER. [A5]
- *Replication dependence between WavLM and AASIST on ASVspoof21* — tested; they are quasi-independent (ρ = 0.21), which *helps* the prospective story. [A3]

## 5. Recommended actions (priority order)

1. Fix the two contradicted numbers and six mislabels (§3); regenerate every report number programmatically from artifacts.
2. Reframe ASVspoof21 results as exploratory; run one decisive held-out confirmation with a single pre-committed predictor (P3) — externally timestamped.
3. Reframe domain-shift fusion as supervised adaptation; report LDA-alone alongside fused; make bona folds speaker-disjoint.
4. Merge s_orth/rog12 into one ITW spread factor or run a discriminating experiment; correct the speaker count.
5. Re-derive I1 inference at the seed level; add seeds for the headline contrasts; retract the residual-C-effect claim.
6. Report the hardness reliability ceiling (0.86) alongside all R²; add the min-utts sensitivity row.
7. Commit the missing ITW fusion script (or delete the artifact and regenerate via a committed script).

**Bottom line.** The central scientific claim — *per-system spread along the corpus-internal natural↔synthetic axis in frozen WavLM-L12 space predicts detection hardness, the axis rotates across recording domains, and hardness is training-domain-conditional rather than architecture-conditional* — survives aggressive red-teaming and is publishable. The ASVspoof21 "prospective" framing, the domain-shift "zero-training fusion" framing, the residual causal-C claim, and roughly a dozen packaged numbers do not survive and would be found by competent reviewers; they should be corrected or reframed exactly as itemized above.
