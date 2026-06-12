#!/usr/bin/env python3
"""Audit A10 — Number-by-number consistency of final_outputs2 claims vs artifacts.

Each row: claim (verbatim source), artifact value(s), verdict.
Verdicts: VERIFIED / MISLABELED / CONTRADICTED / UNVERIFIABLE / MISLEADING.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import audit_common as ac

OUT = ac.OUT_ROOT / "audit10_consistency"
OUT.mkdir(parents=True, exist_ok=True)

def J(p):
    return json.loads(Path(p).read_text())

i3 = J(ac.RES / "i3_position_geometry" / "i3_stats.json")
i5 = J(ac.RES / "i5_itw_transfer" / "i5_stats.json")
j4 = J(ac.RES / "j4_asvspoof21" / "j4_results.json")
j5 = J(ac.RES / "j5_aasist" / "j5_results.json")
j6 = J(ac.RES / "j6_aasist_mlaad" / "j6_results.json")
itwf = J(ac.RES / "i7_axis_fusion" / "itw_fusion_test.json")
i7s = J(ac.RES / "i7_axis_fusion" / "i7_stats.json")
a2 = J(ac.OUT_ROOT / "audit2_sdalong_claim" / "audit2_results.json")
a4 = J(ac.OUT_ROOT / "audit4_axis_rotation" / "audit4_results.json")
a5 = J(ac.OUT_ROOT / "audit5_fusion_claims" / "audit5_results.json")
a8 = J(ac.OUT_ROOT / "audit8_i1_causal" / "audit8_results.json")
kn = J(ac.FO2 / "summary_assets" / "key_numbers.json")

rows = []
def add(claim, source, artifact, verdict, note=""):
    rows.append({"claim": claim, "source": source, "artifact_value": artifact,
                 "verdict": verdict, "note": note})

add("sd_along LOSO R²=0.273 on MLAAD (p=0.0005)", "abstract / §3.1 / key_numbers",
    f"recomputed R²={a2['headline']['recomputed_loso_r2']:.3f}, perm p={a2['headline']['perm_p']:.4f}",
    "VERIFIED",
    "number was hardcoded in build_package.py (no artifact), but independent "
    "recomputation reproduces it; table2 variant says '+0.316' which is the "
    "TRIPLET model R², a different quantity")

add("table2: 'sd_along +0.316 (LOSO R²), p=0.0005'", "tables/table2_hardness_law.md",
    "0.316 is the triplet (s_along+vel_entropy+sd_along) R²; sd_along alone = 0.277",
    "MISLABELED", "two different R² values presented as the same quantity in "
    "abstract (0.273) vs table2 (0.316)")

add("sd_along replicates on AASIST-FT (rho=+0.349, p=0.006)", "abstract / §3.1",
    f"j6 rho={j6['law']['sd_along']['rho']:.3f} p={j6['law']['sd_along']['p']:.4f}; "
    f"BUT j6 LOSO regression r2={j6['law']['loo_pos_sd']['r2']:.3f} (negative, ns)",
    "MISLEADING",
    "Spearman replicates; the LOSO-R² version of the law FAILS on AASIST-FT "
    "(r2=-0.03, perm p=0.13) and this is not reported; metric switched between "
    "detectors")

add("P3 rho=+0.599 (p=0.031) prospective, pre-registered", "abstract / §3.2",
    "verbatim values match j4_results; but registered PRIMARY P1 failed "
    "(rho=0.25, ns); P3 was one of 4 registered predictors; Holm-corrected "
    "p=0.13 (within-family), 0.24 (8-test family); J2 prior prospective "
    "attempt failed on 9 tests",
    "MISLEADING", "selective promotion of a non-primary predictor; "
    "registration 63s before scoring in same script run")

add("P3 replicates for AASIST (rho=+0.643, p=0.018)", "abstract / §3.2",
    "verbatim match (j5 H1); detectors' hardness profiles correlate only "
    "rho=0.21, so quasi-independent; Holm within-family p=0.082",
    "VERIFIED", "with multiplicity caveat; directional robustness confirmed "
    "by bootstrap (never crosses 0) and LOAO jackknife")

add("'P3 achieves 2/3 top-3 system hits'", "§3.2",
    "WavLM: 2/3 (P(>=2 by chance)=0.108); AASIST: 1/3 (not mentioned); "
    "P1 and P2 predicted the SAME top-3 set",
    "MISLEADING", "the hit metric does not distinguish P3 from the failed P1/P2")

add("cos(w_MLAAD, w_ITW) = 0.05 (near-orthogonal)", "abstract / §3.3 / key_numbers",
    f"recomputed: {a4['cos_mlaad_frame']['ml_itw']:.3f} (MLAAD frame), "
    f"{a4['cos_raw_frame']['ml_itw']:.3f} (raw); split-half axis reliability "
    f">0.93 in all corpora -> rotation is real, not estimation noise",
    "VERIFIED", "qualitatively; exact value is frame-dependent (0.03-0.10) and "
    "the 0.0525 artifact has no committed generating script")

add("cos(w_MLAAD, w_ASVspoof21) ≈ 0.36", "§4 Key Findings",
    f"recomputed: {a4['cos_mlaad_frame']['ml_asv21']:.3f} (MLAAD frame), "
    f"{a4['cos_raw_frame']['ml_asv21']:.3f} (raw) — NEGATIVE; 0.362 is "
    "cos(w_mean, w_lda) WITHIN MLAAD from J1, a different quantity",
    "CONTRADICTED", "apparent copy/confusion of an unrelated number")

add("ITW-internal axis fusion: EER 0.363 -> 0.292 (WavLM-GAT)", "§3.3",
    f"artifact: axis ALONE = {itwf['itw_internal_axis_speaker_disjoint']['EER']}, "
    f"fusion = {itwf['fusion_internal']['lam1.5']['EER']} (worse than axis alone); "
    f"recomputed: axis 0.301, fused 0.306",
    "MISLABELED", "0.292 is not a fusion result; fusion HURTS on ITW for "
    "WavLM-GAT; also no committed script produced itw_fusion_test.json")

add("fusion reduces EER by up to 33 pts under shift (AASIST-ZS on ITW 0.486->0.161), "
    "'zero-training-cost'", "abstract / §4 / §3.3",
    f"decomposition: supervised LDA probe ALONE achieves "
    f"{a5['j5_h4_decomposition']['itw']['lda_axis_alone_eer']:.3f} on ITW and "
    f"{a5['j5_h4_decomposition']['mlaad']['lda_axis_alone_eer']:.3f} on MLAAD — "
    "better than the fused 0.161/0.116; strict speaker-disjoint bona split: "
    f"fused={a5['j5_h4_decomposition']['itw']['strict_bona_disjoint']['fused_eer']:.3f}",
    "MISLEADING",
    "the gain comes from a supervised linear probe trained on labeled "
    "eval-corpus data, not from fusing the detector; bona speaker leakage "
    "inflates the ITW number; 'zero-training-cost' is inaccurate for these "
    "headline numbers (it is fair only for I7's MLAAD centroid-axis fusion)")

add("I7 MLAAD fusion 0.272 -> 0.163", "§3.1",
    f"verified: baseline {a5['i7_mlaad']['baseline_mean']:.3f} -> fused "
    f"{a5['i7_mlaad']['fused_mean']:.3f}; per-fold dEER all negative, all seeds",
    "VERIFIED", "system-disjoint protocol is sound; axis-alone=0.187 so fusion "
    "adds genuine value here")

add("iso_0.7 'small causal C effect' (dAUC=-0.0023, p=0.024)", "§5 I1 table",
    "seed signs [-1,+1,-1]; seed-level t-test p=0.21",
    "CONTRADICTED", "effect does not replicate across the 3 seeds; "
    "utterance-level bootstrap p-value is pseudo-replicated")

add("sub_top dAUC=-0.0320*** / sub_res +0.0039*** (direction content causal)",
    "§2.3 / §5",
    "values match artifact; seed-consistent in sign; seed-level p=0.106/0.145 "
    "(n=3); no I1 contrast survives seed-level BH",
    "VERIFIED", "directionally, with weaker statistical support than *** implies")

add("'true gated C-effect is dAUC = -0.0036'", "§4 Key Findings",
    f"gated iso_0.7 vs baseline = {a8['artifact_share']['gated_dAUC']:+.4f}; "
    "-0.0036 is the iso_0.7 vs shift_0.7 contrast",
    "MISLABELED", "")

add("artifact accounts for ~75-80% of prior compaction effect", "§2.3 / §4",
    f"recomputed share = {a8['artifact_share']['share']:.1%}",
    "VERIFIED", "")

add("ITW: 'per-speaker hardness: s_orth rho=+0.534 (p=0.003)'", "§3.3",
    "verbatim match; survives Holm within 7-feature family (p=0.017); BUT "
    "rog12 rho=-0.561 (p=0.0015) is stronger and unreported; the two are "
    "collinear (rho=-0.687) and neither survives partialling out the other",
    "MISLEADING", "the specific attribution to s_orth (an axis quantity) over "
    "plain radius-of-gyration is not supported")

add("ITW: '58 speakers'", "§2.1",
    "subset analyzed: 49 spoof speakers present, 29 with >=10 utts used for "
    "the hardness law", "MISLABELED", "")

add("same-domain cross-arch agreement rho=0.553; cross-domain 0.30-0.36",
    "abstract / §3.4",
    "matrix verified; second same-domain pair (AASIST-ZS~RobustGoat, both "
    "ASVspoof-trained) = 0.544 SUPPORTS the story but is not identified as "
    "same-domain in the paper; same-vs-cross differences not individually "
    "significant at n=61 (bootstrap p=0.06-0.22)",
    "VERIFIED", "direction supported by two independent same-domain pairs; "
    "claim of significance for the difference would be unsupported")

add("J3: adaptive LDA head with 250 labels halves EER (MLAAD -0.136, ITW -0.171)",
    "§3.5 / key_numbers",
    "matches j3 artifacts; calibration is group-disjoint",
    "VERIFIED", "framing should note this is supervised adaptation, "
    "consistent with A5's decomposition")

add("'13 controlled causal interventions' (abstract)", "abstract",
    "I1 has 22 conditions x 3 seeds; 21 non-baseline conditions",
    "MISLABELED", "count inconsistency")

add("fig1 left: 'MLAAD LOSO R²=0.020, p=0.023' under sd_along scatter",
    "figures/fig1",
    "0.020/0.023 are s_along's stats; the scatter shows sd_along whose "
    "R²=0.277", "MISLABELED",
    "understates own result; middle panel mixes i4's rho=-0.714 (different "
    "corpus subset) with a 'P3 (LDA axis score)' x-axis from J4")

T = pd.DataFrame(rows)
T.to_csv(OUT / "claims_vs_artifacts.csv", index=False)
counts = T.verdict.value_counts()
print(counts.to_string())
print()
for r in rows:
    print(f"[{r['verdict']:>12}] {r['claim'][:80]}")
(OUT / "audit10_results.json").write_text(json.dumps(
    {"verdict_counts": counts.to_dict(), "rows": rows}, indent=2))
print(f"done -> {OUT}")
