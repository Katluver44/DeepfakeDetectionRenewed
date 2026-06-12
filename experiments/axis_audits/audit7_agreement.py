#!/usr/bin/env python3
"""Audit A7 — Cross-detector hardness agreement (report §3.4).

Claim: "Same-domain pairs agree ρ≈0.55; cross-domain pairs ρ≈0.30–0.36
regardless of architecture — hardness is training-domain-conditional, not
architecture-conditional."

Red-team angles:
  1. Recompute the 4x4 matrix from per-system hardness CSVs; bootstrap CI
     (over the 61 systems) for every pair.
  2. The "same-domain" cell is a SINGLE pair (AASIST-FT vs WavLM-GAT). Is
     0.553 significantly larger than the cross-domain pairs? Steiger-style
     bootstrap difference test for overlapping correlations.
  3. RobustGoat vs AASIST-ZS = 0.544 is ALSO high and is labeled cross-?
     What domain is RobustGoat trained on? If ASVspoof-trained and AASIST-ZS
     is ASVspoof-trained, that's a second same-domain pair: does the
     domain-conditional story hold for it?
  4. Attenuate: hardness measurement noise differs per detector; report
     split-half-reliability-corrected agreement.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats
import audit_common as ac

OUT = ac.OUT_ROOT / "audit7_agreement"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
res = {}

j6 = pd.read_csv(ac.RES / "j6_aasist_mlaad" / "j6_system_table.csv", index_col=0)
j5h = pd.read_csv(ac.RES / "j5_aasist" / "mlaad_aasist_hardness.csv", index_col=0)
g2 = pd.read_csv(ac.RES / "j2_prospective" / "j2_goat_hardness.csv", index_col=0)
H = pd.DataFrame({
    "aasist_ft": j6["hard_aasist_ft"],          # trained MLAAD
    "aasist_zs": j5h["hard_aasist"],            # trained ASVspoof19
    "wavlm_gat": j6["hard_shared"],             # trained MLAAD
    "robust_goat": g2["hard_goat"],             # trained ASVspoof (robust)
}).dropna()
print(f"n systems = {len(H)}")
M = H.corr(method="spearman")
print(M.round(3).to_string())
res["matrix"] = M.round(4).to_dict()
res["n_systems"] = int(len(H))

pairs = [("aasist_ft", "wavlm_gat", "same-domain (MLAAD/MLAAD)"),
         ("aasist_zs", "robust_goat", "same-domain (ASVspoof/ASVspoof)"),
         ("aasist_ft", "aasist_zs", "same-arch cross-domain"),
         ("aasist_ft", "robust_goat", "cross/cross"),
         ("wavlm_gat", "aasist_zs", "cross/cross"),
         ("wavlm_gat", "robust_goat", "cross/cross")]
rows = []
B = 10000
n = len(H)
bidx = rng.integers(0, n, size=(B, n))
boot = {}
for a, b, lab in pairs:
    rho = stats.spearmanr(H[a], H[b])[0]
    rs = np.array([stats.spearmanr(H[a].values[ix], H[b].values[ix])[0] for ix in bidx[:3000]])
    boot[(a, b)] = rs
    rows.append({"pair": f"{a}~{b}", "type": lab, "rho": float(rho),
                 "ci_lo": float(np.nanpercentile(rs, 2.5)),
                 "ci_hi": float(np.nanpercentile(rs, 97.5))})
P = pd.DataFrame(rows)
P.to_csv(OUT / "pairwise_agreement_ci.csv", index=False)
print(P.round(3).to_string(index=False))
res["pairs"] = P.to_dict("records")

# difference tests: same-domain MLAAD pair vs each cross pair (shared bootstrap)
diffs = []
sd_rs = boot[("aasist_ft", "wavlm_gat")]
for a, b, lab in pairs[2:]:
    d = sd_rs - boot[(a, b)]
    diffs.append({"contrast": f"(aasist_ft~wavlm_gat) - ({a}~{b})",
                  "delta": float(np.nanmean(d)),
                  "ci_lo": float(np.nanpercentile(d, 2.5)),
                  "ci_hi": float(np.nanpercentile(d, 97.5)),
                  "p_two": float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))})
D = pd.DataFrame(diffs)
D.to_csv(OUT / "samedomain_vs_cross_difference.csv", index=False)
print(D.round(3).to_string(index=False))
res["difference_tests"] = D.to_dict("records")

# second same-domain pair check
res["second_same_domain_pair"] = {
    "pair": "aasist_zs~robust_goat (both ASVspoof-trained)",
    "rho": float(stats.spearmanr(H["aasist_zs"], H["robust_goat"])[0]),
    "note": "the paper's own framing implies this is same-domain; it is 0.544, "
            "consistent with the domain-conditional story, but the paper lists "
            "0.544 among unexplained cells without noting this"}

(OUT / "audit7_results.json").write_text(json.dumps(res, indent=2, default=str))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4.5))
ypos = np.arange(len(P))
colors = ["tab:green" if "same" in t else "tab:gray" for t in P.type]
ax.barh(ypos, P.rho, xerr=[P.rho - P.ci_lo, P.ci_hi - P.rho], color=colors)
ax.set_yticks(ypos); ax.set_yticklabels([f"{r.pair}\n{r.type}" for r in P.itertuples()],
                                        fontsize=8)
ax.set_xlabel("Spearman ρ (95% bootstrap CI)")
ax.set_title("cross-detector hardness agreement, with CIs")
fig.tight_layout()
fig.savefig(OUT / "audit7_agreement.png", dpi=150)
print(f"done -> {OUT}")
