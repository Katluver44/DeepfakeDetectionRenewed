#!/usr/bin/env python3
"""Audit A8 — I1 causal intervention contrasts: is the inference unit right?

The I1 contrasts (i1_contrasts.csv) report bootstrap p-values that treat
utterances as the unit, pooling 3 seeds. But several headline contrasts have
INCONSISTENT signs across seeds (seed_signs column), e.g. iso_0.7 = [-1,+1,-1]
yet p=0.024. With n_seed=3, seed is the right unit for "the effect replicates
across training runs".

Tests:
  1. Parse i1_conditions.csv; per-seed per-condition AUC; recompute contrasts
     treating SEED as the unit (paired t-test, n=3; sign consistency).
  2. Which of the paper's headline contrasts (iso_0.7, sub_top, sub_res,
     shuffle, gated-vs-ungated) survive seed-level inference?
  3. Multiplicity over the 25 contrasts (BH).
  4. Verify the "75-80% artifact" arithmetic: gated/ungated AUC deltas.
  5. Verify claim 'true gated C-effect is dAUC=-0.0036' vs artifact value.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats
import audit_common as ac

OUT = ac.OUT_ROOT / "audit8_i1_causal"
OUT.mkdir(parents=True, exist_ok=True)

cond = pd.read_csv(ac.RES / "i1_geometry_causal_decomp" / "i1_conditions.csv")
con = pd.read_csv(ac.RES / "i1_geometry_causal_decomp" / "i1_contrasts.csv")
piv = cond.pivot_table(index="seed", columns="cond", values="AUC")
print("conditions x seeds:", piv.shape)

rows = []
for r in con.itertuples():
    a, b = r.A, r.B
    if a not in piv.columns or b not in piv.columns:
        continue
    d = piv[a] - piv[b]
    t, p_t = stats.ttest_1samp(d, 0)
    signs = np.sign(d.values)
    rows.append({"A": a, "B": b, "dAUC_seedmean": float(d.mean()),
                 "dAUC_reported": r.dAUC, "p_reported": r.p,
                 "seed_consistent": bool(np.all(signs == signs[0])),
                 "t_seed": float(t), "p_seed_ttest": float(p_t),
                 "seed_deltas": [float(x) for x in d.values]})
R = pd.DataFrame(rows)
R["q_bh_seed"] = ac.bh_fdr(R.p_seed_ttest.values)
R.to_csv(OUT / "seed_level_contrasts.csv", index=False)
print(R[["A", "B", "dAUC_seedmean", "p_reported", "seed_consistent",
         "p_seed_ttest", "q_bh_seed"]].round(4).to_string(index=False))

headline = ["iso_0.7", "sub_top_0.7", "sub_res_0.7", "shuffle", "iso_0.7_ungated"]
res = {"seed_level": R.to_dict("records")}
res["headline_verdicts"] = {}
for h in headline:
    row = R[(R.A == h) & (R.B == "baseline")]
    if len(row):
        row = row.iloc[0]
        res["headline_verdicts"][h] = {
            "seed_consistent": bool(row.seed_consistent),
            "p_seed": float(row.p_seed_ttest),
            "verdict": ("survives seed-level inference" if row.seed_consistent and
                        row.p_seed_ttest < 0.05 else
                        "does NOT survive seed-level inference")}
gg = R[(R.A == "iso_0.7") & (R.B == "iso_0.7_ungated")]
if len(gg):
    res["gated_vs_ungated"] = gg.iloc[0][["dAUC_seedmean", "seed_consistent",
                                          "p_seed_ttest"]].to_dict()

# artifact share arithmetic
d_ungated = float(piv["iso_0.7_ungated"].mean() - piv["baseline"].mean())
d_gated = float(piv["iso_0.7"].mean() - piv["baseline"].mean())
res["artifact_share"] = {"ungated_dAUC": d_ungated, "gated_dAUC": d_gated,
                         "share": 1 - d_gated / d_ungated,
                         "paper_claim": "75-80%",
                         "paper_value_for_gated": -0.0036}
print(f"\nartifact share: gated={d_gated:+.4f} ungated={d_ungated:+.4f} "
      f"-> share={(1-d_gated/d_ungated):.1%} (paper: 75-80%); "
      f"paper quotes gated C-effect -0.0036 vs actual {d_gated:+.4f}")

(OUT / "audit8_results.json").write_text(json.dumps(res, indent=2, default=str))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
base = R[R.B == "baseline"].copy()
fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(base))
for i, r in enumerate(base.itertuples()):
    ds = r.seed_deltas
    ax.scatter([i] * len(ds), ds, c=("tab:green" if r.seed_consistent else "tab:red"),
               s=24, zorder=3)
ax.axhline(0, color="k", lw=0.7)
ax.set_xticks(x); ax.set_xticklabels(base.A, rotation=60, fontsize=7, ha="right")
ax.set_ylabel("ΔAUC vs baseline (per seed)")
ax.set_title("I1 contrasts: per-seed deltas (red = seeds disagree in sign)")
fig.tight_layout()
fig.savefig(OUT / "audit8_i1_seeds.png", dpi=150)
print(f"done -> {OUT}")
