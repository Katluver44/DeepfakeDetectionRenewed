#!/usr/bin/env python3
"""
I2b — Within-system fixed-effects panel test.
Does utterance-level rog@L12 predict the utterance's spoof logit *within* system,
with every system-level confounder (generator, vocoder, language, training data)
absorbed by fixed effects? Cluster-robust (by system) inference + permutation null
that shuffles rog within system (preserving all system-level structure).
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

SCRIPTS = Path(__file__).resolve().parent
OUT = SCRIPTS.parent / "results" / "i2_geometry_battery"
fdf = pd.read_csv(OUT / "utt_features.csv")
logits = np.load(OUT / "detector_logits.npy")
fdf["logit"] = logits[fdf["i"].values]
sp = fdf[fdf.label == 1].copy()
sp = sp.groupby("system").filter(lambda g: len(g) >= 8)
print(f"{len(sp)} spoof utts, {sp.system.nunique()} systems")

FEATS = ["rog_L12", "vel_mean_L12", "eff_rank_L12", "top1_frac_L12", "vel_entropy_L12",
         "dist_bona_centroid_L12", "vel_mean_L0"]
rng = np.random.default_rng(42)

def within_demean(df, cols):
    out = df.copy()
    for c in cols + ["logit"]:
        out[c] = df[c] - df.groupby("system")[c].transform("mean")
    return out

res = {}
for f in FEATS:
    d = within_demean(sp, [f])
    x = d[f].values; y = d["logit"].values
    # standardize x within (use overall std of demeaned x)
    xs = x / (x.std() + 1e-12)
    beta = float(np.sum(xs * y) / np.sum(xs * xs))
    # cluster-robust SE (CR0, clusters = system)
    e = y - beta * xs
    num = 0.0
    for _, g in pd.DataFrame({"s": sp.system.values, "x": xs, "e": e}).groupby("s"):
        num += (g.x.values @ g.e.values) ** 2
    se = float(np.sqrt(num) / np.sum(xs * xs))
    t = beta / (se + 1e-12)
    # permutation null: shuffle feature within system
    perms = []
    for _ in range(2000):
        xp = sp.groupby("system")[f].transform(lambda v: rng.permutation(v.values)).values
        xp = xp - pd.Series(xp).groupby(sp.system.values).transform("mean").values
        xp = xp / (xp.std() + 1e-12)
        perms.append(np.sum(xp * y) / np.sum(xp * xp))
    perms = np.array(perms)
    p_perm = float((np.abs(perms) >= abs(beta)).mean())
    res[f] = {"beta_std": beta, "se_cluster": se, "t": t, "p_perm": p_perm}
    print(f"{f:>24}: beta={beta:+.4f} (cluster t={t:+.2f}, perm p={p_perm:.4f})")

# joint: rog vs vel within system — which survives the other?
d = within_demean(sp, ["rog_L12", "vel_mean_L12"])
X = d[["rog_L12", "vel_mean_L12"]].values
X = (X) / (X.std(0) + 1e-12)
y = d["logit"].values
b = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"joint within-system: rog beta={b[0]:+.4f}, vel beta={b[1]:+.4f}")
res["joint_rog_vel"] = {"rog": float(b[0]), "vel": float(b[1])}
(OUT / "i2b_fixed_effects.json").write_text(json.dumps(res, indent=2))
print("saved i2b_fixed_effects.json")
