#!/usr/bin/env python3
"""Audit A4 — Is the "axis rotation" claim (cos(w_MLAAD, w_ITW)=0.05) real
rotation, or just noisy axis estimation / frame artifacts?

Tests:
  1. Recompute cos(w_MLAAD, w_ITW), cos(w_MLAAD, w_ASV21), cos(w_ITW, w_ASV21)
     in (a) the MLAAD-bona standardization frame (what the paper used) and
     (b) raw unstandardized space — frame sensitivity.
  2. Null: |cos| of random directions in 768-d (E[|cos|]≈0.029). Is 0.05
     distinguishable from "unrelated directions"?
  3. Reliability: split-half cosine of each corpus's own axis (50 splits).
     If each axis is internally stable (cos≈1) the cross-corpus cos=0.05 is
     genuine rotation, not estimation noise. Also disattenuated cross-corpus
     cosine = cos_obs / sqrt(rel_A * rel_B).
  4. Subsampled-n stability: ITW axis from n=200..1500/class — does the axis
     converge?
  5. Angle interpretation guard: in high-d, "near-orthogonal" is the default
     for ANY two directions; verify the claim adds content beyond chance by
     comparing cos(w_MLAAD, w_ASV21)=0.36 (same claim says substantial).
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import audit_common as ac

OUT = ac.OUT_ROOT / "audit4_axis_rotation"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

# corpora
ml = ac.load_mlaad()
X_ml, y_ml = ml["X"], ml["labels"]
itw_utt = pd.read_csv(ac.RES / "i5_itw_transfer" / "utt_table.csv")
X_itw = np.load(ac.RES / "i5_itw_transfer" / "features.npz")["X12"]
y_itw = itw_utt["label"].values
z21 = np.load(ac.RES / "j4_asvspoof21" / "waves_sel.npz", allow_pickle=True)
y_21 = (z21["atts"] != "bonafide").astype(int)
X_21 = np.load(ac.RES / "j4_asvspoof21" / "embeddings.npz")["X12"]

def axis(X, y, mu=None, sd=None):
    """centroid-difference axis; standardize by provided (or own-bona) stats"""
    if mu is None:
        mu = X[y == 0].mean(0); sd = X[y == 0].std(0) + 1e-9
    Z = (X - mu) / sd
    w = Z[y == 1].mean(0) - Z[y == 0].mean(0)
    return w / np.linalg.norm(w), mu, sd

res = {}
# ── 1. cross-corpus cosines, two frames ──────────────────────────────────────
w_ml, mu_ml, sd_ml = axis(X_ml, y_ml)
# (a) common MLAAD frame (paper's frame)
w_itw_mlf, _, _ = axis(X_itw, y_itw, mu_ml, sd_ml)
w_21_mlf, _, _ = axis(X_21, y_21, mu_ml, sd_ml)
# (b) raw frame
def axis_raw(X, y):
    w = X[y == 1].mean(0) - X[y == 0].mean(0)
    return w / np.linalg.norm(w)
wr_ml, wr_itw, wr_21 = axis_raw(X_ml, y_ml), axis_raw(X_itw, y_itw), axis_raw(X_21, y_21)
res["cos_mlaad_frame"] = {"ml_itw": float(w_ml @ w_itw_mlf),
                          "ml_asv21": float(w_ml @ w_21_mlf),
                          "itw_asv21": float(w_itw_mlf @ w_21_mlf)}
res["cos_raw_frame"] = {"ml_itw": float(wr_ml @ wr_itw),
                        "ml_asv21": float(wr_ml @ wr_21),
                        "itw_asv21": float(wr_itw @ wr_21)}
print("cosines (MLAAD frame):", {k: round(v, 3) for k, v in res["cos_mlaad_frame"].items()})
print("cosines (raw frame):  ", {k: round(v, 3) for k, v in res["cos_raw_frame"].items()})

# ── 2. random-direction null ─────────────────────────────────────────────────
d = X_ml.shape[1]
null = np.abs([rng.standard_normal(d) @ rng.standard_normal(d) /
               (np.linalg.norm(a := rng.standard_normal(d)) * 1) for _ in range(0)])
# do it properly:
A = rng.standard_normal((20000, d)); A /= np.linalg.norm(A, axis=1, keepdims=True)
B = rng.standard_normal((20000, d)); B /= np.linalg.norm(B, axis=1, keepdims=True)
null = np.abs((A * B).sum(1))
res["random_null"] = {"mean_abs_cos": float(null.mean()),
                      "p95_abs_cos": float(np.percentile(null, 95)),
                      "p_ml_itw_vs_null": float((null >= abs(res["cos_mlaad_frame"]["ml_itw"])).mean())}
print(f"random 768-d null: E|cos|={null.mean():.4f}, 95th pct={np.percentile(null,95):.4f}; "
      f"P(|cos_rand| >= 0.05)={res['random_null']['p_ml_itw_vs_null']:.3f}")

# ── 3. split-half reliability of each axis ───────────────────────────────────
def split_half_rel(X, y, n_rep=50, frame_mu=None, frame_sd=None):
    cs = []
    idx = np.arange(len(y))
    for r in range(n_rep):
        rr = np.random.default_rng(r)
        perm = rr.permutation(idx)
        h1, h2 = perm[: len(idx) // 2], perm[len(idx) // 2:]
        if frame_mu is None:
            w1, _, _ = axis(X[h1], y[h1])
            w2, _, _ = axis(X[h2], y[h2])
        else:
            w1, _, _ = axis(X[h1], y[h1], frame_mu, frame_sd)
            w2, _, _ = axis(X[h2], y[h2], frame_mu, frame_sd)
        cs.append(float(w1 @ w2))
    return float(np.mean(cs)), float(np.std(cs))

rel = {}
for name, X_, y_ in [("mlaad", X_ml, y_ml), ("itw", X_itw, y_itw), ("asv21", X_21, y_21)]:
    m, s = split_half_rel(X_, y_, frame_mu=mu_ml, frame_sd=sd_ml)
    rel[name] = {"split_half_cos_mean": m, "split_half_cos_sd": s}
    print(f"split-half axis reliability {name}: cos={m:.3f}±{s:.3f}")
res["reliability"] = rel

# disattenuated cross-corpus cosines (full-sample axes ≈ sqrt of split-half rel
# via Spearman-Brown-like correction: rel_full ≈ 2r/(1+r))
def rel_full(r): return 2 * r / (1 + r)
for pair, (a, b) in {"ml_itw": ("mlaad", "itw"), "ml_asv21": ("mlaad", "asv21"),
                     "itw_asv21": ("itw", "asv21")}.items():
    ra, rb = rel_full(rel[a]["split_half_cos_mean"]), rel_full(rel[b]["split_half_cos_mean"])
    res.setdefault("disattenuated", {})[pair] = float(
        res["cos_mlaad_frame"][pair] / np.sqrt(ra * rb))
print("disattenuated cosines:", {k: round(v, 3) for k, v in res["disattenuated"].items()})

# ── 4. ITW axis convergence with n ──────────────────────────────────────────
conv = []
for n in [100, 200, 400, 800, 1500]:
    cs = []
    for r in range(20):
        rr = np.random.default_rng(1000 + r)
        bi = rr.choice(np.where(y_itw == 0)[0], n, replace=False)
        si = rr.choice(np.where(y_itw == 1)[0], n, replace=False)
        sub = np.r_[bi, si]
        w_sub, _, _ = axis(X_itw[sub], y_itw[sub], mu_ml, sd_ml)
        cs.append(float(w_sub @ w_itw_mlf))
    conv.append({"n_per_class": n, "cos_to_full_mean": float(np.mean(cs)),
                 "cos_to_full_sd": float(np.std(cs))})
conv = pd.DataFrame(conv)
conv.to_csv(OUT / "itw_axis_convergence.csv", index=False)
print(conv.round(3).to_string(index=False))
res["itw_convergence"] = conv.to_dict("records")

(OUT / "audit4_results.json").write_text(json.dumps(res, indent=2))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].hist(null, bins=60, density=True, alpha=0.7, label="random 768-d |cos|")
for k, v in res["cos_mlaad_frame"].items():
    axes[0].axvline(abs(v), ls="--", label=f"{k}: {v:+.3f}")
axes[0].legend(fontsize=8); axes[0].set_xlabel("|cosine|")
axes[0].set_title("cross-corpus axis cosines vs random null")
axes[1].errorbar(conv.n_per_class, conv.cos_to_full_mean, yerr=conv.cos_to_full_sd, marker="o")
axes[1].set_xlabel("n per class"); axes[1].set_ylabel("cos(subsampled, full ITW axis)")
axes[1].set_title("ITW axis estimation stability")
fig.tight_layout()
fig.savefig(OUT / "audit4_rotation.png", dpi=150)
print(f"done -> {OUT}")
