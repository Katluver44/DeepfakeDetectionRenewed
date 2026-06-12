"""Shared loaders/statistics for the axis_audits suite.

Everything loads from cached artifacts (embeddings, logits, CSVs) produced by
the I/J experiment series — no GPU or torch needed, so every audit is an
independent recomputation from the same raw arrays the original scripts used.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, roc_curve

BASE = Path(__file__).resolve().parents[2]
EXP = BASE / "experiments"
RES = EXP / "results"
FO2 = BASE / "final_outputs2"
OUT_ROOT = EXP / "axis_audits" / "audits_outputs"
WAVE_CACHE = BASE / "outputs" / "px_wave_cache"
TEST_JSON = RES / "mlaad" / "baseline_eval" / "test_in_distribution.json"
MIN_UTTS = 8


def eer_of(y, s):
    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def load_mlaad():
    """MLAAD test set: labels, systems, L12 embeddings, 3-seed detector logits."""
    recs = json.loads(TEST_JSON.read_text())
    ok = np.load(WAVE_CACHE / "i2_full_test_waves.npz", allow_pickle=True)["ok_idx"]
    recs = [recs[i] for i in ok]
    labels = np.array([1 if str(r["label"]).lower().startswith("spoof") else 0 for r in recs])
    systems = np.array([f'{r["attack_system"]}|{r["language"]}' if labels[i] else "bona"
                        for i, r in enumerate(recs)])
    X = np.load(RES / "i3_position_geometry" / "embeddings.npz")["X12"]
    logits = {s: np.load(RES / "i3_position_geometry" / f"logits_{s}.npy")
              for s in ["main", "s42", "s1024"]}
    return dict(labels=labels, systems=systems, X=X, logits=logits, recs=recs)


def sys_list_of(systems, labels, min_utts=MIN_UTTS):
    return sorted({s for s in systems[labels == 1] if (systems == s).sum() >= min_utts})


def hardness_table(labels, systems, logits, sys_list):
    """Per-system hardness 1-AUC vs full bona pool, per seed + shared mean."""
    H = {}
    bona = labels == 0
    for sn, lg in logits.items():
        bl = lg[bona]
        H[sn] = {s: 1.0 - roc_auc_score(
            np.r_[np.zeros(len(bl)), np.ones((systems == s).sum())],
            np.r_[bl, lg[systems == s]]) for s in sys_list}
    H = pd.DataFrame(H)
    H["shared"] = H.mean(1)
    return H


def loso_axis_features(X, labels, systems, sys_list):
    """Replicates I3's LOSO axis: per-system s_along (median), sd_along, s_orth."""
    bona = labels == 0
    mu_b = X[bona].mean(0)
    sd_b = X[bona].std(0) + 1e-9
    Z = (X - mu_b) / sd_b
    mu_bona = Z[bona].mean(0)
    s_along = np.full(len(Z), np.nan)
    s_orth = np.full(len(Z), np.nan)
    for s in list(sys_list) + ["bona"]:
        m = systems == s
        sp_excl = (labels == 1) & (systems != s)
        w = Z[sp_excl].mean(0) - mu_bona
        w = w / (np.linalg.norm(w) + 1e-12)
        d = Z[m] - mu_bona
        a = d @ w
        s_along[m] = a
        s_orth[m] = np.linalg.norm(d - np.outer(a, w), axis=1)
    rows = []
    for s in sys_list:
        m = systems == s
        rows.append({"system": s, "n_utts": int(m.sum()),
                     "s_along": float(np.median(s_along[m])),
                     "sd_along": float(np.std(s_along[m])),
                     "s_orth": float(np.median(s_orth[m]))})
    return pd.DataFrame(rows).set_index("system"), s_along, s_orth


def loo_r2(Xf, y, alpha=1.0):
    """Identical estimator to I3/J1: ridge leave-one-out R^2 on standardized X."""
    Xf = np.asarray(Xf, float)
    if Xf.ndim == 1:
        Xf = Xf.reshape(-1, 1)
    n = len(y)
    Xs = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-12)
    pred = np.empty(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        pred[i] = Ridge(alpha=alpha).fit(Xs[m], y[m]).predict(Xs[i:i + 1])[0]
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def perm_p_loo(Xf, y, obs, n_perm=2000, seed=0):
    rng = np.random.default_rng(seed)
    c = 0
    for _ in range(n_perm):
        c += loo_r2(Xf, rng.permutation(y)) >= obs
    return (c + 1) / (n_perm + 1)


def spearman_perm_p(x, y, n_perm=100000, seed=0, alternative="two-sided"):
    """Exact-style permutation p for Spearman rho (needed at small n)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); y = np.asarray(y, float)
    obs, _ = stats.spearmanr(x, y)
    cnt = 0
    for _ in range(n_perm):
        r, _ = stats.spearmanr(x, rng.permutation(y))
        if alternative == "two-sided":
            cnt += abs(r) >= abs(obs) - 1e-12
        else:
            cnt += r >= obs - 1e-12
    return obs, (cnt + 1) / (n_perm + 1)


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank_idx in range(n - 1, -1, -1):
        i = order[rank_idx]
        val = p[i] * n / (rank_idx + 1)
        prev = min(prev, val)
        q[i] = prev
    return q


def bootstrap_spearman_ci(x, y, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    rs = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        if len(np.unique(x[ix])) < 3 or len(np.unique(y[ix])) < 3:
            continue
        rs.append(stats.spearmanr(x[ix], y[ix])[0])
    rs = np.array(rs)
    return float(np.nanpercentile(rs, 2.5)), float(np.nanpercentile(rs, 97.5))
