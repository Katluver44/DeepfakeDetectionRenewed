#!/usr/bin/env python3
"""
I2 — Within-utterance geometry battery vs per-system hardness (MLAAD, full power)
==================================================================================
Adjudicates whether C = -rog@L12 is (A) the operative geometric quantity, (B) a
proxy for a richer geometric structure (spectrum shape / dynamics / intrinsic dim),
or (C) a low-dimensional summary of a multi-feature factor.

Upgrades over the prior layerwise analysis (63 systems x 2 utts):
  * ALL spoof utterances of the MLAAD test in-distribution split (~15/system),
    units = attack_system x language with >= MIN_UTTS spoof utts.
  * Hardness target measured with the actual detector (mlaad_robust_goat full-model
    logit): per-system AUC vs the full bona pool -> hardness = 1 - AUC.
  * 14-feature within-utterance geometry battery at frozen pretrained WavLM-base
    L12 and L9 (+ L0 smoothness baseline), median-aggregated per system.
  * Inference: LOSO (leave-one-system-out) R^2, system-level permutation nulls,
    unique-variance (commonality) of rog vs the rest of the battery, partial
    Spearman, language-residualized replication, PCA factor structure.

Outputs -> experiments/results/i2_geometry_battery/
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import px_common as px

OUT = px.EXP_DIR / "results" / "i2_geometry_battery"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = px.DEVICE
MIN_UTTS = 8
LAYERS = (0, 9, 12)

# ─── 1. load all MLAAD test in-distribution utterances ────────────────────────
recs = json.loads(px.TEST_JSON.read_text())
print(f"[I2] {len(recs)} records")

def load_wave(r):
    return px._fix(np.asarray(torch.load(px.PROC_DIR / r["audio_path"]), np.float32))

CACHE = px.WAVE_CACHE / "i2_full_test_waves.npz"
if CACHE.exists():
    z = np.load(CACHE, allow_pickle=True)
    waves, ok_idx = z["waves"].astype(np.float32), z["ok_idx"]
else:
    waves, ok_idx = [], []
    for i, r in enumerate(recs):
        try:
            waves.append(load_wave(r).astype(np.float16)); ok_idx.append(i)
        except Exception:
            pass
        if (i + 1) % 400 == 0: print(f"  waves {i+1}/{len(recs)}", flush=True)
    waves = np.stack(waves); ok_idx = np.array(ok_idx)
    np.savez_compressed(CACHE, waves=waves, ok_idx=ok_idx)
    waves = waves.astype(np.float32)
recs = [recs[i] for i in ok_idx]
labels = np.array([1 if str(r["label"]).lower().startswith("spoof") else 0 for r in recs])
systems = np.array([f'{r["attack_system"]}|{r["language"]}' if labels[i] else "bona"
                    for i, r in enumerate(recs)])
langs = np.array([r["language"] for r in recs])
print(f"  loaded {len(recs)} (bona={int((labels==0).sum())} spoof={int(labels.sum())}), "
      f"{len(set(systems))-1} system-language units")

# ─── 2. detector logits (full model) ──────────────────────────────────────────
LOGIT_CACHE = OUT / "detector_logits.npy"
if LOGIT_CACHE.exists():
    logits = np.load(LOGIT_CACHE)
else:
    gm = px.load_detector()
    logits = px.detector_logits(gm, waves)
    np.save(LOGIT_CACHE, logits)
    del gm; torch.cuda.empty_cache()
eer, _ = px.compute_eer(labels, logits)
print(f"  detector EER on full split: {eer:.3f}")

# ─── 3. frozen pretrained WavLM-base hidden states -> geometry battery ────────
def twonn_id(F, rng):
    n = len(F)
    if n < 10: return np.nan
    idx = rng.choice(n, min(n, 120), replace=False)
    X = F[idx]
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    s = np.sort(D, 1)
    mu = s[:, 1] / (s[:, 0] + 1e-12)
    mu = mu[np.isfinite(mu) & (mu > 1)]
    return float(len(mu) / np.sum(np.log(mu))) if len(mu) else np.nan

def battery(F, rng):
    """F: (T,768) float32 frames of one utterance at one layer."""
    T = len(F)
    mu = F.mean(0); S = F - mu
    rog = float(np.sqrt((S ** 2).sum(1).mean()))
    # spectrum of within-utt covariance via SVD of S
    sv = np.linalg.svd(S, compute_uv=False)
    ev = sv ** 2; ev = ev / (ev.sum() + 1e-12)
    eff_rank = float(np.exp(-(ev * np.log(ev + 1e-12)).sum()))
    part_ratio = float(1.0 / ((ev ** 2).sum() + 1e-12))
    top1_frac = float(ev[0])
    v = np.diff(F, axis=0); vn = np.linalg.norm(v, axis=1)
    vel_mean = float(vn.mean()); vel_cv = float(vn.std() / (vn.mean() + 1e-12))
    nb = max(5, min(20, len(vn) // 4))
    h, _ = np.histogram(vn, bins=nb); h = h.astype(float) + 1e-8; h /= h.sum()
    vel_entropy = float(-(h * np.log(h)).sum())
    a = np.diff(v, axis=0); accel_mean = float(np.linalg.norm(a, axis=1).mean())
    cs = (v[1:] * v[:-1]).sum(1) / (np.linalg.norm(v[1:], axis=1) * np.linalg.norm(v[:-1], axis=1) + 1e-12)
    curvature = float((1 - cs).mean())
    tortuosity = float(vn.sum() / (2 * rog + 1e-12))
    # recurrence: far-in-time pairs closer than the median adjacent-frame step
    D = np.linalg.norm(F[:, None] - F[None], axis=-1)
    eps = float(np.median(vn))
    ii, jj = np.triu_indices(T, k=11)
    recurrence = float((D[ii, jj] < eps).mean()) if len(ii) else np.nan
    tid = twonn_id(F, rng)
    return dict(rog=rog, eff_rank=eff_rank, part_ratio=part_ratio, top1_frac=top1_frac,
                vel_mean=vel_mean, vel_cv=vel_cv, vel_entropy=vel_entropy,
                accel_mean=accel_mean, curvature=curvature, tortuosity=tortuosity,
                recurrence=recurrence, twonn_id=tid, mu_norm=float(np.linalg.norm(mu)))

FEAT_CACHE = OUT / "utt_features.csv"
if FEAT_CACHE.exists():
    fdf = pd.read_csv(FEAT_CACHE)
else:
    wl = px.load_frozen_wavlm(DEVICE)
    rows = []
    rng = np.random.default_rng(SEED)
    mu_store = {L: [] for L in LAYERS}  # per-utt centroid at each layer (for bona centroid)
    BS = 8
    with torch.no_grad():
        for b in range(0, len(waves), BS):
            xb = torch.as_tensor(waves[b:b + BS], dtype=torch.float32, device=DEVICE)
            hs = wl(xb, output_hidden_states=True).hidden_states  # 13 x (B,T,768)
            for k in range(xb.shape[0]):
                i = b + k
                row = {"i": i, "label": int(labels[i]), "system": systems[i], "language": langs[i]}
                for L in LAYERS:
                    F = hs[L][k].float().cpu().numpy()
                    if L == 0:
                        v = np.diff(F, axis=0)
                        row["vel_mean_L0"] = float(np.linalg.norm(v, axis=1).mean())
                        row["rog_L0"] = float(np.sqrt(((F - F.mean(0)) ** 2).sum(1).mean()))
                        mu_store[L].append(F.mean(0))
                        continue
                    feats = battery(F, rng)
                    row.update({f"{n}_L{L}": x for n, x in feats.items()})
                    mu_store[L].append(F.mean(0))
                rows.append(row)
            if (b // BS) % 20 == 0:
                print(f"  battery {b}/{len(waves)}", flush=True)
    fdf = pd.DataFrame(rows)
    # distance of utt centroid to the bona centroid cloud (L12 and L9)
    for L in (9, 12):
        M = np.stack(mu_store[L]); bona_mu = M[labels == 0].mean(0)
        fdf[f"dist_bona_centroid_L{L}"] = np.linalg.norm(M - bona_mu, axis=1)
    fdf.to_csv(FEAT_CACHE, index=False)
    del wl; torch.cuda.empty_cache()
fdf["logit"] = logits[fdf["i"].values]
print(f"  battery: {fdf.shape}")

# ─── 4. per-system hardness + feature aggregation ─────────────────────────────
from sklearn.metrics import roc_auc_score
bona_logits = fdf.loc[fdf.label == 0, "logit"].values
sys_rows = []
for s, g in fdf[fdf.label == 1].groupby("system"):
    if len(g) < MIN_UTTS: continue
    y = np.r_[np.zeros(len(bona_logits)), np.ones(len(g))]
    sc = np.r_[bona_logits, g["logit"].values]
    auc = roc_auc_score(y, sc)
    row = {"system": s, "language": g["language"].iloc[0], "n_utt": len(g),
           "hardness": 1.0 - auc, "median_logit": float(g["logit"].median())}
    for c in g.columns:
        if c.endswith(("_L9", "_L12", "_L0")):
            row[c] = float(g[c].median())
    sys_rows.append(row)
sdf = pd.DataFrame(sys_rows).dropna(axis=1, how="any")
sdf.to_csv(OUT / "system_table.csv", index=False)
FEATS = [c for c in sdf.columns if c.endswith(("_L9", "_L12", "_L0"))]
print(f"  {len(sdf)} system units (>= {MIN_UTTS} utts), {len(FEATS)} features; "
      f"hardness range [{sdf.hardness.min():.3f}, {sdf.hardness.max():.3f}]")

# ─── 5. inference ─────────────────────────────────────────────────────────────
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression

H = sdf["hardness"].values
N = len(sdf)
rng = np.random.default_rng(SEED)

def loo_r2(X, y, alpha=1.0):
    X = np.asarray(X, float); y = np.asarray(y, float)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    pred = np.empty(N)
    for i in range(N):
        m = np.ones(N, bool); m[i] = False
        r = Ridge(alpha=alpha).fit(Xs[m], y[m])
        pred[i] = r.predict(Xs[i:i + 1])[0]
    ss = ((y - pred) ** 2).sum(); st = ((y - y.mean()) ** 2).sum()
    return 1 - ss / st, pred

def perm_p_loo(X, y, obs, n=2000, alpha=1.0):
    cnt = 0
    for _ in range(n):
        yp = rng.permutation(y)
        r2, _ = loo_r2(X, yp, alpha)
        cnt += (r2 >= obs)
    return (cnt + 1) / (n + 1)

# 5a. univariate screen
uni = []
for f in FEATS:
    x = sdf[f].values
    rho, p = stats.spearmanr(x, H)
    r, pr = stats.pearsonr(x, H)
    r2, _ = loo_r2(x[:, None], H)
    uni.append({"feature": f, "spearman": rho, "p_spearman": p, "pearson": r,
                "loo_r2": r2})
uni = pd.DataFrame(uni).sort_values("loo_r2", ascending=False)
def bh_fdr(p):
    p = np.asarray(p, float); n = len(p)
    order = np.argsort(p); q = np.empty(n)
    prev = 1.0
    for rank, i in list(enumerate(order, 1))[::-1]:
        prev = min(prev, p[i] * n / rank); q[i] = prev
    return q
uni["q_fdr"] = bh_fdr(uni["p_spearman"].values)
uni.to_csv(OUT / "univariate.csv", index=False)
print("\n[5a] top univariate (by LOO R^2):")
print(uni.head(12).to_string(index=False))

# 5b. the C question: unique variance of rog_L12 vs the rest
rog = sdf["rog_L12"].values
others = [f for f in FEATS if f != "rog_L12"]
r2_rog, _ = loo_r2(rog[:, None], H)
p_rog = perm_p_loo(rog[:, None], H, r2_rog, 2000)
r2_others, _ = loo_r2(sdf[others].values, H, alpha=10.0)
r2_all, _ = loo_r2(sdf[FEATS].values, H, alpha=10.0)
uniq_rog = r2_all - r2_others
# reverse: does rog subsume the best competitor?
best_comp = uni.loc[uni.feature != "rog_L12", "feature"].iloc[0]
xc = sdf[best_comp].values
r2_comp, _ = loo_r2(xc[:, None], H)
r2_pair, _ = loo_r2(np.c_[rog, xc], H)
# partial spearman rog|comp and comp|rog
def partial_spearman(x, y, z):
    rx = x - LinearRegression().fit(z, x).predict(z)
    ry = y - LinearRegression().fit(z, y).predict(z)
    return stats.spearmanr(rx, ry)
ps_rog = partial_spearman(rog, H, np.c_[xc])
ps_comp = partial_spearman(xc, H, np.c_[rog])
# smoothness-residualized replication (the original "residual hardness" target)
sm = sdf["vel_mean_L0"].values[:, None]
Hres = H - LinearRegression().fit(sm, H).predict(sm)
rho_res, p_res = stats.spearmanr(rog, Hres)
# language-residualized
Ld = pd.get_dummies(sdf["language"]).values.astype(float)
Hlang = H - LinearRegression().fit(Ld, H).predict(Ld)
rho_lang, p_lang = stats.spearmanr(rog, Hlang)

print(f"\n[5b] rog_L12 alone LOO R^2={r2_rog:+.3f} (perm p={p_rog:.4f})")
print(f"     battery w/o rog={r2_others:+.3f}  full battery={r2_all:+.3f}  unique(rog)={uniq_rog:+.3f}")
print(f"     best competitor={best_comp}: alone={r2_comp:+.3f}, rog+comp={r2_pair:+.3f}")
print(f"     partial spearman rog|comp: rho={ps_rog[0]:+.3f} p={ps_rog[1]:.4f}")
print(f"     partial spearman comp|rog: rho={ps_comp[0]:+.3f} p={ps_comp[1]:.4f}")
print(f"     rog vs smoothness-residualized hardness: rho={rho_res:+.3f} p={p_res:.4f}")
print(f"     rog vs language-residualized hardness:   rho={rho_lang:+.3f} p={p_lang:.4f}")

# 5c. factor structure: PCA of the (z-scored) battery
X = sdf[FEATS].values
Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
U_, sv_, Vt_ = np.linalg.svd(Xs, full_matrices=False)
pcs = U_ * sv_
pc_rows = []
for k in range(min(6, pcs.shape[1])):
    rho, p = stats.spearmanr(pcs[:, k], H)
    load = pd.Series(Vt_[k], index=FEATS)
    top = load.abs().sort_values(ascending=False).head(5)
    pc_rows.append({"pc": k + 1, "var_frac": float(sv_[k] ** 2 / (sv_ ** 2).sum()),
                    "spearman_hardness": rho, "p": p,
                    "rog_L12_loading": float(load["rog_L12"]),
                    "top_loadings": "; ".join(f"{i}:{load[i]:+.2f}" for i in top.index)})
pcdf = pd.DataFrame(pc_rows)
pcdf.to_csv(OUT / "pca_factors.csv", index=False)
print("\n[5c] PCA factors vs hardness:")
print(pcdf.to_string(index=False))

# 5d. exhaustive pairs: best 2-feature model (does any pair beat rog-containing pairs?)
pair_rows = []
top_feats = uni.head(10)["feature"].tolist()
for i in range(len(top_feats)):
    for j in range(i + 1, len(top_feats)):
        r2p, _ = loo_r2(sdf[[top_feats[i], top_feats[j]]].values, H)
        pair_rows.append({"f1": top_feats[i], "f2": top_feats[j], "loo_r2": r2p})
pairdf = pd.DataFrame(pair_rows).sort_values("loo_r2", ascending=False)
pairdf.to_csv(OUT / "pairs.csv", index=False)
print("\n[5d] top pairs:"); print(pairdf.head(8).to_string(index=False))

# ─── 6. summary ───────────────────────────────────────────────────────────────
summary = {
    "n_systems": int(N), "min_utts": MIN_UTTS, "detector_eer_full_split": float(eer),
    "rog_L12": {"loo_r2": float(r2_rog), "perm_p": float(p_rog),
                "spearman": float(uni.set_index('feature').loc['rog_L12','spearman']),
                "rho_smoothness_resid": float(rho_res), "p": float(p_res),
                "rho_language_resid": float(rho_lang)},
    "battery_without_rog_loo_r2": float(r2_others),
    "full_battery_loo_r2": float(r2_all), "unique_rog": float(uniq_rog),
    "best_competitor": {"name": best_comp, "alone": float(r2_comp), "with_rog": float(r2_pair),
                        "partial_rho_rog_given_comp": float(ps_rog[0]), "p": float(ps_rog[1]),
                        "partial_rho_comp_given_rog": float(ps_comp[0]), "p2": float(ps_comp[1])},
}
(OUT / "i2_stats.json").write_text(json.dumps(summary, indent=2))
print(f"\n[I2] done -> {OUT}")
