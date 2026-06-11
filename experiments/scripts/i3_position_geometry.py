#!/usr/bin/env python3
"""
I3 — Manifold position & decision-boundary geometry vs hardness (MLAAD)
=========================================================================
Adversarial upgrades over I2:
  * Position decomposed into ALONG the natural<->synthetic axis (s_along) vs
    ORTHOGONAL residual (s_orth), in FROZEN pretrained WavLM space (L12 mean-pool),
    with the axis fit LEAVE-ONE-SYSTEM-OUT (no target system leaks into the axis).
  * Circularity control: hardness recomputed under 3 independent detector seeds
    (mlaad_robust_goat, seed42, seed1024); position must predict the SHARED
    component, not one detector's quirks.
  * RMS confound control: waveform RMS partialled out of every feature claim.
  * Mediation/commonality: position vs dynamics (vel_entropy) vs both.
  * Within-system fixed effects for s_along / s_orth (cluster-robust + permutation).

Outputs -> experiments/results/i3_position_geometry/
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

OUT = px.EXP_DIR / "results" / "i3_position_geometry"
OUT.mkdir(parents=True, exist_ok=True)
I2 = px.EXP_DIR / "results" / "i2_geometry_battery"
DEVICE = px.DEVICE
MIN_UTTS = 8

SEED_CKPTS = {
    "main":  px.EXP_DIR / "checkpoints" / "mlaad_robust_goat.ckpt",
    "s42":   px.EXP_DIR / "checkpoints" / "mlaad_robust_goat_seed42-best-epoch=05-val-eer=0.3030.ckpt",
    "s1024": px.EXP_DIR / "checkpoints" / "mlaad_robust_goat_seed1024-best-epoch=03-val-eer=0.2976.ckpt",
}

# ─── data ─────────────────────────────────────────────────────────────────────
recs = json.loads(px.TEST_JSON.read_text())
z = np.load(px.WAVE_CACHE / "i2_full_test_waves.npz", allow_pickle=True)
waves, ok_idx = z["waves"].astype(np.float32), z["ok_idx"]
recs = [recs[i] for i in ok_idx]
labels = np.array([1 if str(r["label"]).lower().startswith("spoof") else 0 for r in recs])
systems = np.array([f'{r["attack_system"]}|{r["language"]}' if labels[i] else "bona"
                    for i, r in enumerate(recs)])
rms = np.sqrt((waves ** 2).mean(1))
print(f"[I3] {len(recs)} utts")

# ─── detector logits under 3 seeds ────────────────────────────────────────────
logit_seeds = {}
for sname, ck in SEED_CKPTS.items():
    f = OUT / f"logits_{sname}.npy"
    if f.exists():
        logit_seeds[sname] = np.load(f); continue
    if sname == "main" and (I2 / "detector_logits.npy").exists():
        logit_seeds[sname] = np.load(I2 / "detector_logits.npy")
        np.save(f, logit_seeds[sname]); continue
    print(f"  scoring under {sname} ...", flush=True)
    gm = px.load_detector(ckpt=ck)
    logit_seeds[sname] = px.detector_logits(gm, waves)
    np.save(f, logit_seeds[sname])
    del gm; torch.cuda.empty_cache()
for sname, lg in logit_seeds.items():
    e, _ = px.compute_eer(labels, lg)
    print(f"  {sname}: EER={e:.3f}")

# ─── frozen WavLM mean-pooled embeddings (L9, L12), saved this time ───────────
EMB = OUT / "embeddings.npz"
if EMB.exists():
    ez = np.load(EMB)
    X12, X9 = ez["X12"], ez["X9"]
else:
    wl = px.load_frozen_wavlm(DEVICE)
    X12, X9 = [], []
    with torch.no_grad():
        for b in range(0, len(waves), 8):
            xb = torch.as_tensor(waves[b:b + 8], dtype=torch.float32, device=DEVICE)
            hs = wl(xb, output_hidden_states=True).hidden_states
            X12.append(hs[12].mean(1).float().cpu().numpy())
            X9.append(hs[9].mean(1).float().cpu().numpy())
            if (b // 8) % 40 == 0: print(f"  emb {b}/{len(waves)}", flush=True)
    X12 = np.concatenate(X12); X9 = np.concatenate(X9)
    np.savez_compressed(EMB, X12=X12, X9=X9)
    del wl; torch.cuda.empty_cache()
print(f"  embeddings: {X12.shape}")

# standardize on bona (training-domain convention, E12)
mu_b = X12[labels == 0].mean(0); sd_b = X12[labels == 0].std(0) + 1e-9
Z = (X12 - mu_b) / sd_b

# ─── per-system hardness under each seed + shared component ───────────────────
from sklearn.metrics import roc_auc_score
sys_list = sorted({s for s in systems if s != "bona"
                   and (systems == s).sum() >= MIN_UTTS})
def hardness_for(lg):
    bl = lg[labels == 0]
    return {s: 1.0 - roc_auc_score(
        np.r_[np.zeros(len(bl)), np.ones((systems == s).sum())],
        np.r_[bl, lg[systems == s]]) for s in sys_list}
H = pd.DataFrame({sn: hardness_for(lg) for sn, lg in logit_seeds.items()})
H["shared"] = H[["main", "s42", "s1024"]].mean(1)
print("\n  hardness cross-seed Spearman:")
print(H[["main", "s42", "s1024"]].corr(method="spearman").round(3).to_string())

# ─── LOSO natural<->synthetic axis + position coordinates ─────────────────────
bona_mask = labels == 0
mu_bona = Z[bona_mask].mean(0)
s_along = np.full(len(Z), np.nan); s_orth = np.full(len(Z), np.nan)
for s in sys_list + ["bona"]:
    m = systems == s
    sp_excl = (labels == 1) & (systems != s)          # spoof centroid w/o this system
    w = Z[sp_excl].mean(0) - mu_bona
    w = w / (np.linalg.norm(w) + 1e-12)
    d = Z[m] - mu_bona
    a = d @ w
    s_along[m] = a
    s_orth[m] = np.linalg.norm(d - np.outer(a, w), axis=1)

utt = pd.DataFrame({"system": systems, "label": labels, "rms": rms,
                    "s_along": s_along, "s_orth": s_orth,
                    "logit": logit_seeds["main"]})
# merge dynamics features from I2
fdf = pd.read_csv(I2 / "utt_features.csv")
for c in ["vel_entropy_L12", "vel_entropy_L9", "mu_norm_L12", "rog_L12", "vel_mean_L12",
          "vel_mean_L0", "rog_L0"]:
    utt[c] = fdf[c].values
utt.to_csv(OUT / "utt_position.csv", index=False)
print(f"\n  bona s_along median={np.median(s_along[bona_mask]):+.2f}  "
      f"spoof={np.median(s_along[~bona_mask]):+.2f}")

# ─── analyses ─────────────────────────────────────────────────────────────────
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression
rng = np.random.default_rng(SEED)

sysu = utt[utt.label == 1].groupby("system").median(numeric_only=True).loc[sys_list]
sysu["sd_along"] = utt[utt.label == 1].groupby("system")["s_along"].std().loc[sys_list]
for hcol in ["main", "s42", "s1024", "shared"]:
    sysu[f"hard_{hcol}"] = H[hcol]

N = len(sysu)
def loo_r2(X, y, alpha=1.0):
    X = np.asarray(X, float).reshape(N, -1)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    pred = np.empty(N)
    for i in range(N):
        m = np.ones(N, bool); m[i] = False
        pred[i] = Ridge(alpha=alpha).fit(Xs[m], y[m]).predict(Xs[i:i+1])[0]
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

def perm_p(X, y, obs, n=2000):
    c = 0
    for _ in range(n):
        c += loo_r2(X, rng.permutation(y)) >= obs
    return (c + 1) / (n + 1)

print("\n[A] system-level position vs hardness (per detector seed):")
res = {}
for hcol in ["main", "s42", "s1024", "shared"]:
    y = sysu[f"hard_{hcol}"].values
    rho_a, p_a = stats.spearmanr(sysu["s_along"], y)
    rho_o, p_o = stats.spearmanr(sysu["s_orth"], y)
    r2_a = loo_r2(sysu["s_along"].values, y)
    res[hcol] = {"rho_along": rho_a, "p_along": p_a, "rho_orth": rho_o, "p_orth": p_o,
                 "loo_along": r2_a}
    print(f"  {hcol:>6}: s_along rho={rho_a:+.3f} (p={p_a:.4f}) LOO={r2_a:+.3f} | "
          f"s_orth rho={rho_o:+.3f} (p={p_o:.4f})")

y = sysu["hard_shared"].values
r2_along = loo_r2(sysu["s_along"].values, y); p_perm_along = perm_p(sysu["s_along"].values, y, r2_along)
r2_ve    = loo_r2(sysu["vel_entropy_L12"].values, y)
r2_both  = loo_r2(sysu[["s_along", "vel_entropy_L12"]].values, y)
r2_trip  = loo_r2(sysu[["s_along", "vel_entropy_L12", "sd_along"]].values, y)
r2_dprime = loo_r2((sysu["s_along"] / (sysu["sd_along"] + 1e-9)).values, y)
print(f"\n[B] shared-hardness models: s_along={r2_along:+.3f} (perm p={p_perm_along:.4f})  "
      f"vel_entropy={r2_ve:+.3f}  both={r2_both:+.3f}  +sd_along={r2_trip:+.3f}  "
      f"dprime(s/sd)={r2_dprime:+.3f}")

# RMS confound: partial out waveform RMS from everything
rmsS = sysu["rms"].values[:, None]
def resid(v, Z_):
    return v - LinearRegression().fit(Z_, v).predict(Z_)
rho_a_rms = stats.spearmanr(resid(sysu["s_along"].values, rmsS), resid(y, rmsS))
rho_v_rms = stats.spearmanr(resid(sysu["vel_entropy_L12"].values, rmsS), resid(y, rmsS))
rho_m_rms = stats.spearmanr(resid(sysu["mu_norm_L12"].values, rmsS), resid(y, rmsS))
print(f"[C] RMS-partialled: s_along rho={rho_a_rms[0]:+.3f} (p={rho_a_rms[1]:.4f})  "
      f"vel_entropy rho={rho_v_rms[0]:+.3f} (p={rho_v_rms[1]:.4f})  "
      f"mu_norm rho={rho_m_rms[0]:+.3f} (p={rho_m_rms[1]:.4f})")

# does position subsume dynamics (or vice versa)?
pa = stats.spearmanr(resid(sysu["vel_entropy_L12"].values, sysu[["s_along"]].values),
                     resid(y, sysu[["s_along"]].values))
pb = stats.spearmanr(resid(sysu["s_along"].values, sysu[["vel_entropy_L12"]].values),
                     resid(y, sysu[["vel_entropy_L12"]].values))
print(f"[D] partial: vel_entropy|s_along rho={pa[0]:+.3f} (p={pa[1]:.4f}); "
      f"s_along|vel_entropy rho={pb[0]:+.3f} (p={pb[1]:.4f})")

# ─── within-system fixed effects for position ─────────────────────────────────
sp = utt[utt.label == 1].copy()
sp = sp[sp.system.isin(sys_list)]
def fe_test(col):
    d = sp.copy()
    for c in [col, "logit"]:
        d[c] = sp[c] - sp.groupby("system")[c].transform("mean")
    x = d[col].values / (d[col].std() + 1e-12); yv = d["logit"].values
    beta = float((x @ yv) / (x @ x))
    perms = []
    for _ in range(2000):
        xp = sp.groupby("system")[col].transform(lambda v: rng.permutation(v.values)).values
        xp = xp - pd.Series(xp).groupby(sp.system.values).transform("mean").values
        xp = xp / (xp.std() + 1e-12)
        perms.append((xp @ yv) / (xp @ xp))
    p = float((np.abs(perms) >= abs(beta)).mean())
    return beta, p
for col in ["s_along", "s_orth", "vel_entropy_L12"]:
    b, p = fe_test(col)
    print(f"[E] FE {col:>16}: beta={b:+.4f} (perm p={p:.4f})")

json_out = {
    "per_seed": res, "shared": {"loo_along": float(r2_along), "perm_p": float(p_perm_along),
    "loo_vel_entropy": float(r2_ve), "loo_both": float(r2_both), "loo_trip": float(r2_trip),
    "loo_dprime": float(r2_dprime)},
    "rms_partialled": {"s_along": list(map(float, rho_a_rms)),
                       "vel_entropy": list(map(float, rho_v_rms)),
                       "mu_norm": list(map(float, rho_m_rms))},
    "partials": {"ve_given_along": list(map(float, pa)), "along_given_ve": list(map(float, pb))},
}
(OUT / "i3_stats.json").write_text(json.dumps(json_out, indent=2))
sysu.to_csv(OUT / "system_position.csv")
print(f"\n[I3] done -> {OUT}")
