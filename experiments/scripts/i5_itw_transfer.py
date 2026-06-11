#!/usr/bin/env python3
"""
I5 — In-The-Wild: do the surviving hardness factors transfer, and does T survive?
==================================================================================
Questions (E10–E13 context):
  Q1  Does T (vel_entropy) survive on ITW?  Per-spoof-speaker hardness (1−AUC vs
      the ITW bona pool) vs median vel_entropy@L12/L9; plus within-speaker FE.
  Q2  Does the MLAAD-fit natural<->synthetic axis transfer?  Project ITW utts onto
      the FROZEN-space axis w fit on MLAAD (I3): (a) E12 replication — ITW bona
      slides toward spoof along w; (b) does s_along predict ITW spoof hardness?
  Q3  Why factors do/do not hold: variance decomposition — how much of ITW position
      variance is the channel direction (bona shift) vs the synthesis direction.

Uses the same 1500+1500 ITW subset as I6 (logits read from i6_logits.npz).
Outputs -> experiments/results/i5_itw_transfer/
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

OUT = px.EXP_DIR / "results" / "i5_itw_transfer"
OUT.mkdir(parents=True, exist_ok=True)
I3DIR = px.EXP_DIR / "results" / "i3_position_geometry"
I6DIR = px.EXP_DIR / "results" / "i6_testtime_boost"
DEVICE = px.DEVICE

# ─── same ITW subset as I6 ────────────────────────────────────────────────────
itw = px.load_itw_waves(n_per_class=3000)
rngi = np.random.default_rng(SEED)
sel_b = rngi.choice(np.where(itw["labels"] == 0)[0], 1500, replace=False)
sel_s = rngi.choice(np.where(itw["labels"] == 1)[0], 1500, replace=False)
isel = np.sort(np.r_[sel_b, sel_s])
waves, labels = itw["waves"][isel], itw["labels"][isel]
speakers = itw["speakers"][isel]
rms = np.sqrt((waves ** 2).mean(1))
lz = np.load(I6DIR / "i6_logits.npz")
logits = lz["itw__main__baseline"]
assert len(logits) == len(waves)
from sklearn.metrics import roc_auc_score, roc_curve
def eer_of(y, s):
    fpr, tpr, _ = roc_curve(y, s, pos_label=1); fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr))); return float((fpr[i]+fnr[i])/2)
print(f"[I5] ITW {len(waves)} utts, EER={eer_of(labels, logits):.3f}")

# ─── frozen WavLM features ────────────────────────────────────────────────────
FEAT = OUT / "features.npz"
if FEAT.exists():
    fz = np.load(FEAT)
    X12, ve12, ve9, rog12, vmean12, vmean0 = (fz[k] for k in
        ["X12", "ve12", "ve9", "rog12", "vmean12", "vmean0"])
else:
    wl = px.load_frozen_wavlm(DEVICE)
    X12l, ve12, ve9, rog12, vmean12, vmean0 = [], [], [], [], [], []
    def vent(F):
        vn = np.linalg.norm(np.diff(F, axis=0), axis=1)
        nb = max(5, min(20, len(vn) // 4))
        h, _ = np.histogram(vn, bins=nb); h = h.astype(float) + 1e-8; h /= h.sum()
        return float(-(h * np.log(h)).sum())
    with torch.no_grad():
        for b in range(0, len(waves), 8):
            xb = torch.as_tensor(waves[b:b+8], dtype=torch.float32, device=DEVICE)
            hs = wl(xb, output_hidden_states=True).hidden_states
            X12l.append(hs[12].mean(1).float().cpu().numpy())
            for k in range(xb.shape[0]):
                F12 = hs[12][k].float().cpu().numpy(); F9 = hs[9][k].float().cpu().numpy()
                F0 = hs[0][k].float().cpu().numpy()
                ve12.append(vent(F12)); ve9.append(vent(F9))
                S = F12 - F12.mean(0)
                rog12.append(float(np.sqrt((S**2).sum(1).mean())))
                vmean12.append(float(np.linalg.norm(np.diff(F12, axis=0), axis=1).mean()))
                vmean0.append(float(np.linalg.norm(np.diff(F0, axis=0), axis=1).mean()))
            if (b // 8) % 40 == 0: print(f"  feats {b}/{len(waves)}", flush=True)
    X12 = np.concatenate(X12l)
    ve12, ve9 = np.array(ve12), np.array(ve9)
    rog12, vmean12, vmean0 = np.array(rog12), np.array(vmean12), np.array(vmean0)
    np.savez_compressed(FEAT, X12=X12, ve12=ve12, ve9=ve9, rog12=rog12,
                        vmean12=vmean12, vmean0=vmean0)
    del wl; torch.cuda.empty_cache()

# ─── MLAAD-fit axis transfer ──────────────────────────────────────────────────
mz = np.load(I3DIR / "embeddings.npz")
import json as _json
mrecs = _json.loads(px.TEST_JSON.read_text())
mok = np.load(px.WAVE_CACHE / "i2_full_test_waves.npz", allow_pickle=True)["ok_idx"]
mrecs = [mrecs[i] for i in mok]
mlab = np.array([1 if str(r["label"]).lower().startswith("spoof") else 0 for r in mrecs])
MX = mz["X12"]
mu_b = MX[mlab == 0].mean(0); sd_b = MX[mlab == 0].std(0) + 1e-9   # MLAAD-bona frame
Zm = (MX - mu_b) / sd_b
w = Zm[mlab == 1].mean(0) - Zm[mlab == 0].mean(0)
w /= np.linalg.norm(w)
mu_bona_m = Zm[mlab == 0].mean(0)

Zi = (X12 - mu_b) / sd_b                       # ITW in the MLAAD frame
d = Zi - mu_bona_m
s_along = d @ w
s_orth = np.linalg.norm(d - np.outer(s_along, w), axis=1)

m_along_bona  = float(np.median((Zm[mlab == 0] - mu_bona_m) @ w))
m_along_spoof = float(np.median((Zm[mlab == 1] - mu_bona_m) @ w))
gap = m_along_spoof - m_along_bona
i_b = float(np.median(s_along[labels == 0])); i_s = float(np.median(s_along[labels == 1]))
print(f"\n[Q2a] E12 replication, s_along medians (MLAAD bona={m_along_bona:.2f}, "
      f"spoof={m_along_spoof:.2f}, gap={gap:.2f}):")
print(f"      ITW bona={i_b:.2f} ({(i_b-m_along_bona)/gap:+.2f} gaps slid)  "
      f"ITW spoof={i_s:.2f} ({(i_s-m_along_bona)/gap:+.2f})")

from scipy import stats
rho_fp = stats.spearmanr(s_along[labels == 0], logits[labels == 0])
print(f"[Q2a'] within ITW-bona: s_along vs spoof-logit rho={rho_fp[0]:+.3f} (p={rho_fp[1]:.2e})")

# ─── per-speaker spoof hardness ───────────────────────────────────────────────
utt = pd.DataFrame({"speaker": speakers, "label": labels, "logit": logits, "rms": rms,
                    "s_along": s_along, "s_orth": s_orth, "ve12": ve12, "ve9": ve9,
                    "rog12": rog12, "vmean12": vmean12, "vmean0": vmean0})
utt.to_csv(OUT / "utt_table.csv", index=False)
bona_l = logits[labels == 0]
spk_units = [s for s, g in utt[utt.label == 1].groupby("speaker") if len(g) >= 10]
rows = []
for s in spk_units:
    g = utt[(utt.label == 1) & (utt.speaker == s)]
    yv = np.r_[np.zeros(len(bona_l)), np.ones(len(g))]
    sc = np.r_[bona_l, g.logit.values]
    r = {"speaker": s, "n": len(g), "hard": 1 - roc_auc_score(yv, sc)}
    for c in ["s_along", "s_orth", "ve12", "ve9", "rog12", "vmean12", "vmean0", "rms"]:
        r[c] = float(g[c].median())
    rows.append(r)
spk = pd.DataFrame(rows)
spk.to_csv(OUT / "speaker_table.csv", index=False)
print(f"\n[Q1] {len(spk)} spoof speakers (>=10 utts), hardness range "
      f"[{spk.hard.min():.3f}, {spk.hard.max():.3f}]")
res = {}
for c in ["ve12", "ve9", "s_along", "s_orth", "rog12", "vmean12", "vmean0"]:
    rho, p = stats.spearmanr(spk[c], spk["hard"])
    res[c] = {"rho": float(rho), "p": float(p)}
    print(f"  {c:>8}: rho={rho:+.3f} (p={p:.4f})")

# RMS-partialled (channel/level confound)
from sklearn.linear_model import LinearRegression
def resid(v, Zc): return v - LinearRegression().fit(Zc, v).predict(Zc)
Zc = spk[["rms"]].values
for c in ["ve12", "s_along"]:
    rho, p = stats.spearmanr(resid(spk[c].values, Zc), resid(spk["hard"].values, Zc))
    res[f"{c}_rms_partial"] = {"rho": float(rho), "p": float(p)}
    print(f"  {c:>8} | rms: rho={rho:+.3f} (p={p:.4f})")

# within-speaker FE on spoof logit
rng2 = np.random.default_rng(SEED)
spd = utt[(utt.label == 1) & (utt.speaker.isin(spk_units))].copy()
def fe(col):
    dd = spd.copy()
    for c in [col, "logit"]:
        dd[c] = spd[c] - spd.groupby("speaker")[c].transform("mean")
    x = dd[col].values / (dd[col].std() + 1e-12); yv = dd["logit"].values
    beta = float((x @ yv) / (x @ x))
    perms = []
    for _ in range(2000):
        xp = spd.groupby("speaker")[col].transform(lambda v: rng2.permutation(v.values)).values
        xp = xp - pd.Series(xp).groupby(spd.speaker.values).transform("mean").values
        xp /= (xp.std() + 1e-12)
        perms.append((xp @ yv) / (xp @ xp))
    return beta, float((np.abs(perms) >= abs(beta)).mean())
print("\n[Q1-FE] within-speaker fixed effects (spoof logit):")
fe_res = {}
for c in ["ve12", "s_along", "s_orth", "rog12", "vmean12"]:
    b, p = fe(c)
    fe_res[c] = {"beta": b, "p_perm": p}
    print(f"  {c:>8}: beta={b:+.4f} (perm p={p:.4f})")

# Q3: variance decomposition — channel vs synthesis direction
v_b = float(np.var(s_along[labels == 0])); v_s = float(np.var(s_along[labels == 1]))
v_bm = float(np.var((Zm[mlab == 0] - mu_bona_m) @ w))
print(f"\n[Q3] var(s_along): MLAAD-bona={v_bm:.2f}  ITW-bona={v_b:.2f}  ITW-spoof={v_s:.2f}")
(OUT / "i5_stats.json").write_text(json.dumps(
    {"e12_replication": {"mlaad_gap": gap, "itw_bona_slide_gaps": (i_b - m_along_bona) / gap,
                         "itw_spoof_slide_gaps": (i_s - m_along_bona) / gap,
                         "bona_along_vs_logit": list(map(float, rho_fp))},
     "speaker_level": res, "fixed_effects": fe_res,
     "var_decomp": {"mlaad_bona": v_bm, "itw_bona": v_b, "itw_spoof": v_s}}, indent=2))
print(f"\n[I5] done -> {OUT}")
