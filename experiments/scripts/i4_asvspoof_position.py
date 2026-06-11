#!/usr/bin/env python3
"""
I4 — Cross-dataset replication on ASVspoof 2019 LA (A07–A19)
=============================================================
Do the I2/I3 hardness factors (velocity entropy, manifold position along the
natural<->synthetic axis) survive on ASVspoof — the dataset where C famously
flipped sign? Uses the same 800+800 eval subset and the 3 robust_GOAT seeds'
baseline logits stored by I1 (no new detector compute), plus frozen pretrained
WavLM features.

Unit of analysis: 13 attacks (low power — exact p reported) AND within-attack
fixed effects over 800 spoof utterances (high power, confound-free).
Also quantifies factor spread (range compression) to explain failures.

Outputs -> experiments/results/i4_asvspoof_position/
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

BASE = px.BASE
OUT = px.EXP_DIR / "results" / "i4_asvspoof_position"
OUT.mkdir(parents=True, exist_ok=True)
I1DIR = px.EXP_DIR / "results" / "i1_geometry_causal_decomp"
DEVICE = px.DEVICE
TARGET_LEN = 48_000
EVAL_ATTACKS = [f"A{i:02d}" for i in range(7, 20)]

# ─── reproduce I1's deterministic selection to recover attack ids ─────────────
def _crop(wav):
    if wav.ndim > 1: wav = wav.mean(0)
    if len(wav) < TARGET_LEN: wav = wav.repeat(-(-TARGET_LEN // len(wav)))
    mid = (len(wav) - TARGET_LEN) // 2
    return wav[mid:mid + TARGET_LEN]

from datasets import load_dataset, Audio as HFAudio
ds = load_dataset("Bisher/as_vspoof_2019_la", cache_dir=str(BASE/"data"/"asvspoof_2019_la"),
                  trust_remote_code=True)["test"]
sysids = ds["system_id"]
rng = np.random.default_rng(SEED)
spoof_idx = [i for i, s in enumerate(sysids) if s in EVAL_ATTACKS]
bona_idx  = [i for i, s in enumerate(sysids) if s == "-"]
sel = sorted(rng.choice(spoof_idx, 800, replace=False).tolist() +
             rng.choice(bona_idx, 800, replace=False).tolist())
labels = np.array([0 if sysids[i] == "-" else 1 for i in sel])
attacks = np.array([sysids[i] for i in sel])
lz = np.load(I1DIR / "i1_logits.npz")
assert np.array_equal(lz["labels"], labels), "I1 selection mismatch"
logit_seeds = {s: lz[f"{s}__baseline"] for s in ["s1", "s3", "s7"]}
print(f"[I4] {len(sel)} utts, attacks: {sorted(set(attacks)) }")

# ─── frozen WavLM features ────────────────────────────────────────────────────
FEAT = OUT / "features.npz"
if FEAT.exists():
    fz = np.load(FEAT)
    X12, ve12, ve9, rog12, vmean12, rms = (fz[k] for k in
        ["X12", "ve12", "ve9", "rog12", "vmean12", "rms"])
else:
    sub = ds.select(sel).cast_column("audio", HFAudio(sampling_rate=16_000))
    waves = np.stack([_crop(torch.tensor(sub[i]["audio"]["array"], dtype=torch.float32)).numpy()
                      for i in range(len(sub))])
    rms = np.sqrt((waves ** 2).mean(1))
    wl = px.load_frozen_wavlm(DEVICE)
    X12, ve12, ve9, rog12, vmean12 = [], [], [], [], []
    def vent(F):
        vn = np.linalg.norm(np.diff(F, axis=0), axis=1)
        nb = max(5, min(20, len(vn) // 4))
        h, _ = np.histogram(vn, bins=nb); h = h.astype(float) + 1e-8; h /= h.sum()
        return float(-(h * np.log(h)).sum())
    with torch.no_grad():
        for b in range(0, len(waves), 8):
            xb = torch.as_tensor(waves[b:b+8], dtype=torch.float32, device=DEVICE)
            hs = wl(xb, output_hidden_states=True).hidden_states
            X12.append(hs[12].mean(1).float().cpu().numpy())
            for k in range(xb.shape[0]):
                F12 = hs[12][k].float().cpu().numpy(); F9 = hs[9][k].float().cpu().numpy()
                ve12.append(vent(F12)); ve9.append(vent(F9))
                S = F12 - F12.mean(0)
                rog12.append(float(np.sqrt((S**2).sum(1).mean())))
                vmean12.append(float(np.linalg.norm(np.diff(F12, axis=0), axis=1).mean()))
            if (b // 8) % 40 == 0: print(f"  feats {b}/{len(waves)}", flush=True)
    X12 = np.concatenate(X12)
    ve12, ve9 = np.array(ve12), np.array(ve9)
    rog12, vmean12 = np.array(rog12), np.array(vmean12)
    np.savez_compressed(FEAT, X12=X12, ve12=ve12, ve9=ve9, rog12=rog12,
                        vmean12=vmean12, rms=rms)
    del wl; torch.cuda.empty_cache()

# ─── position coordinates (LOAO axis) ─────────────────────────────────────────
mu_b = X12[labels == 0].mean(0); sd_b = X12[labels == 0].std(0) + 1e-9
Z = (X12 - mu_b) / sd_b
mu_bona = Z[labels == 0].mean(0)
s_along = np.full(len(Z), np.nan); s_orth = np.full(len(Z), np.nan)
units = sorted(set(attacks) - {"-"})
for a in units + ["-"]:
    m = attacks == a
    sp_excl = (labels == 1) & (attacks != a)
    w = Z[sp_excl].mean(0) - mu_bona; w /= (np.linalg.norm(w) + 1e-12)
    d = Z[m] - mu_bona
    al = d @ w
    s_along[m] = al
    s_orth[m] = np.linalg.norm(d - np.outer(al, w), axis=1)

# ─── per-attack hardness (mean over 3 seeds) ──────────────────────────────────
from sklearn.metrics import roc_auc_score
from scipy import stats
def hard(lg, a):
    bl = lg[labels == 0]; sl = lg[attacks == a]
    return 1 - roc_auc_score(np.r_[np.zeros(len(bl)), np.ones(len(sl))], np.r_[bl, sl])
adf = pd.DataFrame({"attack": units})
for sname, lg in logit_seeds.items():
    adf[f"hard_{sname}"] = [hard(lg, a) for a in units]
adf["hard"] = adf[[f"hard_{s}" for s in logit_seeds]].mean(1)
for col, v in [("s_along", s_along), ("s_orth", s_orth), ("ve12", ve12), ("ve9", ve9),
               ("rog12", rog12), ("vmean12", vmean12), ("rms", rms)]:
    adf[col] = [np.median(v[attacks == a]) for a in units]
adf.to_csv(OUT / "attack_table.csv", index=False)

print("\n[A] per-attack (N=13) Spearman vs shared hardness:")
res = {}
for col in ["s_along", "s_orth", "ve12", "ve9", "rog12", "vmean12"]:
    rho, p = stats.spearmanr(adf[col], adf["hard"])
    res[col] = {"rho": float(rho), "p": float(p)}
    print(f"  {col:>9}: rho={rho:+.3f} (p={p:.4f})")
print(adf[["attack", "hard", "s_along", "ve12", "rog12"]].sort_values("hard").to_string(index=False))

# spread comparison vs MLAAD (range-compression explanation)
i3sys = pd.read_csv(px.EXP_DIR / "results" / "i3_position_geometry" / "system_position.csv") \
        if (px.EXP_DIR / "results" / "i3_position_geometry" / "system_position.csv").exists() else None
spread = {"asvspoof": {c: float(adf[c].std()) for c in ["s_along", "ve12", "rog12"]}}
if i3sys is not None:
    spread["mlaad"] = {"s_along": float(i3sys["s_along"].std()),
                       "ve12": float(i3sys["vel_entropy_L12"].std()),
                       "rog12": float(i3sys["rog_L12"].std())}
print(f"\n[B] factor spread (SD across systems): {json.dumps(spread, indent=2)}")

# ─── within-attack fixed effects (logit averaged over seeds) ──────────────────
mlogit = np.mean([logit_seeds[s] for s in logit_seeds], 0)
sp_mask = labels == 1
spdf = pd.DataFrame({"attack": attacks[sp_mask], "logit": mlogit[sp_mask],
                     "s_along": s_along[sp_mask], "s_orth": s_orth[sp_mask],
                     "ve12": ve12[sp_mask], "rog12": rog12[sp_mask],
                     "vmean12": vmean12[sp_mask]})
rng2 = np.random.default_rng(SEED)
def fe(col):
    d = spdf.copy()
    for c in [col, "logit"]:
        d[c] = spdf[c] - spdf.groupby("attack")[c].transform("mean")
    x = d[col].values / (d[col].std() + 1e-12); yv = d["logit"].values
    beta = float((x @ yv) / (x @ x))
    perms = []
    for _ in range(2000):
        xp = spdf.groupby("attack")[col].transform(lambda v: rng2.permutation(v.values)).values
        xp = xp - pd.Series(xp).groupby(spdf.attack.values).transform("mean").values
        xp /= (xp.std() + 1e-12)
        perms.append((xp @ yv) / (xp @ xp))
    return beta, float((np.abs(perms) >= abs(beta)).mean())
print("\n[C] within-attack fixed effects (spoof logit, 3-seed mean):")
fe_res = {}
for col in ["s_along", "s_orth", "ve12", "rog12", "vmean12"]:
    b, p = fe(col)
    fe_res[col] = {"beta": b, "p_perm": p}
    print(f"  {col:>9}: beta={b:+.4f} (perm p={p:.4f})")

(OUT / "i4_stats.json").write_text(json.dumps(
    {"per_attack": res, "spread": spread, "fixed_effects": fe_res}, indent=2))
print(f"\n[I4] done -> {OUT}")
