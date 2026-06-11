#!/usr/bin/env python3
"""
J6 — AASIST fine-tuned on MLAAD: the missing in-domain non-WavLM cell
======================================================================
J5 left one cell open: does the MLAAD position law hold for an IN-DOMAIN
non-WavLM detector? (Zero-shot AASIST mixes domain shift into hardness.)

Recipe: official AASIST weights -> fine-tune on MLAAD train split (10,639 utts)
with the official-style weighted CE ([0.1 bona, 0.9 spoof->... weights per class]),
Adam 1e-4, cosine decay, 8 epochs, select by val EER. Then:
  * score the same 1846-utt MLAAD test subset used everywhere,
  * per-system hardness (>=8 utts, 61 units),
  * test the law: s_along / sd_along / s_orth / vel_entropy (frozen-WavLM geometry
    from I3) -> Spearman + LOO R^2 (position+spread),
  * family agreement vs MLAAD-GAT seeds / robust_goat / pretrained AASIST,
  * corpus-internal LDA-axis fusion gain for the fine-tuned model.

Outputs -> experiments/results/j6_aasist_mlaad/
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import px_common as px

OUT = px.EXP_DIR / "results" / "j6_aasist_mlaad"
OUT.mkdir(parents=True, exist_ok=True)
R = px.EXP_DIR / "results"
DEVICE = px.DEVICE
NB_SAMP = 64600
EPOCHS, BS, LR = 8, 24, 1e-4

sys.path.insert(0, str(px.BASE / "baselines" / "aasist"))
from models.AASIST import Model as AASIST
conf = json.loads((px.BASE / "baselines" / "aasist" / "config" / "AASIST.conf").read_text())

def tile(w):
    w = np.asarray(w, np.float32)
    if len(w) >= NB_SAMP: return w[:NB_SAMP]
    return np.tile(w, -(-NB_SAMP // len(w)))[:NB_SAMP]

# ─── data ─────────────────────────────────────────────────────────────────────
SPL = px.EXP_DIR / "data" / "mlaad_tiny_processed" / "splits"
def load_split(name):
    recs = json.loads((SPL / f"{name}.json").read_text())
    cache = px.WAVE_CACHE / f"j6_{name}_waves.npz"
    if cache.exists():
        z = np.load(cache)
        return z["w"].astype(np.float32), z["y"]
    ws, ys = [], []
    for i, r in enumerate(recs):
        try:
            ws.append(px._fix(np.asarray(torch.load(px.PROC_DIR / r["audio_path"]),
                                         np.float32)).astype(np.float16))
            ys.append(1 if str(r["label"]).lower().startswith("spoof") else 0)
        except Exception:
            pass
        if (i+1) % 2000 == 0: print(f"  {name} {i+1}/{len(recs)}", flush=True)
    w = np.stack(ws); y = np.array(ys)
    np.savez_compressed(cache, w=w, y=y)
    return w.astype(np.float32), y

wtr, ytr = load_split("train")
wva, yva = load_split("val")
print(f"[J6] train {len(ytr)} (spoof={ytr.sum()}), val {len(yva)}")

from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
def eer_of(y, s):
    fpr, tpr, _ = roc_curve(y, s, pos_label=1); fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr))); return float((fpr[i] + fnr[i]) / 2)

@torch.no_grad()
def score(model, waves, bs=32):
    model.eval(); out = []
    for b in range(0, len(waves), bs):
        xb = torch.as_tensor(np.stack([tile(w) for w in waves[b:b+bs]]), device=DEVICE)
        _, o = model(xb)
        out.append(o[:, 1].float().cpu().numpy())
    return -np.concatenate(out)        # orient: higher = spoof (J5 orientation)

# ─── train ────────────────────────────────────────────────────────────────────
CKPT = OUT / "aasist_mlaad.pth"
model = AASIST(conf["model_config"]).to(DEVICE)
model.load_state_dict(torch.load(px.BASE / "baselines" / "aasist" / "models" /
                                 "weights" / "AASIST.pth", map_location=DEVICE,
                                 weights_only=False))
if CKPT.exists():
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=False))
    print("  loaded fine-tuned checkpoint")
else:
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    steps = EPOCHS * (len(ytr) // BS)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=5e-6)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9], device=DEVICE))
    best = 1.0
    rng = np.random.default_rng(SEED)
    for ep in range(EPOCHS):
        model.train()
        order = rng.permutation(len(ytr))
        t0, tot = time.time(), 0.0
        for b in range(0, len(order) - BS + 1, BS):
            idx = order[b:b+BS]
            xb = torch.as_tensor(np.stack([tile(wtr[i]) for i in idx]), device=DEVICE)
            yb = torch.as_tensor(ytr[idx], device=DEVICE, dtype=torch.long)
            # AASIST official label convention: out[:,1]=bonafide -> target 1-y
            _, o = model(xb)
            loss = crit(o, 1 - yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss)
        sv = score(model, wva)
        ve = eer_of(yva, sv)
        print(f"  epoch {ep+1}/{EPOCHS} loss={tot/(len(order)//BS):.4f} "
              f"val EER={ve:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if ve < best:
            best = ve
            torch.save(model.state_dict(), CKPT)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=False))
    print(f"  best val EER={best:.4f}")

# ─── evaluate on the standard 1846-utt test subset ───────────────────────────
zml = np.load(px.WAVE_CACHE / "i2_full_test_waves.npz", allow_pickle=True)
wml, mok = zml["waves"].astype(np.float32), zml["ok_idx"]
recs = [json.loads(px.TEST_JSON.read_text())[i] for i in mok]
yml = np.array([1 if str(r["label"]).lower().startswith("spoof") else 0 for r in recs])
sml = np.array([f'{r["attack_system"]}|{r["language"]}' if yml[i] else "bona"
                for i, r in enumerate(recs)])
sc = score(model, wml)
np.save(OUT / "aasist_mlaad_test_scores.npy", sc)
print(f"\n[J6] fine-tuned AASIST on MLAAD test: EER={eer_of(yml, sc):.4f} "
      f"AUC={roc_auc_score(yml, sc):.4f}")

big = sorted({u for u in sml if u != "bona" and (sml == u).sum() >= 8})
bl = sc[yml == 0]
H = {u: 1 - roc_auc_score(np.r_[np.zeros(len(bl)), np.ones((sml == u).sum())],
                          np.r_[bl, sc[sml == u]]) for u in big}
sysu = pd.read_csv(R / "i3_position_geometry" / "system_position.csv", index_col=0).loc[big]
sysu["hard_aasist_ft"] = [H[u] for u in big]

print("\n[law] MLAAD position law under IN-DOMAIN AASIST (non-WavLM):")
res = {}
for c in ["s_along", "sd_along", "s_orth", "vel_entropy_L12", "rog_L12"]:
    rho, p = stats.spearmanr(sysu[c], sysu["hard_aasist_ft"])
    res[c] = {"rho": float(rho), "p": float(p)}
    print(f"  {c:>16}: rho={rho:+.3f} (p={p:.4f})")

from sklearn.linear_model import Ridge
N = len(big)
def loo(Xf, y):
    Xf = np.asarray(Xf, float).reshape(N, -1)
    Xs = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-12)
    pred = np.empty(N)
    for i in range(N):
        m = np.ones(N, bool); m[i] = False
        pred[i] = Ridge(alpha=1.0).fit(Xs[m], y[m]).predict(Xs[i:i+1])[0]
    return 1 - ((y-pred)**2).sum() / ((y-y.mean())**2).sum()
y_ = sysu["hard_aasist_ft"].values
r2 = loo(sysu[["s_along", "sd_along"]].values, y_)
rng = np.random.default_rng(SEED)
p_perm = (sum(loo(sysu[["s_along", "sd_along"]].values, rng.permutation(y_)) >= r2
              for _ in range(2000)) + 1) / 2001
print(f"  LOO(s_along+sd_along)={r2:+.3f} (perm p={p_perm:.4f})")
res["loo_pos_sd"] = {"r2": float(r2), "p_perm": float(p_perm)}

gh = pd.read_csv(R / "j2_prospective" / "j2_goat_hardness.csv").set_index("system").loc[big]
aas0 = pd.read_csv(R / "j5_aasist" / "mlaad_aasist_hardness.csv", index_col=0).loc[big]
A = pd.DataFrame({"aasist_ft": sysu["hard_aasist_ft"], "aasist_zeroshot": aas0["hard_aasist"],
                  "mlaad_gat": gh["hard_mlaaddet"], "robust_goat": gh["hard_goat"]})
print("\n[agreement] (Spearman):")
print(A.corr(method="spearman").round(3).to_string())

(OUT / "j6_results.json").write_text(json.dumps(
    {"eer_test": eer_of(yml, sc), "auc_test": float(roc_auc_score(yml, sc)),
     "law": res, "agreement": A.corr(method="spearman").to_dict()}, indent=2))
sysu.to_csv(OUT / "j6_system_table.csv")
print(f"\n[J6] done -> {OUT}")
