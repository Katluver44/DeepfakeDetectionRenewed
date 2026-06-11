#!/usr/bin/env python3
"""
J3 — Axis-adaptive detector head
=================================
Deployable instantiation of the position law: keep the detector frozen; at
deployment estimate the domain's natural<->synthetic axis from a small calibration
set in frozen WavLM space and emit  score = z(logit) + lam * z(axis_proj).

Variants:
  (a) supervised-lite: n in {5,10,25,50,100,250} labeled utts/class -> centroid
      axis (and shrinkage-LDA when n>=25); 20 random calibration draws; evaluated
      on calibration-disjoint utterances (ITW additionally speaker-disjoint).
  (b) unsupervised: pseudo-label the domain with the detector's own most-confident
      tails (top/bottom q=20% of logits), fit centroid axis on pseudo-labels.

Datasets / detectors:
  MLAAD  x mlaad_robust_goat (3 seeds)      [in-domain]
  ITW    x mlaad_robust_goat (main)         [channel shift — the hard case]
  ASVspoof A07-A19 x robust_goat (3 seeds)  [strong detector, unseen attacks]

Outputs -> experiments/results/j3_axis_adaptive/
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SEED = 42
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import px_common as px

OUT = px.EXP_DIR / "results" / "j3_axis_adaptive"
OUT.mkdir(parents=True, exist_ok=True)
R = px.EXP_DIR / "results"

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
def eer_of(y, s):
    fpr, tpr, _ = roc_curve(y, s, pos_label=1); fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr))); return float((fpr[i] + fnr[i]) / 2)

# ─── assemble datasets: X (frozen L12), labels, logits dict, group ids ────────
import json as _json
DS = {}

recs = _json.loads(px.TEST_JSON.read_text())
ok = np.load(px.WAVE_CACHE / "i2_full_test_waves.npz", allow_pickle=True)["ok_idx"]
recs = [recs[i] for i in ok]
mlab = np.array([1 if str(r["label"]).lower().startswith("spoof") else 0 for r in recs])
DS["mlaad"] = dict(
    X=np.load(R / "i3_position_geometry" / "embeddings.npz")["X12"], y=mlab,
    logits={s: np.load(R / "i3_position_geometry" / f"logits_{s}.npy")
            for s in ["main", "s42", "s1024"]},
    group=np.array([f'{r["attack_system"]}|{r["language"]}' if mlab[i] else f"bona{i%5}"
                    for i, r in enumerate(recs)]))

iutt = pd.read_csv(R / "i5_itw_transfer" / "utt_table.csv")
DS["itw"] = dict(
    X=np.load(R / "i5_itw_transfer" / "features.npz")["X12"], y=iutt["label"].values,
    logits={"main": iutt["logit"].values}, group=iutt["speaker"].values)

az = np.load(R / "i4_asvspoof_position" / "features.npz")
lz = np.load(R / "i1_geometry_causal_decomp" / "i1_logits.npz")
DS["asvspoof"] = dict(
    X=az["X12"], y=lz["labels"],
    logits={s: lz[f"{s}__baseline"] for s in ["s1", "s3", "s7"]},
    group=None)  # groups not needed: utterance-disjoint only

# rebuild ASVspoof groups (attacks) deterministically as in I4
from datasets import load_dataset
ds_ = load_dataset("Bisher/as_vspoof_2019_la",
                   cache_dir=str(px.BASE / "data" / "asvspoof_2019_la"),
                   trust_remote_code=True)["test"]
sysids = ds_["system_id"]
rr = np.random.default_rng(SEED)
EVAL_ATTACKS = [f"A{i:02d}" for i in range(7, 20)]
sp_i = [i for i, s in enumerate(sysids) if s in EVAL_ATTACKS]
bo_i = [i for i, s in enumerate(sysids) if s == "-"]
sel = sorted(rr.choice(sp_i, 800, replace=False).tolist() +
             rr.choice(bo_i, 800, replace=False).tolist())
DS["asvspoof"]["group"] = np.array([sysids[i] if sysids[i] != "-" else f"bona{i%7}"
                                    for i in sel])

# ─── head logic ───────────────────────────────────────────────────────────────
def fuse(lg_te, ax_te, lg_cal, ax_cal, y_cal, lams=(0.5, 1.0, 1.5, 2.0)):
    zl = (lg_te - lg_cal.mean()) / (lg_cal.std() + 1e-12)
    zp = (ax_te - ax_cal.mean()) / (ax_cal.std() + 1e-12)
    zlc = (lg_cal - lg_cal.mean()) / (lg_cal.std() + 1e-12)
    zpc = (ax_cal - ax_cal.mean()) / (ax_cal.std() + 1e-12)
    if len(set(y_cal)) > 1 and len(y_cal) >= 10:
        lam = max(lams, key=lambda L: roc_auc_score(y_cal, zlc + L * zpc))
    else:
        lam = 1.5
    return zl + lam * zp

def centroid_axis(Xc, yc):
    mu = Xc[yc == 0].mean(0); sd = Xc[yc == 0].std(0) + 1e-9
    Z = (Xc - mu) / sd
    w = Z[yc == 1].mean(0) - Z[yc == 0].mean(0); w /= (np.linalg.norm(w) + 1e-12)
    return lambda Q: ((Q - mu) / sd) @ w

rows = []
rngm = np.random.default_rng(SEED)
N_CAL = [5, 10, 25, 50, 100, 250]
REPS = 20

for dname, D in DS.items():
    X, y, group = D["X"], D["y"], D["group"]
    for sn, lg in D["logits"].items():
        base = {"dataset": dname, "seed": sn,
                "EER_det": eer_of(y, lg), "AUC_det": roc_auc_score(y, lg)}
        # (a) supervised-lite calibration curves
        for n in N_CAL:
            ee, aa, ee_l, aa_l = [], [], [], []
            for rep in range(REPS):
                r2 = np.random.default_rng(1000 * rep + n)
                # group-aware calibration draw: pick calib groups, then utts
                gb = np.unique(group[y == 0]); gs = np.unique(group[y == 1])
                cal_idx = []
                for cls, gpool in [(0, gb), (1, gs)]:
                    gsel = r2.permutation(gpool)
                    pool = []
                    for g in gsel:
                        pool.extend(np.where((group == g) & (y == cls))[0].tolist())
                        if len(pool) >= n: break
                    cal_idx.extend(pool[:n])
                cal_idx = np.array(cal_idx)
                cal_groups = set(group[cal_idx].tolist())
                te = np.array([i for i in range(len(y))
                               if i not in set(cal_idx.tolist())
                               and group[i] not in cal_groups])
                if len(te) < 200 or len(set(y[te])) < 2: continue
                ax = centroid_axis(X[cal_idx], y[cal_idx])
                f = fuse(lg[te], ax(X[te]), lg[cal_idx], ax(X[cal_idx]), y[cal_idx])
                ee.append(eer_of(y[te], f)); aa.append(roc_auc_score(y[te], f))
                if n >= 25:
                    try:
                        ld = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                        ld.fit(X[cal_idx], y[cal_idx])
                        f2 = fuse(lg[te], ld.decision_function(X[te]),
                                  lg[cal_idx], ld.decision_function(X[cal_idx]), y[cal_idx])
                        ee_l.append(eer_of(y[te], f2)); aa_l.append(roc_auc_score(y[te], f2))
                    except Exception:
                        pass
                # detector-only reference on the same te subset
            if ee:
                rows.append({**base, "head": "centroid", "n_cal": n,
                             "EER": float(np.mean(ee)), "EER_sd": float(np.std(ee)),
                             "AUC": float(np.mean(aa))})
            if ee_l:
                rows.append({**base, "head": "lda", "n_cal": n,
                             "EER": float(np.mean(ee_l)), "EER_sd": float(np.std(ee_l)),
                             "AUC": float(np.mean(aa_l))})
        # (b) unsupervised pseudo-label head (whole set, q=20% tails)
        q = 0.2
        lo, hi = np.quantile(lg, [q, 1 - q])
        pl = np.full(len(y), -1); pl[lg <= lo] = 0; pl[lg >= hi] = 1
        m = pl >= 0
        ax = centroid_axis(X[m], pl[m])
        f = fuse(lg, ax(X), lg[m], ax(X[m]), pl[m])
        rows.append({**base, "head": "pseudo_unsup", "n_cal": 0,
                     "EER": eer_of(y, f), "EER_sd": 0.0, "AUC": roc_auc_score(y, f)})
        print(f"  [{dname}/{sn}] det EER={base['EER_det']:.4f} | "
              f"unsup EER={eer_of(y, f):.4f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "j3_results.csv", index=False)

# summary table: mean over seeds
print("\n[J3] summary (mean over seeds):")
summ = df.groupby(["dataset", "head", "n_cal"]).agg(
    EER=("EER", "mean"), AUC=("AUC", "mean"),
    EER_det=("EER_det", "mean")).reset_index()
summ["dEER"] = summ["EER"] - summ["EER_det"]
summ.to_csv(OUT / "j3_summary.csv", index=False)
print(summ.round(4).to_string(index=False))
print(f"\n[J3] done -> {OUT}")
