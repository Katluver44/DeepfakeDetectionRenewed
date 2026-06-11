#!/usr/bin/env python3
"""
J5 — Non-WavLM baseline: do the hypotheses hold for AASIST?
============================================================
Detector: OFFICIAL pretrained AASIST (raw-waveform sinc-conv + heterogeneous graph
attention; no SSL anywhere), trained on ASVspoof 2019 LA train. Using the official
checkpoint avoids "you crippled the baseline" critiques; training-on-MLAAD is
phase 2 only if needed.

Hypotheses tested:
  H1 (prospective, cross-family): J4's registered geometry predictions (written
      before ANY detector scored these systems) predict AASIST's per-attack
      hardness on ASVspoof 2021 LA clean. In-domain for AASIST (unseen attacks).
  H2 (position law, zero-shot): frozen-WavLM corpus-internal geometry predicts
      AASIST's per-system hardness on MLAAD (zero-shot domain for AASIST).
  H3 (detector-family agreement): hardness agreement matrix AASIST vs MLAAD-GAT
      seeds vs robust_goat.
  H4 (under-reading/fusion): does corpus-internal frozen-axis fusion improve
      AASIST on 2021 / MLAAD / ITW? (system/attack/speaker-disjoint folds)
  H5 (ITW): AASIST EER on ITW; per-speaker hardness vs s_orth / vel_entropy.

Outputs -> experiments/results/j5_aasist/
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
SEED = 42
rng = np.random.default_rng(SEED)
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import px_common as px

OUT = px.EXP_DIR / "results" / "j5_aasist"
OUT.mkdir(parents=True, exist_ok=True)
R = px.EXP_DIR / "results"
DEVICE = px.DEVICE
NB_SAMP = 64600

# ─── AASIST ───────────────────────────────────────────────────────────────────
sys.path.insert(0, str(px.BASE / "baselines" / "aasist"))
from models.AASIST import Model as AASIST
conf = json.loads((px.BASE / "baselines" / "aasist" / "config" / "AASIST.conf").read_text())
model = AASIST(conf["model_config"]).to(DEVICE)
sd = torch.load(px.BASE / "baselines" / "aasist" / "models" / "weights" / "AASIST.pth",
                map_location=DEVICE, weights_only=False)
model.load_state_dict(sd)
model.eval()
print("[J5] official AASIST loaded")

def tile(w):
    w = np.asarray(w, np.float32)
    if len(w) >= NB_SAMP: return w[:NB_SAMP]
    reps = -(-NB_SAMP // len(w))
    return np.tile(w, reps)[:NB_SAMP]

@torch.no_grad()
def aasist_scores(waves, bs=24):
    out = []
    for b in range(0, len(waves), bs):
        xb = torch.as_tensor(np.stack([tile(w) for w in waves[b:b+bs]]), device=DEVICE)
        _, o = model(xb)
        out.append(o[:, 1].float().cpu().numpy())   # official: col 1 = bonafide score
    return np.concatenate(out)

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
def eer_of(y, s):
    fpr, tpr, _ = roc_curve(y, s, pos_label=1); fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr))); return float((fpr[i] + fnr[i]) / 2)

# ─── datasets ─────────────────────────────────────────────────────────────────
# (a) ASVspoof 2021 clean subset (J4)
z21 = np.load(R / "j4_asvspoof21" / "waves_sel.npz", allow_pickle=True)
w21, atts = z21["waves"].astype(np.float32), z21["atts"]
y21 = (atts != "bonafide").astype(int)
X21 = np.load(R / "j4_asvspoof21" / "embeddings.npz")["X12"]
# (b) MLAAD test
zml = np.load(px.WAVE_CACHE / "i2_full_test_waves.npz", allow_pickle=True)
wml, mok = zml["waves"].astype(np.float32), zml["ok_idx"]
recs = [json.loads(px.TEST_JSON.read_text())[i] for i in mok]
yml = np.array([1 if str(r["label"]).lower().startswith("spoof") else 0 for r in recs])
sml = np.array([f'{r["attack_system"]}|{r["language"]}' if yml[i] else "bona"
                for i, r in enumerate(recs)])
Xml = np.load(R / "i3_position_geometry" / "embeddings.npz")["X12"]
# (c) ITW subset (same as I5/I6)
itw = px.load_itw_waves(n_per_class=3000)
r2 = np.random.default_rng(SEED)
sel = np.sort(np.r_[r2.choice(np.where(itw["labels"] == 0)[0], 1500, replace=False),
                    r2.choice(np.where(itw["labels"] == 1)[0], 1500, replace=False)])
wit, yit, spk = itw["waves"][sel], itw["labels"][sel], itw["speakers"][sel]
Xit = np.load(R / "i5_itw_transfer" / "features.npz")["X12"]
iutt = pd.read_csv(R / "i5_itw_transfer" / "utt_table.csv")

# ─── score everything ─────────────────────────────────────────────────────────
SC = OUT / "aasist_scores.npz"
if SC.exists():
    s = np.load(SC); s21, sml_sc, sit = s["s21"], s["sml"], s["sit"]
else:
    print("  scoring ASVspoof21 ...", flush=True); s21 = aasist_scores(w21)
    print("  scoring MLAAD ...", flush=True);      sml_sc = aasist_scores(wml)
    print("  scoring ITW ...", flush=True);        sit = aasist_scores(wit)
    np.savez_compressed(SC, s21=s21, sml=sml_sc, sit=sit)

# orient: higher = spoof (our convention)
if np.median(s21[y21 == 1]) > np.median(s21[y21 == 0]):
    orient = +1.0
else:
    orient = -1.0
s21, sml_sc, sit = orient * s21, orient * sml_sc, orient * sit
print(f"  orientation={orient:+.0f}  EER: 2021={eer_of(y21, s21):.4f}  "
      f"MLAAD={eer_of(yml, sml_sc):.4f}  ITW={eer_of(yit, sit):.4f}")

# ─── H1: prospective predictions vs AASIST hardness on 2021 ──────────────────
ATT = [f"A{i:02d}" for i in range(7, 20)]
def hard_units(scores, labels, units, unit_arr):
    bl = scores[labels == 0]
    return {u: 1 - roc_auc_score(np.r_[np.zeros(len(bl)), np.ones((unit_arr == u).sum())],
                                 np.r_[bl, scores[unit_arr == u]]) for u in units}
H21 = hard_units(s21, y21, ATT, atts)
reg = json.loads((R / "j4_asvspoof21" / "j4_preregistered_predictions.json").read_text())
P = pd.DataFrame(reg["table"]).set_index("attack")
P["hard_aasist"] = [H21[a] for a in P.index]
print("\n[H1] AASIST per-attack hardness vs J4 registered predictions:")
h1 = {}
for k in ["P1", "P2", "P3", "P4"]:
    rho, p = stats.spearmanr(P[k], P["hard_aasist"])
    top3p = set(P[k].sort_values(ascending=False).index[:3])
    top3a = set(P["hard_aasist"].sort_values(ascending=False).index[:3])
    h1[k] = {"rho": float(rho), "p": float(p), "top3_hits": len(top3p & top3a)}
    print(f"  {k}: rho={rho:+.3f} (p={p:.4f}) top3 hits={len(top3p & top3a)}/3")
print("  AASIST hardest:", P["hard_aasist"].sort_values(ascending=False).head(4).round(3).to_dict())
mlhard21 = pd.read_csv(R / "j4_asvspoof21" / "j4_table.csv").set_index("attack")["hard_actual"]
rho_fam21, p_fam21 = stats.spearmanr(P["hard_aasist"], mlhard21.loc[P.index])
print(f"  family agreement on 2021 (AASIST vs MLAAD-det): rho={rho_fam21:+.3f} (p={p_fam21:.4f})")

# ─── H2: position law for AASIST on MLAAD (zero-shot) ─────────────────────────
big = sorted({u for u in sml if u != "bona" and (sml == u).sum() >= 8})
Hml_aasist = hard_units(sml_sc, yml, big, sml)
sysu = pd.read_csv(R / "i3_position_geometry" / "system_position.csv", index_col=0).loc[big]
sysu["hard_aasist"] = [Hml_aasist[u] for u in big]
print("\n[H2] MLAAD position law under AASIST (zero-shot):")
h2 = {}
for c in ["s_along", "sd_along", "vel_entropy_L12", "s_orth"]:
    rho, p = stats.spearmanr(sysu[c], sysu["hard_aasist"])
    h2[c] = {"rho": float(rho), "p": float(p)}
    print(f"  {c:>16}: rho={rho:+.3f} (p={p:.4f})")
# H3 family agreement matrix
gh = pd.read_csv(R / "j2_prospective" / "j2_goat_hardness.csv").set_index("system").loc[big]
A = pd.DataFrame({"aasist": sysu["hard_aasist"], "mlaad_det": gh["hard_mlaaddet"],
                  "robust_goat": gh["hard_goat"]})
print("\n[H3] detector-family hardness agreement (MLAAD systems, Spearman):")
print(A.corr(method="spearman").round(3).to_string())

# ─── H4: corpus-internal axis fusion for AASIST ───────────────────────────────
def fusion_test(X, y, scores, groups, n_folds=5):
    units = sorted(set(groups[y == 1]))
    rng2 = np.random.default_rng(SEED)
    rng2.shuffle(units)
    folds = np.array_split(np.array(units), n_folds)
    bidx = np.where(y == 0)[0]
    bfolds = np.array_split(rng2.permutation(bidx), n_folds)
    fused = np.full(len(y), np.nan)
    for k in range(n_folds):
        te_u = set(folds[k]); te_b = set(bfolds[k].tolist())
        te = np.array([(groups[i] in te_u) or (i in te_b) for i in range(len(y))])
        tr = ~te
        trb = tr & (y == 0); trs = tr & (y == 1)
        ld = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        ld.fit(X[tr], y[tr])
        ax_tr, ax_te = ld.decision_function(X[tr]), ld.decision_function(X[te])
        zl = (scores[te] - scores[tr].mean()) / (scores[tr].std() + 1e-12)
        zp = (ax_te - ax_tr.mean()) / (ax_tr.std() + 1e-12)
        zlc = (scores[tr] - scores[tr].mean()) / (scores[tr].std() + 1e-12)
        zpc = (ax_tr - ax_tr.mean()) / (ax_tr.std() + 1e-12)
        lam = max([0.5, 1.0, 1.5, 2.0],
                  key=lambda L: roc_auc_score(y[tr], zlc + L * zpc))
        fused[te] = zl + lam * zp
    return eer_of(y, fused), roc_auc_score(y, fused)

print("\n[H4] corpus-internal LDA-axis fusion for AASIST:")
h4 = {}
for name, X_, y_, s_, g_ in [("asvspoof21", X21, y21, s21, atts),
                             ("mlaad", Xml, yml, sml_sc, sml),
                             ("itw", Xit, yit, sit, spk)]:
    e0 = eer_of(y_, s_)
    e1, a1 = fusion_test(X_, y_, s_, g_)
    h4[name] = {"EER_aasist": e0, "EER_fused": e1, "AUC_fused": a1}
    print(f"  {name:>10}: AASIST EER={e0:.4f} -> fused {e1:.4f}  (dEER={e1-e0:+.4f})")

# ─── H5: ITW per-speaker law under AASIST ─────────────────────────────────────
spk_units = [u for u in sorted(set(spk[yit == 1])) if (spk[yit == 1] == u).sum() >= 10]
Hit = hard_units(sit, yit, spk_units, spk)
sp = iutt[iutt.label == 1].groupby("speaker").median(numeric_only=True)
sp = sp.loc[[u for u in spk_units if u in sp.index]]
sp["hard_aasist"] = [Hit[u] for u in sp.index]
print("\n[H5] ITW per-speaker law under AASIST:")
h5 = {}
for c in ["s_orth", "s_along", "ve12", "rog12"]:
    rho, p = stats.spearmanr(sp[c], sp["hard_aasist"])
    h5[c] = {"rho": float(rho), "p": float(p)}
    print(f"  {c:>8}: rho={rho:+.3f} (p={p:.4f})")

(OUT / "j5_results.json").write_text(json.dumps(
    {"eer": {"asvspoof21": eer_of(y21, s21), "mlaad": eer_of(yml, sml_sc),
             "itw": eer_of(yit, sit)},
     "H1": h1, "H1_family21": {"rho": float(rho_fam21), "p": float(p_fam21)},
     "H2": h2, "H3": A.corr(method="spearman").to_dict(),
     "H4": h4, "H5": h5}, indent=2))
sysu.to_csv(OUT / "mlaad_aasist_hardness.csv")
P.to_csv(OUT / "asvspoof21_aasist.csv")
print(f"\n[J5] done -> {OUT}")
