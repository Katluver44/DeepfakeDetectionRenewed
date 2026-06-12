#!/usr/bin/env python3
"""Audit A5 — Fusion claims: leakage, mislabeling, and the "zero-training" framing.

Claims under test:
  (a) "Axis fusion reduces EER 0.272->0.163 on MLAAD (WavLM-GAT)" [I7]
  (b) "ITW-internal axis fusion: EER 0.363 -> 0.292 (WavLM-GAT)" [report 3.3]
      -> itw_fusion_test.json actually shows 0.292 is the axis ALONE;
         fusion was 0.312. Also: no committed script produced that JSON.
  (c) "AASIST zero-shot 0.486 -> 0.161 on ITW; 0.376 -> 0.116 on MLAAD" [J5 H4]
      -> the axis there is a *supervised shrinkage-LDA trained on the eval
         corpus* (group-disjoint). How much of the gain is the LDA alone vs
         genuine fusion synergy? Is "zero-training-cost" fair?
  (d) per-fold consistency of I7 gains (concatenated-EER artifact check).

All recomputed from cached scores/embeddings.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
import audit_common as ac
from audit_common import eer_of

OUT = ac.OUT_ROOT / "audit5_fusion_claims"
OUT.mkdir(parents=True, exist_ok=True)
res = {}

# ── (a)+(d) I7 MLAAD fusion verification ─────────────────────────────────────
i7 = np.load(ac.RES / "i7_axis_fusion" / "i7_scores.npz")
y = i7["labels"]
eers = {}
for sn in ["main", "s42", "s1024"]:
    eers[sn] = {"baseline": eer_of(y, i7[f"logit_{sn}"]),
                "fused": eer_of(y, i7[f"fused_{sn}"])}
base_m = float(np.mean([v["baseline"] for v in eers.values()]))
fus_m = float(np.mean([v["fused"] for v in eers.values()]))
res["i7_mlaad"] = {"per_seed": eers, "baseline_mean": base_m, "fused_mean": fus_m,
                   "axis_alone_eer": eer_of(y, i7["s_along"]),
                   "claimed": {"baseline": 0.272, "fused": 0.163}}
print(f"[a] I7 MLAAD: baseline={base_m:.3f} fused={fus_m:.3f} "
      f"axis_alone={res['i7_mlaad']['axis_alone_eer']:.3f} (claimed 0.272->0.163)")

# per-fold check needs fold assignment: reproduce I7's fold RNG
ml = ac.load_mlaad()
labels, systems, X = ml["labels"], ml["systems"], ml["X"]
rng = np.random.default_rng(42)
sys_units = sorted(set(systems) - {"bona"})
rng.shuffle(sys_units)
folds = np.array_split(np.array(sys_units), 5)
bona_idx = np.where(labels == 0)[0]
bona_folds = np.array_split(rng.permutation(bona_idx), 5)
pf = []
for k in range(5):
    te_sys = set(folds[k]); te_b = set(bona_folds[k].tolist())
    te = np.array([(systems[i] in te_sys) or (i in te_b) for i in range(len(labels))])
    row = {"fold": k, "n_te": int(te.sum())}
    for sn in ["main", "s42", "s1024"]:
        row[f"dEER_{sn}"] = eer_of(y[te], i7[f"fused_{sn}"][te]) - eer_of(y[te], i7[f"logit_{sn}"][te])
    pf.append(row)
pf = pd.DataFrame(pf)
pf.to_csv(OUT / "i7_per_fold_dEER.csv", index=False)
res["i7_per_fold"] = pf.to_dict("records")
print("[d] I7 per-fold dEER:")
print(pf.round(4).to_string(index=False))

# ── (b) ITW internal axis, speaker-disjoint, recomputed from scratch ─────────
itw_utt = pd.read_csv(ac.RES / "i5_itw_transfer" / "utt_table.csv")
X_itw = np.load(ac.RES / "i5_itw_transfer" / "features.npz")["X12"]
y_itw = itw_utt["label"].values
spk = itw_utt["speaker"].values
logit_itw = itw_utt["logit"].values
print(f"\n[b] ITW: baseline detector EER={eer_of(y_itw, logit_itw):.4f} (claimed 0.363)")

rng = np.random.default_rng(0)
spk_units = np.array(sorted(set(spk)))
rng.shuffle(spk_units)
sfolds = np.array_split(spk_units, 5)
axis_sc = np.full(len(y_itw), np.nan)
fused_sc = np.full(len(y_itw), np.nan)
LAMS = [0.5, 1.0, 1.5, 2.0]
for k in range(5):
    te = np.isin(spk, sfolds[k]); tr = ~te
    trb = tr & (y_itw == 0); trs = tr & (y_itw == 1)
    mu = X_itw[trb].mean(0); sd = X_itw[trb].std(0) + 1e-9
    w = ((X_itw[trs] - mu) / sd).mean(0) - ((X_itw[trb] - mu) / sd).mean(0)
    w /= np.linalg.norm(w)
    proj_tr = ((X_itw[tr] - mu) / sd) @ w
    proj_te = ((X_itw[te] - mu) / sd) @ w
    axis_sc[te] = proj_te
    pm, ps = proj_tr.mean(), proj_tr.std() + 1e-12
    lm, ls = logit_itw[tr].mean(), logit_itw[tr].std() + 1e-12
    ztr_l = (logit_itw[tr] - lm) / ls; ztr_p = (proj_tr - pm) / ps
    lam = max(LAMS, key=lambda L: roc_auc_score(y_itw[tr], ztr_l + L * ztr_p))
    fused_sc[te] = (logit_itw[te] - lm) / ls + lam * (proj_te - pm) / ps
res["itw_internal"] = {
    "baseline_eer": eer_of(y_itw, logit_itw),
    "axis_alone_eer_speaker_disjoint": eer_of(y_itw, axis_sc),
    "fused_eer_speaker_disjoint": eer_of(y_itw, fused_sc),
    "json_artifact": json.loads((ac.RES / "i7_axis_fusion" / "itw_fusion_test.json").read_text()),
    "note": "report's '0.363 -> 0.292 fusion' number is the AXIS ALONE in the artifact; "
            "true fusion was worse (0.312); no committed script produced this JSON"}
print(f"    recomputed: axis_alone={res['itw_internal']['axis_alone_eer_speaker_disjoint']:.4f} "
      f"fused={res['itw_internal']['fused_eer_speaker_disjoint']:.4f} "
      f"(artifact: axis 0.2917, fused 0.3123)")

# ── (c) J5 H4 decomposition: LDA-alone vs fused for AASIST ───────────────────
asc = np.load(ac.RES / "j5_aasist" / "aasist_scores.npz")
def orient(s, yv):
    return -s if np.median(s[yv == 1]) < np.median(s[yv == 0]) else s

datasets = {}
datasets["mlaad"] = dict(X=X, y=labels, s=orient(asc["sml"], labels), g=systems)
datasets["itw"] = dict(X=X_itw, y=y_itw, s=orient(asc["sit"], y_itw), g=spk)
z21 = np.load(ac.RES / "j4_asvspoof21" / "waves_sel.npz", allow_pickle=True)
y21 = (z21["atts"] != "bonafide").astype(int)
X21 = np.load(ac.RES / "j4_asvspoof21" / "embeddings.npz")["X12"]
datasets["asvspoof21"] = dict(X=X21, y=y21, s=orient(asc["s21"], y21), g=z21["atts"])

def fusion_decomp(Xd, yd, sd_, gd, strict_group_bona=False, seed=42):
    """Mirror j5's fusion_test (spoof units group-disjoint, bona RANDOM folds),
    plus a strict variant where bona is also group-disjoint (no speaker leak)."""
    rng = np.random.default_rng(seed)
    units = sorted(set(gd[yd == 1]))
    rng.shuffle(units)
    gfolds = np.array_split(np.array(units), 5)
    if strict_group_bona:
        bunits = sorted(set(gd[yd == 0]))
        rng.shuffle(bunits)
        bfolds_u = np.array_split(np.array(bunits), 5)
    else:
        bidx = np.where(yd == 0)[0]
        bfolds = np.array_split(rng.permutation(bidx), 5)
    lda_sc = np.full(len(yd), np.nan); fus = np.full(len(yd), np.nan)
    for k in range(5):
        te_u = set(gfolds[k])
        if strict_group_bona:
            te_b_u = set(bfolds_u[k])
            te = np.array([(gd[i] in te_u) if yd[i] else (gd[i] in te_b_u)
                           for i in range(len(yd))])
        else:
            te_b = set(bfolds[k].tolist())
            te = np.array([(gd[i] in te_u) or (i in te_b) for i in range(len(yd))])
        tr = ~te
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xd[tr], yd[tr])
        ptr, pte = lda.decision_function(Xd[tr]), lda.decision_function(Xd[te])
        lda_sc[te] = pte
        pm, ps = ptr.mean(), ptr.std() + 1e-12
        lm, ls = sd_[tr].mean(), sd_[tr].std() + 1e-12
        zl = (sd_[tr] - lm) / ls; zp = (ptr - pm) / ps
        lam = max([0.5, 1.0, 1.5, 2.0], key=lambda L: roc_auc_score(yd[tr], zl + L * zp))
        fus[te] = (sd_[te] - lm) / ls + lam * (pte - pm) / ps
    return {"lda_axis_alone_eer": eer_of(yd, lda_sc), "fused_eer": eer_of(yd, fus)}

dec = {}
for name, D in datasets.items():
    Xd, yd, sd_, gd = D["X"], D["y"], np.asarray(D["s"], float), np.asarray(D["g"])
    out = {"aasist_eer": eer_of(yd, sd_)}
    out.update(fusion_decomp(Xd, yd, sd_, gd))
    if name == "itw":  # bona speakers leak across folds in j5's split — quantify
        strict = fusion_decomp(Xd, yd, sd_, gd, strict_group_bona=True)
        out["strict_bona_disjoint"] = strict
    dec[name] = out
    print(f"[c] {name:>10}: AASIST={out['aasist_eer']:.4f}  "
          f"LDA-alone={out['lda_axis_alone_eer']:.4f}  fused={out['fused_eer']:.4f}"
          + (f"  | strict: LDA={strict['lda_axis_alone_eer']:.4f} "
             f"fused={strict['fused_eer']:.4f}" if name == "itw" else ""))
res["j5_h4_decomposition"] = dec
res["j5_h4_verdict"] = ("the large 'fusion' gains under domain shift are driven by the "
                        "supervised corpus-internal LDA axis itself (trained on labeled "
                        "eval-corpus data, group-disjoint); calling this 'zero-training-cost' "
                        "is misleading — it is a supervised linear probe + ensemble")

(OUT / "audit5_results.json").write_text(json.dumps(res, indent=2, default=str))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4.2))
names = list(dec.keys())
xpos = np.arange(len(names))
for off, key, lab in [(-0.25, "aasist_eer", "AASIST alone"),
                      (0.0, "lda_axis_alone_eer", "LDA axis alone (supervised)"),
                      (0.25, "fused_eer", "fused")]:
    ax.bar(xpos + off, [dec[n][key] for n in names], width=0.23, label=lab)
ax.set_xticks(xpos); ax.set_xticklabels(names)
ax.set_ylabel("EER"); ax.legend()
ax.set_title("J5-H4 decomposition: how much of the 'fusion' gain is just the LDA probe?")
fig.tight_layout()
fig.savefig(OUT / "audit5_fusion_decomposition.png", dpi=150)
print(f"done -> {OUT}")
