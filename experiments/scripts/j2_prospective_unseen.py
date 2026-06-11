#!/usr/bin/env python3
"""
J2 — Prospective hardness prediction for UNSEEN systems (pre-registered)
=========================================================================
Predict how hard never-before-evaluated synthesis systems are for the detector,
from frozen-space geometry alone, BEFORE running the detector on them.

Unseen test beds:
  (a) WaveFake (10 vocoder/TTS systems; LJSpeech+JSUT domains) — unseen by the
      MLAAD detector.
  (b) VCC2020 task-1 submissions (voice conversion teams T01..T12) — unseen.
  (c) Reverse transport: robust_goat (ASVspoof-trained, never saw MLAAD) scored on
      the 61 MLAAD systems; predictions = the MLAAD geometry model fitted to the
      MLAAD-detector's hardness (tests detector-family transport of the law).

Protocol (strict order, enforced in code):
  PHASE A  features + predictions -> j2_preregistered_predictions.json (+ flush)
  PHASE B  detector scoring (3 MLAAD seeds on (a,b); 3 robust_goat seeds on (c))
  PHASE C  evaluation (Spearman pred vs actual; per-corpus breakdown)

Geometry predictors (both registered; LDA primary per J1, centroid secondary):
  pos/sd along MLAAD shrinkage-LDA direction; pos/sd along MLAAD centroid axis.
  Ridge models fit on the 61 MLAAD systems' shared hardness.

Outputs -> experiments/results/j2_prospective/
"""
from __future__ import annotations
import json, sys, time, warnings
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

OUT = px.EXP_DIR / "results" / "j2_prospective"
OUT.mkdir(parents=True, exist_ok=True)
I3 = px.EXP_DIR / "results" / "i3_position_geometry"
DEVICE = px.DEVICE
N_PER_SYS = 40

# ─── MLAAD reference: embeddings, axes, hardness model ────────────────────────
recs = json.loads(px.TEST_JSON.read_text())
ok = np.load(px.WAVE_CACHE / "i2_full_test_waves.npz", allow_pickle=True)["ok_idx"]
recs = [recs[i] for i in ok]
mlab = np.array([1 if str(r["label"]).lower().startswith("spoof") else 0 for r in recs])
msys = np.array([f'{r["attack_system"]}|{r["language"]}' if mlab[i] else "bona"
                 for i, r in enumerate(recs)])
MX = np.load(I3 / "embeddings.npz")["X12"]
mlogits = {s: np.load(I3 / f"logits_{s}.npy") for s in ["main", "s42", "s1024"]}

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats

mu_b = MX[mlab == 0].mean(0); sd_b = MX[mlab == 0].std(0) + 1e-9
Zm = (MX - mu_b) / sd_b
w_mean = Zm[mlab == 1].mean(0) - Zm[mlab == 0].mean(0); w_mean /= np.linalg.norm(w_mean)
lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(MX, mlab)

def coords(Q):
    Zq = (Q - mu_b) / sd_b
    return Zq @ w_mean, lda.decision_function(Q)

big = sorted({s for s in msys if s != "bona" and (msys == s).sum() >= 8})
def sys_hard(lgd, mask_sys, lab, bona_mask):
    return np.mean([1 - roc_auc_score(
        np.r_[np.zeros(bona_mask.sum()), np.ones(mask_sys.sum())],
        np.r_[lg[bona_mask], lg[mask_sys]]) for lg in lgd.values()])
H_ml = np.array([sys_hard(mlogits, msys == s, mlab, mlab == 0) for s in big])
pos_m, pos_l = coords(MX)
feat_ml = {}
for s in big:
    m = msys == s
    feat_ml[s] = [np.median(pos_m[m]), np.std(pos_m[m]),
                  np.median(pos_l[m]), np.std(pos_l[m])]
F = pd.DataFrame(feat_ml, index=["pos_mean", "sd_mean", "pos_lda", "sd_lda"]).T

def fit_ridge(cols):
    Xf = F[cols].values
    mu = Xf.mean(0); sd = Xf.std(0) + 1e-12
    r = Ridge(alpha=1.0).fit((Xf - mu) / sd, H_ml)
    return lambda Q: r.predict((np.asarray(Q) - mu) / sd)
pred_lda  = fit_ridge(["pos_lda", "sd_lda"])
pred_mean = fit_ridge(["pos_mean", "sd_mean"])

# ─── PHASE A: load unseen audio, compute features, REGISTER predictions ───────
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
def load16k(p):
    y, sr = sf.read(str(p), dtype="float32", always_2d=False)
    if y.ndim > 1: y = y.mean(1)
    if sr != px.SR:
        g = gcd(px.SR, sr); y = resample_poly(y, px.SR // g, sr // g).astype(np.float32)
    return px._fix(y)

def gather(root: Path, sys_dirs):
    items = []
    for d in sys_dirs:
        wavs = sorted((root / d).rglob("*.wav"))
        if len(wavs) < 10: continue
        sel = rng.choice(len(wavs), min(N_PER_SYS, len(wavs)), replace=False)
        for i in sel: items.append((d, wavs[i]))
    return items

WF_ROOT = px.BASE / "data" / "wavefake" / "generated_audio"
VC_ROOT = px.BASE / "data" / "vcc2020" / "audio"
wf_sys = sorted([p.name for p in WF_ROOT.iterdir() if p.is_dir()])
vc_sys = sorted([p.name for p in VC_ROOT.iterdir() if p.is_dir()])
items = [("wavefake", s, p) for s, p in gather(WF_ROOT, wf_sys)] + \
        [("vcc2020", s, p) for s, p in gather(VC_ROOT, vc_sys)]
print(f"[J2-A] {len(items)} utts across "
      f"{len(set((c, s) for c, s, _ in items))} unseen systems")

FEAT = OUT / "unseen_embeddings.npz"
if FEAT.exists():
    fz = np.load(FEAT, allow_pickle=True)
    UX, ucorp, usys = fz["UX"], fz["ucorp"], fz["usys"]
    uwaves = fz["uwaves"].astype(np.float32)
else:
    uwaves = np.stack([load16k(p) for _, _, p in items]).astype(np.float32)
    ucorp = np.array([c for c, _, _ in items]); usys = np.array([s for _, s, _ in items])
    wl = px.load_frozen_wavlm(DEVICE)
    UX = []
    with torch.no_grad():
        for b in range(0, len(uwaves), 8):
            xb = torch.as_tensor(uwaves[b:b+8], dtype=torch.float32, device=DEVICE)
            UX.append(wl(xb, output_hidden_states=True).hidden_states[12]
                      .mean(1).float().cpu().numpy())
            if (b // 8) % 20 == 0: print(f"  emb {b}/{len(uwaves)}", flush=True)
    UX = np.concatenate(UX)
    np.savez_compressed(FEAT, UX=UX, ucorp=ucorp, usys=usys,
                        uwaves=uwaves.astype(np.float16))
    del wl; torch.cuda.empty_cache()

upos_m, upos_l = coords(UX)
units = sorted(set(zip(ucorp.tolist(), usys.tolist())))
reg_rows = []
for c, s in units:
    m = (ucorp == c) & (usys == s)
    fm = [np.median(upos_m[m]), np.std(upos_m[m])]
    fl = [np.median(upos_l[m]), np.std(upos_l[m])]
    reg_rows.append({"corpus": c, "system": s, "n": int(m.sum()),
                     "pos_mean": fm[0], "sd_mean": fm[1],
                     "pos_lda": fl[0], "sd_lda": fl[1],
                     "pred_hard_lda": float(pred_lda([fl])[0]),
                     "pred_hard_mean": float(pred_mean([fm])[0])})
# (c) reverse transport: register MLAAD geometry predictions for robust_goat
for s in big:
    reg_rows.append({"corpus": "mlaad_x_robustgoat", "system": s,
                     "n": int((msys == s).sum()),
                     "pos_mean": F.loc[s, "pos_mean"], "sd_mean": F.loc[s, "sd_mean"],
                     "pos_lda": F.loc[s, "pos_lda"], "sd_lda": F.loc[s, "sd_lda"],
                     "pred_hard_lda": float(pred_lda([F.loc[s, ["pos_lda", "sd_lda"]].values])[0]),
                     "pred_hard_mean": float(pred_mean([F.loc[s, ["pos_mean", "sd_mean"]].values])[0])})
reg = pd.DataFrame(reg_rows)
reg_file = OUT / "j2_preregistered_predictions.json"
reg_file.write_text(json.dumps({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "note": "written BEFORE any detector scoring of these systems",
                                "predictions": reg.to_dict("records")}, indent=2))
print(f"[J2-A] predictions REGISTERED -> {reg_file}  ({len(reg)} units)")

# ─── PHASE B: detector scoring ────────────────────────────────────────────────
SEED_CKPTS = {
    "main":  px.EXP_DIR / "checkpoints" / "mlaad_robust_goat.ckpt",
    "s42":   px.EXP_DIR / "checkpoints" / "mlaad_robust_goat_seed42-best-epoch=05-val-eer=0.3030.ckpt",
    "s1024": px.EXP_DIR / "checkpoints" / "mlaad_robust_goat_seed1024-best-epoch=03-val-eer=0.2976.ckpt",
}
GOAT_CKPTS = {"g1": px.BASE / "models" / "robust_goat.ckpt",
              "g3": px.BASE / "models" / "robust_goat_seed3.ckpt",
              "g7": px.BASE / "models" / "robust_goat_seed7.ckpt"}

ulog_f = OUT / "unseen_logits.npz"
if ulog_f.exists():
    uz = np.load(ulog_f); ulogits = {k: uz[k] for k in uz.files}
else:
    ulogits = {}
    for sn, ck in SEED_CKPTS.items():
        print(f"[J2-B] scoring unseen audio under MLAAD {sn} ...", flush=True)
        gm = px.load_detector(ckpt=ck)
        ulogits[sn] = px.detector_logits(gm, uwaves)
        del gm; torch.cuda.empty_cache()
    np.savez_compressed(ulog_f, **ulogits)

mwaves = np.load(px.WAVE_CACHE / "i2_full_test_waves.npz", allow_pickle=True)["waves"].astype(np.float32)
glog_f = OUT / "robustgoat_mlaad_logits.npz"
if glog_f.exists():
    gz = np.load(glog_f); glogits = {k: gz[k] for k in gz.files}
else:
    # robust_goat checkpoints use the I1 loader (Phoneme_GAT_lit on ASVspoof ckpts)
    sys.path.insert(0, str(SCRIPTS))
    from argparse import Namespace
    import phoneme_GAT.modules as mm, phoneme_GAT.phoneme_model as pm
    from phoneme_GAT.phoneme_model import BaseModule, network_param, optim_param
    def _load(network_name="wavlm", pretrained_path=None, total_num_phonemes=198):
        network_param.network_name = network_name
        network_param.pretrained_name = "microsoft/wavlm-base"
        network_param.vocab_size = total_num_phonemes
        return BaseModule(network_param, optim_param, tokenizer=None,
                          total_num_phonemes=total_num_phonemes)
    pm.load_phoneme_model = _load; mm.load_phoneme_model = _load
    from phoneme_GAT.modules import Phoneme_GAT_lit
    def _n_edges(ck):
        c = torch.load(str(ck), weights_only=False, map_location="cpu") \
            .get("hyper_parameters", {}).get("cfg", None)
        n = getattr(getattr(c, "PhonemeGAT", None), "n_edges", None) if c else None
        return int(n) if n is not None else 10
    glogits = {}
    for sn, ck in GOAT_CKPTS.items():
        print(f"[J2-B] scoring MLAAD under robust_goat {sn} ...", flush=True)
        cfg = Namespace(PhonemeGAT=Namespace(backbone="wavlm", use_raw=False, use_GAT=True,
                        n_edges=_n_edges(ck), use_aug=True, use_pool=True, use_clip=True))
        lit = Phoneme_GAT_lit.load_from_checkpoint(str(ck), cfg=cfg,
                                                   map_location=DEVICE, strict=True)
        lit.to(DEVICE).eval(); lit.freeze()
        glogits[sn] = px.detector_logits(lit.model, mwaves)
        del lit; torch.cuda.empty_cache()
    np.savez_compressed(glog_f, **glogits)

# ─── PHASE C: evaluation ──────────────────────────────────────────────────────
print("\n[J2-C] evaluation")
res = {}
# (a,b) unseen corpora: hardness vs MLAAD bona pool, per MLAAD seed
rows = []
for c, s in units:
    m = (ucorp == c) & (usys == s)
    h = np.mean([1 - roc_auc_score(
        np.r_[np.zeros((mlab == 0).sum()), np.ones(m.sum())],
        np.r_[mlogits[sn][mlab == 0], ulogits[sn][m]]) for sn in SEED_CKPTS])
    rows.append({"corpus": c, "system": s, "hard_actual": h})
act = pd.DataFrame(rows).merge(reg[reg.corpus != "mlaad_x_robustgoat"],
                               on=["corpus", "system"])
for c in ["wavefake", "vcc2020", "ALL"]:
    sub = act if c == "ALL" else act[act.corpus == c]
    for pcol in ["pred_hard_lda", "pred_hard_mean"]:
        rho, p = stats.spearmanr(sub[pcol], sub["hard_actual"])
        res[f"{c}_{pcol}"] = {"rho": float(rho), "p": float(p), "n": len(sub)}
        print(f"  {c:>9} {pcol:>15}: rho={rho:+.3f} (p={p:.4f}, n={len(sub)})")
act.to_csv(OUT / "j2_unseen_actual.csv", index=False)

# (c) robust_goat on MLAAD systems
H_goat = np.array([np.mean([1 - roc_auc_score(
    np.r_[np.zeros((mlab == 0).sum()), np.ones((msys == s).sum())],
    np.r_[glogits[g][mlab == 0], glogits[g][msys == s]]) for g in GOAT_CKPTS])
    for s in big])
regg = reg[reg.corpus == "mlaad_x_robustgoat"].set_index("system").loc[big]
for pcol in ["pred_hard_lda", "pred_hard_mean"]:
    rho, p = stats.spearmanr(regg[pcol], H_goat)
    res[f"goat_{pcol}"] = {"rho": float(rho), "p": float(p), "n": len(big)}
    print(f"  goatXmlaad {pcol:>15}: rho={rho:+.3f} (p={p:.4f}, n={len(big)})")
rho_dd, p_dd = stats.spearmanr(H_ml, H_goat)
print(f"  detector-family hardness agreement (MLAAD-det vs robust_goat): "
      f"rho={rho_dd:+.3f} (p={p_dd:.6f})")
res["detector_family_agreement"] = {"rho": float(rho_dd), "p": float(p_dd)}
pd.DataFrame({"system": big, "hard_mlaaddet": H_ml, "hard_goat": H_goat}) \
    .to_csv(OUT / "j2_goat_hardness.csv", index=False)

(OUT / "j2_results.json").write_text(json.dumps(res, indent=2))
print(f"\n[J2] done -> {OUT}")
