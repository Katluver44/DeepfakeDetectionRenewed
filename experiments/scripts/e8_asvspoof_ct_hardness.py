#!/usr/bin/env python3
"""
E8 ASVspoof — Does C/T predict per-attack detection hardness in ASVspoof?
=========================================================================

The MLAAD roadmap showed two frozen-WavLM difficulty axes predict per-system
detection hardness across 63 systems:
  C (Deep Compactness)      = -rog@L12       (higher = more compact = harder)
  T (Trajectory Irregularity)= vel_entropy@L9 (higher = burstier = harder)

E4 only peeked at A01-A06 (n=6, p~0.4 -> uninterpretable, and those are the
*training* attacks). This experiment does the proper test on the ASVspoof 2019 LA
EVAL split = A07-A19 (13 attacks, the canonical unseen-attack difficulty
benchmark), at two granularities:

  PER-ATTACK (n=13, the MLAAD-analogue): per-attack mean C/T vs per-attack
    EER/AUC/Acc (scored against a shared bona-fide pool). Spearman/Pearson with
    bootstrap CIs + permutation p; R^2(C+T)+LOO; quartile stratification.
  PER-UTTERANCE (n~few-thousand, high power): pooled C/T vs raw logit on spoof,
    plus a between- vs within-attack decomposition (is a per-attack correlation a
    real per-utterance mechanism, or just 13 attack means lining up?).

Two detectors (C/T are model-agnostic; the model only supplies the hardness label):
  GOAT        models/goat.ckpt          (cleaner exposure baseline)
  robust_GOAT models/robust_goat.ckpt   (NOTE: trained on train+test of ASVspoof,
              so it has seen A07-A19 -> its per-attack EER is partly in-distribution;
              reported for robustness, GOAT is the headline.)

Conservative interpretation: report effect sizes + CIs; n=13 is small, so the
per-utterance analysis is the real power. Evaluation only -- no retraining.
"""
from __future__ import annotations
import sys
import warnings
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio.transforms as TAT
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
OUTDIR  = BASE / "outputs" / "asvspoof_ct_hardness"
OUTDIR.mkdir(parents=True, exist_ok=True)
for _p in (str(BASE), str(EXP_DIR), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR     = 16_000
TARGET_LEN    = 3 * TARGET_SR
NF_PER_SAMPLE = TARGET_LEN // 320 - 1
BATCH_SIZE    = 16
CACHE_DIR     = BASE / "data" / "asvspoof_2019_la"
EVAL_ATTACKS  = [f"A{i:02d}" for i in range(7, 20)]   # A07..A19
N_PER_ATTACK  = 300
N_BONAFIDE    = 3000

MODELS = {"GOAT": BASE / "models" / "goat.ckpt",
          "robust_GOAT": BASE / "models" / "robust_goat.ckpt"}
HEADLINE = "GOAT"
SIG_FEATURES = BASE / "outputs" / "sig_layer_features.csv"

from _ablation_common import compute_eer, _all_metrics, _quartile_buckets  # noqa: E402

print(f"[E8 ASVspoof] device={DEVICE}  out={OUTDIR}")

# ═══════════════════════════════════════════════════════════════════════════════
# Trajectory metrics (verbatim from e4/e5) + audio crop
# ═══════════════════════════════════════════════════════════════════════════════
def rog(frames):
    c = frames.mean(0)
    return float(np.sqrt(np.mean(np.sum((frames - c) ** 2, 1))))

def vel_entropy(frames):
    vels = np.linalg.norm(frames[1:] - frames[:-1], axis=1)
    n = max(5, min(20, len(vels) // 4))
    h, _ = np.histogram(vels, bins=n)
    h = h.astype(float) + 1e-8
    h /= h.sum()
    return float(-np.sum(h * np.log(h)))

def _center_crop_pad(wav):
    if wav.ndim > 1:
        wav = wav.mean(0)
    if len(wav) < TARGET_LEN:
        reps = -(-TARGET_LEN // len(wav))
        wav = wav.repeat(reps)
    mid = (len(wav) - TARGET_LEN) // 2
    return wav[mid: mid + TARGET_LEN]

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load ASVspoof eval split (A07-A19) + bonafide, balanced sample
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1] Loading ASVspoof 2019 LA test split (A07-A19) ...", flush=True)
from datasets import load_dataset, Audio as HFAudio
ds = load_dataset("Bisher/as_vspoof_2019_la", cache_dir=str(CACHE_DIR), trust_remote_code=True)
test = ds["test"]
sysids = test["system_id"]
keys = test["key"]

rng = np.random.default_rng(SEED)
by_attack = defaultdict(list)
bona_idx = []
for i, sid in enumerate(sysids):
    if sid in EVAL_ATTACKS:
        by_attack[sid].append(i)
    elif sid == "-":
        bona_idx.append(i)

sel_spoof = {}
for a in EVAL_ATTACKS:
    idx = by_attack[a]
    sel_spoof[a] = rng.choice(idx, min(N_PER_ATTACK, len(idx)), replace=False).tolist()
sel_bona = rng.choice(bona_idx, min(N_BONAFIDE, len(bona_idx)), replace=False).tolist()
all_sel = sorted([i for a in EVAL_ATTACKS for i in sel_spoof[a]] + sel_bona)
sel_set = set(all_sel)
attack_of = {i: sysids[i] for a in EVAL_ATTACKS for i in sel_spoof[a]}
print(f"  attacks={len(EVAL_ATTACKS)}  spoof={sum(len(v) for v in sel_spoof.values())}  "
      f"bonafide={len(sel_bona)}  total={len(all_sel)}")

test_sel = test.select(all_sel).cast_column("audio", HFAudio(sampling_rate=TARGET_SR))
# label/attack aligned to selection order
sel_attack = [attack_of.get(i, "-") for i in all_sel]
sel_label  = [0 if sysids[i] == "-" else 1 for i in all_sel]

# ═══════════════════════════════════════════════════════════════════════════════
# 2. WavLM per-utterance C/T (frozen microsoft/wavlm-base)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2] WavLM C/T extraction ...", flush=True)
from transformers import WavLMForCTC
wavlm = WavLMForCTC.from_pretrained("microsoft/wavlm-base").wavlm.eval().to(DEVICE)

@torch.no_grad()
def wavlm_ct(wav_batch):
    hs = wavlm(wav_batch.to(DEVICE), output_hidden_states=True).hidden_states
    h9, h12 = hs[9].float().cpu().numpy(), hs[12].float().cpu().numpy()
    return ([rog(h12[i]) for i in range(len(h12))],
            [vel_entropy(h9[i]) for i in range(len(h9))])

wavs = []
print("  caching + cropping waveforms ...", flush=True)
for i in range(len(test_sel)):
    arr = test_sel[i]["audio"]["array"]
    wavs.append(_center_crop_pad(torch.tensor(arr, dtype=torch.float32)))

rogs, vels = [], []
for b in range(0, len(wavs), BATCH_SIZE):
    wb = torch.stack(wavs[b:b + BATCH_SIZE])
    r, v = wavlm_ct(wb)
    rogs.extend(r); vels.extend(v)
    if (b // BATCH_SIZE) % 20 == 0:
        print(f"    wavlm {b}/{len(wavs)}", flush=True)
rogs = np.array(rogs); vels = np.array(vels)
C = -rogs          # higher = more compact = harder
T = vels
del wavlm
torch.cuda.empty_cache()

utt = pd.DataFrame({"attack": sel_attack, "label": sel_label,
                    "C": C, "T": T, "rog_L12": rogs, "vel_L9": vels})
print(f"  extracted C/T for {len(utt)} utterances")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Model inference -> per-utterance logits (batched), both checkpoints
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3] Detector inference ...", flush=True)
def patch_phoneme_loader():
    import phoneme_GAT.modules as mm
    import phoneme_GAT.phoneme_model as pm
    from phoneme_GAT.phoneme_model import BaseModule, network_param, optim_param
    def _load(network_name="wavlm", pretrained_path=None, total_num_phonemes=198):
        network_param.network_name = network_name
        network_param.pretrained_name = "microsoft/wavlm-base"
        network_param.vocab_size = total_num_phonemes
        return BaseModule(network_param, optim_param, tokenizer=None,
                          total_num_phonemes=total_num_phonemes)
    pm.load_phoneme_model = _load
    mm.load_phoneme_model = _load

def _detect_n_edges(ckpt_path):
    ck = torch.load(str(ckpt_path), weights_only=False, map_location="cpu")
    cfg = ck.get("hyper_parameters", {}).get("cfg", None)
    n = getattr(getattr(cfg, "PhonemeGAT", None), "n_edges", None) if cfg else None
    return int(n) if n is not None else 10

def load_model(ckpt_path):
    from phoneme_GAT.modules import Phoneme_GAT_lit
    cfg = Namespace(PhonemeGAT=Namespace(backbone="wavlm", use_raw=False, use_GAT=True,
                    n_edges=_detect_n_edges(ckpt_path), use_aug=True, use_pool=True, use_clip=True))
    lit = Phoneme_GAT_lit.load_from_checkpoint(str(ckpt_path), cfg=cfg,
                                               map_location=DEVICE, strict=True)
    lit.to(DEVICE); lit.eval(); lit.freeze()
    return lit

patch_phoneme_loader()
try:
    from pandas import Series as _PS
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination as _PTR
    torch.serialization.add_safe_globals([Namespace, _PS, _PTR])
except Exception:
    torch.serialization.add_safe_globals([Namespace])

@torch.no_grad()
def infer_logits(lit):
    gm = lit.model
    out = []
    for b in range(0, len(wavs), BATCH_SIZE):
        wb = torch.stack(wavs[b:b + BATCH_SIZE]).to(DEVICE)
        nf = torch.full((wb.shape[0],), NF_PER_SAMPLE, device=DEVICE)
        res = gm(wb, nf, use_aug=False, stage="val")
        out.extend(res["logit"].cpu().numpy().tolist())
    return np.array(out)

for name, ckpt in MODELS.items():
    if not Path(ckpt).exists():
        print(f"  [skip] {name}"); continue
    print(f"  {name} ...", flush=True)
    lit = load_model(ckpt)
    utt[f"logit_{name}"] = infer_logits(lit)
    del lit
    torch.cuda.empty_cache()
utt.to_csv(OUTDIR / "asvspoof_utterance_ct.csv", index=False)
print(f"  saved asvspoof_utterance_ct.csv ({len(utt)} rows)")

model_names = [n for n in MODELS if f"logit_{n}" in utt.columns]
spoof = utt[utt.label == 1]
bona  = utt[utt.label == 0]

# ═══════════════════════════════════════════════════════════════════════════════
# Stats helpers
# ═══════════════════════════════════════════════════════════════════════════════
def corr_ci(x, y, method="spearman", nboot=5000, rng=None):
    rng = rng or np.random.default_rng(SEED)
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]; n = len(x)
    f = stats.spearmanr if method == "spearman" else stats.pearsonr
    r0, p0 = (float(v) for v in f(x, y))
    bs = np.array([f(x[i], y[i])[0] for i in (rng.integers(0, n, n) for _ in range(nboot))])
    return r0, p0, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)), n

def perm_p(x, y, method="spearman", nperm=10000, rng=None):
    rng = rng or np.random.default_rng(SEED)
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    f = stats.spearmanr if method == "spearman" else stats.pearsonr
    r0 = f(x, y)[0]
    cnt = sum(abs(f(rng.permutation(x), y)[0]) >= abs(r0) for _ in range(nperm))
    return float(r0), float((cnt + 1) / (nperm + 1))

def ols_r2(X, y):
    X = np.atleast_2d(X);
    if X.shape[0] != len(y): X = X.T
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1); X, y = X[m], y[m]; n = len(y)
    A = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ beta
    sst = float(((y - y.mean()) ** 2).sum()); ssr = float((e ** 2).sum())
    r2 = 1 - ssr / sst if sst > 0 else np.nan
    try:
        h = np.clip(np.diag(A @ np.linalg.pinv(A.T @ A) @ A.T), 0, 1 - 1e-9)
        loo = 1 - float(((e / (1 - h)) ** 2).sum()) / sst if sst > 0 else np.nan
    except Exception:
        loo = np.nan
    return r2, loo

# ═══════════════════════════════════════════════════════════════════════════════
# 4. PER-ATTACK analysis (n=13)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4] Per-attack analysis (n=13) ...", flush=True)
bona_logits = {n: bona[f"logit_{n}"].values for n in model_names}
rows = []
for a in EVAL_ATTACKS:
    s = spoof[spoof.attack == a]
    row = {"attack": a, "n_spoof": len(s), "C": s["C"].mean(), "T": s["T"].mean()}
    for n in model_names:
        lab = np.r_[np.ones(len(s)), np.zeros(len(bona))]
        sco = np.r_[s[f"logit_{n}"].values, bona_logits[n]]
        mt = _all_metrics(lab, sco)
        row[f"EER_{n}"] = mt["eer"]; row[f"AUC_{n}"] = mt["auc"]; row[f"Acc_{n}"] = mt["acc"]
    rows.append(row)
attack_df = pd.DataFrame(rows)
attack_df.to_csv(OUTDIR / "asvspoof_per_attack.csv", index=False)
print(attack_df[["attack", "C", "T"] + [f"EER_{HEADLINE}"]].to_string(index=False))

Cv, Tv = attack_df["C"].values, attack_df["T"].values
corr_rows = []
for n in model_names:
    for metric in ["EER", "AUC", "Acc"]:
        y = attack_df[f"{metric}_{n}"].values
        for axis, xv in [("C", Cv), ("T", Tv)]:
            sr, sp, slo, shi, nn = corr_ci(xv, y, "spearman")
            pr, pp, plo, phi, _ = corr_ci(xv, y, "pearson")
            _, permp = perm_p(xv, y, "spearman")
            corr_rows.append({"model": n, "metric": metric, "axis": axis, "n": nn,
                              "spearman": sr, "spearman_p": sp, "spearman_ci_lo": slo, "spearman_ci_hi": shi,
                              "spearman_perm_p": permp, "pearson": pr, "pearson_p": pp})
        r2, loo = ols_r2(np.column_stack([Cv, Tv]), y)
        corr_rows.append({"model": n, "metric": metric, "axis": "C+T", "n": len(Cv), "r2": r2, "loo_r2": loo})
corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(OUTDIR / "asvspoof_correlations.csv", index=False)

# MLAAD reference (per-system C/T -> known mechanism). Compute MLAAD ρ(C/T, residual) sign.
mlaad_ref = ""
if SIG_FEATURES.exists():
    sig = pd.read_csv(SIG_FEATURES)
    if {"rog_L12", "vel_entropy_L9", "residual"}.issubset(sig.columns):
        mC = stats.spearmanr(-sig["rog_L12"], sig["residual"]).correlation
        mT = stats.spearmanr(sig["vel_entropy_L9"], sig["residual"]).correlation
        mlaad_ref = f"MLAAD per-system reference (n={len(sig)}): ρ(C,resid)={mC:+.2f}, ρ(T,resid)={mT:+.2f}"
        print(f"  {mlaad_ref}")

# Quartile stratification (n=13 -> small; report Q1/Q4 contrast)
ct_attack = {r["attack"]: {"C": r["C"], "T": r["T"]} for _, r in attack_df.iterrows()}
def quart(axis, n):
    buckets = _quartile_buckets(list(ct_attack), ct_attack, axis)
    eer = {r["attack"]: r[f"EER_{n}"] for _, r in attack_df.iterrows()}
    return [float(np.mean([eer[a] for a in b])) if b else np.nan for b in buckets]
q_summary = {axis: quart(axis, HEADLINE) for axis in ["C", "T"]}

# ═══════════════════════════════════════════════════════════════════════════════
# 5. PER-UTTERANCE (high power) + between/within-attack decomposition
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5] Per-utterance + decomposition ...", flush=True)
utt_rows, decomp_rows = [], []
for n in model_names:
    lg = spoof[f"logit_{n}"].values
    for axis, xv in [("C", spoof["C"].values), ("T", spoof["T"].values)]:
        sr, sp, slo, shi, nn = corr_ci(xv, lg, "spearman", nboot=2000)
        utt_rows.append({"model": n, "axis": axis, "n": nn, "spearman_logit": sr,
                         "p": sp, "ci_lo": slo, "ci_hi": shi})
    # between vs within attack on C
    g = spoof.groupby("attack")
    mC = g["C"].transform("mean"); mL = g[f"logit_{n}"].transform("mean")
    between = stats.spearmanr(g["C"].mean(), g[f"logit_{n}"].mean()).correlation
    within = stats.spearmanr((spoof["C"] - mC).values, (spoof[f"logit_{n}"] - mL).values).correlation
    pooled = stats.spearmanr(spoof["C"].values, lg).correlation
    decomp_rows.append({"model": n, "pooled_rho_C_logit": pooled,
                        "between_attack": between, "within_attack": within})
utt_corr_df = pd.DataFrame(utt_rows); utt_corr_df.to_csv(OUTDIR / "asvspoof_utterance_corr.csv", index=False)
decomp_df = pd.DataFrame(decomp_rows); decomp_df.to_csv(OUTDIR / "asvspoof_decomposition.csv", index=False)
# ICC of C between attacks (spoof)
grand = spoof["C"].mean()
ssb = sum(len(g) * (g["C"].mean() - grand) ** 2 for _, g in spoof.groupby("attack"))
sst = float(((spoof["C"] - grand) ** 2).sum())
icc_C = float(ssb / sst) if sst > 0 else np.nan

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Figures
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[6] Figures ...", flush=True)
# Fig 1: per-attack C/T vs EER (headline model), labeled
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, axis, xv in [(axes[0], "C", Cv), (axes[1], "T", Tv)]:
    y = attack_df[f"EER_{HEADLINE}"].values
    ax.scatter(xv, y, s=60, c="#d62728", edgecolors="k", linewidths=0.4, zorder=3)
    for i, a in enumerate(attack_df["attack"]):
        ax.annotate(a, (xv[i], y[i]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    b1, b0 = np.polyfit(xv, y, 1); xs = np.linspace(xv.min(), xv.max(), 50)
    ax.plot(xs, b1 * xs + b0, "k--", lw=1)
    rr = corr_df[(corr_df.model == HEADLINE) & (corr_df.metric == "EER") & (corr_df.axis == axis)].iloc[0]
    ax.set_xlabel(f"{axis} = {'-rog@L12' if axis=='C' else 'vel_entropy@L9'} (higher=harder)")
    ax.set_ylabel(f"per-attack EER ({HEADLINE})")
    ax.set_title(f"{axis}: ρ={rr['spearman']:+.2f} [{rr['spearman_ci_lo']:+.2f},{rr['spearman_ci_hi']:+.2f}] "
                 f"perm p={rr['spearman_perm_p']:.2g}")
plt.suptitle("E8 ASVspoof: per-attack C/T vs EER (A07–A19, n=13)", fontweight="bold")
plt.tight_layout(); plt.savefig(OUTDIR / "asvspoof_per_attack_ct_eer.png", dpi=150); plt.close()

# Fig 2: per-utterance logit vs C/T (headline), hexbin
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
lg = spoof[f"logit_{HEADLINE}"].values
for ax, axis, xv in [(axes[0], "C", spoof["C"].values), (axes[1], "T", spoof["T"].values)]:
    hb = ax.hexbin(xv, lg, gridsize=40, cmap="viridis", mincnt=1)
    b1, b0 = np.polyfit(xv, lg, 1); xs = np.linspace(xv.min(), xv.max(), 50)
    ax.plot(xs, b1 * xs + b0, "r--", lw=1.5)
    rr = utt_corr_df[(utt_corr_df.model == HEADLINE) & (utt_corr_df.axis == axis)].iloc[0]
    ax.set_xlabel(f"{axis}"); ax.set_ylabel(f"spoof raw logit ({HEADLINE})")
    ax.set_title(f"{axis}: ρ={rr['spearman_logit']:+.3f} (p={rr['p']:.1e}, n={rr['n']})")
plt.suptitle("E8 ASVspoof: per-utterance C/T vs logit (spoof; high power)", fontweight="bold")
plt.tight_layout(); plt.savefig(OUTDIR / "asvspoof_per_utterance_ct_logit.png", dpi=150); plt.close()
print("  figures saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Summary
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[7] Summary ...", flush=True)
def gv(model, metric, axis, col):
    s = corr_df[(corr_df.model == model) & (corr_df.metric == metric) & (corr_df.axis == axis)]
    return s[col].values[0] if len(s) and col in s else np.nan

# Logit convention here (verified): higher logit = more spoof-like = correctly flagged =
# EASIER. So on spoof, ρ(C/T, logit) > 0 => higher C/T => easier; MLAAD's spoof ρ(C,logit)
# is ≈ -0.15 (higher C => harder). We judge each axis by the HIGH-POWER per-utterance test
# (n~3900) and report DIRECTION vs MLAAD, rather than a single underpowered n=13 sign test.
def axis_predictive(axis, alpha=0.05):
    s = utt_corr_df[(utt_corr_df.model == HEADLINE) & (utt_corr_df.axis == axis)]
    if not len(s):
        return False, 0.0
    rho = float(s["spearman_logit"].values[0]); p = float(s["p"].values[0])
    # require both significance AND a non-trivial effect (|ρ| >= 0.10) to call it predictive
    return (p < alpha and abs(rho) >= 0.10), rho

c_pred, c_rho = axis_predictive("C")
t_pred, t_rho = axis_predictive("T")
# MLAAD spoof reference sign for C is negative (higher C -> harder). Reversed if signs differ.
MLAAD_C_LOGIT_SIGN = -1.0
c_reversed = c_pred and (np.sign(c_rho) != np.sign(MLAAD_C_LOGIT_SIGN))
def axis_phrase(pred, rho, reversed_):
    if not pred:
        return "not predictive"
    return ("predictive, REVERSED vs MLAAD" if reversed_ else "predictive, same sign as MLAAD")
verdict = (f"C={axis_phrase(c_pred, c_rho, c_reversed)}; "
           f"T={axis_phrase(t_pred, t_rho, False)}")
c_sup, t_sup = c_pred, t_pred

L = ["# E8 ASVspoof — Does C/T predict per-attack detection hardness?", "",
     f"ASVspoof 2019 LA **eval split A07–A19** (13 unseen attacks), "
     f"{N_PER_ATTACK}/attack spoof + {len(bona)} bona-fide. C/T from frozen WavLM "
     "(microsoft/wavlm-base, L12/L9). Per-attack EER scored vs the shared bona-fide pool.",
     f"Headline detector: **{HEADLINE}**. (robust_GOAT also run, but it was trained on "
     "ASVspoof train+test so it has *seen* A07–A19 — its per-attack EER is partly "
     "in-distribution and reported only as a cross-check.)",
     "",
     f"## Verdict (high-power per-utterance, {HEADLINE}): {verdict}",
     "_Logit convention: higher logit = more spoof-like = easier. ρ(C,logit)>0 ⇒ compact "
     "spoofs are EASIER; MLAAD spoof ρ(C,logit)≈−0.15 ⇒ compact HARDER, so a positive ρ here "
     "is a polarity REVERSAL, not a null._", "",
     "## 1. Per-attack correlations (Spearman ρ, 95% bootstrap CI, permutation p)",
     "| model | metric | axis | ρ | CI | perm p | Pearson r | R²(C+T) | LOO |",
     "|---|---|---|---|---|---|---|---|---|"]
for n in model_names:
    for metric in ["EER", "AUC", "Acc"]:
        for axis in ["C", "T"]:
            rho = gv(n, metric, axis, "spearman"); lo = gv(n, metric, axis, "spearman_ci_lo")
            hi = gv(n, metric, axis, "spearman_ci_hi"); pp = gv(n, metric, axis, "spearman_perm_p")
            pr = gv(n, metric, axis, "pearson")
            r2 = gv(n, metric, "C+T", "r2"); loo = gv(n, metric, "C+T", "loo_r2")
            L.append(f"| {n} | {metric} | {axis} | {rho:+.2f} | [{lo:+.2f},{hi:+.2f}] | {pp:.2g} | "
                     f"{pr:+.2f} | {r2:.2f} | {loo:.2f} |")
if mlaad_ref:
    L += ["", f"_{mlaad_ref} (sign reference: in MLAAD higher C/T → harder → ρ>0 with EER/residual)._"]

L += ["", "## 2. Quartile stratification (headline EER; Q4 = highest C/T = predicted hardest)",
      "| axis | Q1 | Q2 | Q3 | Q4 | Q4>Q1? |", "|---|---|---|---|---|---|"]
for axis in ["C", "T"]:
    q = q_summary[axis]
    flag = "yes" if (np.isfinite(q[3]) and np.isfinite(q[0]) and q[3] > q[0]) else "no"
    L.append(f"| {axis} | " + " | ".join(f"{v:.3f}" for v in q) + f" | {flag} |")

L += ["", "## 3. Per-utterance (high power) C/T → logit on spoof",
      "Sign note (verified): higher logit = more spoof-like = correctly flagged = EASIER. "
      "So ρ(C/T, logit) > 0 ⇒ higher C/T → easier to detect. MLAAD spoof ρ(C,logit)≈−0.15 "
      "(higher C → harder), so a positive ρ here is a REVERSAL of the MLAAD polarity.",
      "| model | axis | ρ(C/T, logit) | CI | p | n |", "|---|---|---|---|---|---|"]
for _, r in utt_corr_df.iterrows():
    L.append(f"| {r['model']} | {r['axis']} | {r['spearman_logit']:+.3f} | "
             f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] | {r['p']:.1e} | {int(r['n'])} |")

L += ["", "## 4. Between- vs within-attack decomposition (C → logit)",
      "| model | pooled | between-attack | within-attack |", "|---|---|---|---|"]
for _, r in decomp_df.iterrows():
    L.append(f"| {r['model']} | {r['pooled_rho_C_logit']:+.3f} | {r['between_attack']:+.3f} | "
             f"{r['within_attack']:+.3f} |")
L += ["", f"- C variance between-attack (ICC-like): **{icc_C:.3f}**", ""]

# interpretation
head_C = gv(HEADLINE, "EER", "C", "spearman"); head_T = gv(HEADLINE, "EER", "T", "spearman")
L += ["## Interpretation",
      f"- **Per-attack (n=13):** ρ(C,EER)={head_C:+.2f}, ρ(T,EER)={head_T:+.2f} ({HEADLINE}). "
      f"At n=13 the CIs are wide; treat per-attack as {'significant' if (c_sup or t_sup) else 'suggestive only'}.",
      f"- **Per-utterance:** the high-power test ({len(spoof)} spoof utts) is the reliable "
      "read on whether the mechanism exists at all in ASVspoof.",
      "- **Decomposition** shows whether any per-attack signal is a genuine per-utterance "
      "effect (within-attack ρ≠0) or just 13 attack means aligning (between-only).",
      "", "## Files",
      "- `asvspoof_per_attack.csv`, `asvspoof_correlations.csv`, `asvspoof_utterance_ct.csv`",
      "- `asvspoof_utterance_corr.csv`, `asvspoof_decomposition.csv`",
      "- figures: `asvspoof_per_attack_ct_eer.png`, `asvspoof_per_utterance_ct_logit.png`"]
(OUTDIR / "asvspoof_summary.md").write_text("\n".join(L))
print(f"  summary -> {OUTDIR/'asvspoof_summary.md'}")
print("\n[E8 ASVspoof] done.")
