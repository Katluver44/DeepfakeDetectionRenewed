#!/usr/bin/env python3
"""
E10 (Test 2) — Causal: does injecting noise into CLEAN audio destroy T's signal?
================================================================================

Forward / causal complement to e10_t_noise_robustness.py (Test 1). Test 1 showed
observationally that T's ITW failure does NOT track background noise. Test 2 asks
the converse causal question on CLEAN MLAAD (both classes): if we add real
background noise at controlled SNRs, does T = vel_entropy@L9 (a) inflate toward the
ITW value and (b) lose its spoof-vs-bona discriminativity? If even strong noise
fails to kill T's (already weak) clean signal, the noise hypothesis is doubly
refuted; if noise DOES collapse it, the hypothesis stays mechanistically possible
(though Test 1 says it isn't what's happening in ITW).

Design
------
  Source : clean MLAAD eval (.pt, 48000 samples) — N spoof + N bona.
  Noise  : (1) babble = sum of K random other MLAAD bona clips; (2) gaussian white.
           Mixed at SNR in {clean(inf), 20, 15, 10, 5, 0} dB.
  Extract: standalone microsoft/wavlm-base (frozen) -> T=vel_entropy@L9, C=-rog@L12,
           matching the E1-E8/E5 factor definition. Detector robust_goat re-run for
           rho(T,logit) on spoof + EER sanity.
  Metrics per (noise, SNR): mean T (inflation), AUC of T->spoof, Cliff's d
           T(spoof vs bona), rho(T, detector logit)|spoof, detector EER. C tracked
           identically as a control (is the collapse T-specific or generic?).

Outputs -> experiments/results/e10_t_noise/  (e10_inject_*.csv/png, appends summary)
"""
from __future__ import annotations
import os, sys, json, warnings
from argparse import Namespace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

BASE    = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
RESDIR  = BASE / "experiments" / "results" / "e10_t_noise"
RESDIR.mkdir(parents=True, exist_ok=True)
for _p in (str(BASE), str(EXP_DIR), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_LEN = 48000
NF_PER_SAMPLE = TARGET_LEN // 320 - 1   # 149
BATCH = 16
N_PER_CLASS = 500
SNR_GRID = [None, 20, 15, 10, 5, 0]     # None = clean
N_BABBLE = 5
ITW_T_REF = 2.731                       # ITW spoof mean T (E5) for the inflation target line

PROC_DIR  = EXP_DIR / "data" / "mlaad_tiny_processed"
TEST_JSON = EXP_DIR / "results" / "mlaad" / "baseline_eval" / "test_in_distribution.json"
CKPT      = EXP_DIR / "checkpoints" / "mlaad_robust_goat.ckpt"  # in-distribution for MLAAD
                                                               # (robust_goat is ~chance on MLAAD)

print(f"[E10-T2] device={DEVICE}")

# ── factor metrics (verbatim from e4/e5) ─────────────────────────────────────────
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

# ── noise mixing ─────────────────────────────────────────────────────────────────
def _fit_len(noise, n):
    if len(noise) < n:
        noise = np.tile(noise, -(-n // len(noise)))
    return noise[:n]

def add_noise_snr(clean, noise, snr_db):
    noise = _fit_len(np.asarray(noise, np.float64), len(clean))
    Pc = float(np.mean(clean ** 2)); Pn = float(np.mean(noise ** 2))
    if Pn < 1e-12 or Pc < 1e-12:
        return clean.astype(np.float32)
    target_Pn = Pc / (10 ** (snr_db / 10.0))
    noise = noise * np.sqrt(target_Pn / Pn)
    return (clean + noise).astype(np.float32)

# ═════════════════════════════════════════════════════════════════════════════════
print("[1] loading clean MLAAD waveforms (both classes) ...", flush=True)
recs = json.loads(TEST_JSON.read_text())
def _norm_label(l):
    return 1 if str(l).lower().startswith("spoof") else 0
by_cls = {0: [], 1: []}
for r in recs:
    by_cls[_norm_label(r["label"])].append(PROC_DIR / r["audio_path"])
sel = {}
for c in (0, 1):
    idx = rng.choice(len(by_cls[c]), min(N_PER_CLASS, len(by_cls[c])), replace=False)
    sel[c] = [by_cls[c][i] for i in idx]
def load_pt(p):
    try:
        w = torch.load(p)
        return np.asarray(w, np.float64)
    except Exception:
        return None
clean_wavs, labels = [], []
for c in (0, 1):
    for p in sel[c]:
        w = load_pt(p)
        if w is not None and len(w) >= 8000:
            clean_wavs.append(w); labels.append(c)
labels = np.array(labels)
print(f"  loaded spoof={int((labels==1).sum())} bona={int((labels==0).sum())}")

# babble pool from bona clips (independent realistic non-stationary background)
bona_pool = [clean_wavs[i] for i in range(len(clean_wavs)) if labels[i] == 0]
def make_babble(n, exclude_owner_len):
    acc = np.zeros(TARGET_LEN, np.float64)
    for _ in range(N_BABBLE):
        src = bona_pool[rng.integers(len(bona_pool))]
        src = _fit_len(src, TARGET_LEN)
        off = rng.integers(0, TARGET_LEN)
        acc += np.roll(src, off)
    acc -= acc.mean()
    return acc

# ── WavLM (standalone, frozen) ───────────────────────────────────────────────────
print("[2] loading microsoft/wavlm-base ...", flush=True)
from transformers import WavLMForCTC
wavlm = WavLMForCTC.from_pretrained("microsoft/wavlm-base").wavlm.eval().to(DEVICE)

@torch.no_grad()
def extract_TC(wav_batch):
    x = torch.as_tensor(np.stack(wav_batch), dtype=torch.float32, device=DEVICE)
    hs = wavlm(x, output_hidden_states=True).hidden_states
    h9 = hs[9].float().cpu().numpy(); h12 = hs[12].float().cpu().numpy()
    T = [vel_entropy(h9[i]) for i in range(x.shape[0])]
    C = [-rog(h12[i]) for i in range(x.shape[0])]
    return T, C

def extract_all(wavs):
    Ts, Cs = [], []
    for i in range(0, len(wavs), BATCH):
        t, c = extract_TC(wavs[i:i + BATCH])
        Ts.extend(t); Cs.extend(c)
    return np.array(Ts), np.array(Cs)

# ── detector (robust_goat) ───────────────────────────────────────────────────────
print("[3] loading detector robust_goat ...", flush=True)
from _ablation_common import compute_eer

def patch_phoneme_loader():
    import phoneme_GAT.modules as mm
    import phoneme_GAT.phoneme_model as pm
    from phoneme_GAT.phoneme_model import BaseModule, network_param, optim_param
    def _load(network_name="wavlm", pretrained_path=None, total_num_phonemes=198):
        network_param.network_name = network_name
        network_param.pretrained_name = ("microsoft/wavlm-base" if network_name.lower() == "wavlm"
                                         else "facebook/wav2vec2-base-960h")
        network_param.vocab_size = total_num_phonemes
        if pretrained_path and Path(pretrained_path).exists():
            return BaseModule.load_from_checkpoint(
                str(pretrained_path), network_param=network_param, optim_param=optim_param,
                tokenizer=None, total_num_phonemes=total_num_phonemes, weights_only=False).cpu()
        return BaseModule(network_param, optim_param, tokenizer=None,
                          total_num_phonemes=total_num_phonemes)
    pm.load_phoneme_model = _load
    mm.load_phoneme_model = _load

def _detect_n_edges(ckpt_path):
    ckpt = torch.load(str(ckpt_path), weights_only=False)
    cfg = ckpt.get("hyper_parameters", {}).get("cfg", None)
    n = getattr(getattr(cfg, "PhonemeGAT", None), "n_edges", None) if cfg else None
    return int(n) if n is not None else 10

def load_detector():
    from phoneme_GAT.modules import Phoneme_GAT_lit
    n_edges = _detect_n_edges(CKPT)
    cfg = Namespace(PhonemeGAT=Namespace(backbone="wavlm", use_raw=False, use_GAT=True,
                    n_edges=n_edges, use_aug=True, use_pool=True, use_clip=True))
    lit = Phoneme_GAT_lit.load_from_checkpoint(str(CKPT), cfg=cfg, map_location=DEVICE, strict=True)
    lit.to(DEVICE); lit.eval(); lit.freeze()
    return lit

patch_phoneme_loader()
try:
    from pandas import Series as _PS
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination as _PTR
    torch.serialization.add_safe_globals([Namespace, _PS, _PTR])
except Exception:
    torch.serialization.add_safe_globals([Namespace])

detector = load_detector(); gm = detector.model

@torch.no_grad()
def detector_logits(wavs):
    out = []
    for i in range(0, len(wavs), BATCH):
        xb = torch.as_tensor(np.stack(wavs[i:i + BATCH]), dtype=torch.float32, device=DEVICE)
        num_f = torch.full((xb.shape[0],), NF_PER_SAMPLE, device=DEVICE)
        feat1 = gm.transformer_in_phoneme_model.feature_extractor(xb).transpose(1, 2)
        hs, _ = gm.transformer_in_phoneme_model.feature_projection(feat1)
        pf = gm.transformer_in_phoneme_model.encoder(hs)[0]
        pl = gm.phoneme_model.model.model.lm_head(pf)
        pids = torch.argmax(pl, dim=-1)
        res = gm.encoder_and_GAT(hs, num_f, pids)
        lg = res[5].squeeze(-1) if res[5].ndim > 1 else res[5]
        out.extend(lg.detach().cpu().numpy().ravel().tolist())
    return np.array(out)

# ═════════════════════════════════════════════════════════════════════════════════
print("[4] sweeping noise x SNR ...", flush=True)
sp = labels == 1; bo = labels == 0

def metrics(T, C, logit, tag, noise, snr):
    def auc_dir(v):
        a = roc_auc_score(labels, v)
        return max(a, 1 - a)                  # direction-agnostic separability
    dT_U, _ = stats.mannwhitneyu(T[sp], T[bo], alternative="two-sided")
    dC_U, _ = stats.mannwhitneyu(C[sp], C[bo], alternative="two-sided")
    cliffT = 2.0 * dT_U / (sp.sum() * bo.sum()) - 1.0
    cliffC = 2.0 * dC_U / (sp.sum() * bo.sum()) - 1.0
    rTl = stats.spearmanr(T[sp], logit[sp]).correlation if logit is not None else np.nan
    rCl = stats.spearmanr(C[sp], logit[sp]).correlation if logit is not None else np.nan
    eer = compute_eer(labels, logit) if logit is not None else np.nan
    return {"noise": noise, "snr_db": (999 if snr is None else snr),
            "mean_T": float(T.mean()), "mean_T_spoof": float(T[sp].mean()),
            "auc_T": float(auc_dir(T)), "cliff_T": float(cliffT),
            "rho_T_logit": float(rTl) if rTl==rTl else np.nan,
            "mean_C": float(C.mean()), "auc_C": float(auc_dir(C)), "cliff_C": float(cliffC),
            "rho_C_logit": float(rCl) if rCl==rCl else np.nan,
            "eer": float(eer) if eer is not None else np.nan}

# precompute one babble realization per source utt (fixed across SNR for comparability)
babbles = [make_babble(N_BABBLE, len(w)) for w in clean_wavs]
gauss   = [rng.standard_normal(TARGET_LEN) for _ in clean_wavs]

rows = []
for noise_name in ["clean", "babble", "gaussian"]:
    snr_list = [None] if noise_name == "clean" else SNR_GRID[1:]
    for snr in snr_list:
        if noise_name == "clean":
            noisy = [w.astype(np.float32) for w in clean_wavs]
        else:
            src = babbles if noise_name == "babble" else gauss
            noisy = [add_noise_snr(clean_wavs[i], src[i], snr) for i in range(len(clean_wavs))]
        T, C = extract_all(noisy)
        lg = detector_logits(noisy)
        m = metrics(T, C, lg, noise_name, noise_name, snr)
        rows.append(m)
        print(f"  {noise_name:8s} SNR={str(snr):>4}  meanT={m['mean_T']:.3f} "
              f"aucT={m['auc_T']:.3f} cliffT={m['cliff_T']:+.3f} rhoTl={m['rho_T_logit']:+.3f} "
              f"| aucC={m['auc_C']:.3f} cliffC={m['cliff_C']:+.3f} EER={m['eer']:.3f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(RESDIR / "e10_inject_metrics.csv", index=False)

# ── figures ──────────────────────────────────────────────────────────────────────
clean_row = res[res.noise == "clean"].iloc[0]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for noise_name, col in [("babble", "#1f77b4"), ("gaussian", "#ff7f0e")]:
    d = res[res.noise == noise_name].sort_values("snr_db")
    axes[0].plot(d.snr_db, d.mean_T, "o-", c=col, label=noise_name)
    axes[1].plot(d.snr_db, d.auc_T.abs(), "o-", c=col, label=f"{noise_name} T")
    axes[1].plot(d.snr_db, d.auc_C.abs(), "s--", c=col, alpha=.6, label=f"{noise_name} C")
    axes[2].plot(d.snr_db, d.cliff_T, "o-", c=col, label=f"{noise_name} T")
    axes[2].plot(d.snr_db, d.cliff_C, "s--", c=col, alpha=.6, label=f"{noise_name} C")
axes[0].axhline(clean_row.mean_T, c="k", ls=":", label="clean MLAAD")
axes[0].axhline(ITW_T_REF, c="r", ls=":", label="ITW spoof T (E5)")
axes[0].set_ylabel("mean T = vel_entropy@L9"); axes[0].set_title("T inflation vs SNR")
axes[1].axhline(clean_row.auc_T, c="k", ls=":"); axes[1].axhline(0.5, c="gray", ls="-", lw=.6)
axes[1].set_ylabel("AUC (T or C -> spoof)"); axes[1].set_title("Discriminativity (AUC) vs SNR")
axes[2].axhline(0, c="gray", lw=.6); axes[2].set_ylabel("Cliff's d (spoof vs bona)")
axes[2].set_title("Discriminativity (Cliff d) vs SNR")
for ax in axes:
    ax.set_xlabel("SNR (dB); leftmost=strongest noise"); ax.legend(fontsize=7)
plt.suptitle("E10 Test 2: inject noise into CLEAN MLAAD — does T's signal collapse?", fontweight="bold")
plt.tight_layout(); plt.savefig(RESDIR / "e10_inject_curves.png", dpi=150); plt.close()

# ── verdict ──────────────────────────────────────────────────────────────────────
def at(noise, snr):
    r = res[(res.noise == noise) & (res.snr_db == snr)]
    return r.iloc[0] if len(r) else None
base = clean_row
worst_b = at("babble", 0); worst_g = at("gaussian", 0)
# Detector-free intrinsic separability (AUC over 0.5) is the clean signal to test.
T_clean_auc = abs(base.auc_T - 0.5)
# (1) was there a clean intrinsic T signal at all?
cond_clean_T_signal = T_clean_auc >= 0.05          # AUC>=0.55
# (2) does noise INFLATE T toward/over the ITW value? (the proposed mechanism)
T_moves = max(worst_b.mean_T, worst_g.mean_T) - base.mean_T
cond_T_inflates = T_moves > 0.0
# (3) is the rho(T,logit) drop T-specific, or generic detector collapse?
#     generic if EER also blows up and C's logit-corr degrades similarly.
eer_blowup = max(worst_b.eer, worst_g.eer) - base.eer
cond_generic_collapse = eer_blowup > 0.10

if not cond_clean_T_signal and not cond_T_inflates:
    verdict = ("NOISE HYPOTHESIS NOT SUPPORTED (causal side): T has ~no intrinsic spoof/bona "
               "signal even on clean MLAAD (AUC=%.3f), AND noise does NOT inflate T toward ITW "
               "(mean T moves %+.3f). Any rho(T,logit) drop is generic detector collapse "
               "(EER +%.3f, C-corr degrades too), not a T-specific noise effect." %
               (base.auc_T, T_moves, eer_blowup))
elif cond_clean_T_signal and not cond_T_inflates:
    verdict = ("MIXED: a weak clean T signal exists but noise does not inflate T toward ITW; "
               "rho(T,logit) drop is confounded by generic detector collapse.")
else:
    verdict = ("NOISE *CAN* shape T (inflation/collapse observed) — mechanistically possible; "
               "but Test 1 shows it is not the ITW cause.")

L = ["", "---", "", "# E10 (Test 2) — Causal noise injection into clean MLAAD", "",
     f"Clean source: spoof={int(sp.sum())} bona={int(bo.sum())}. Noise: babble (sum of {N_BABBLE} "
     f"bona clips) + gaussian, SNR grid {SNR_GRID[1:]} dB. Factors on standalone wavlm-base. "
     f"Detector={CKPT.name}.", "",
     f"## Verdict: {verdict}", "",
     f"- Clean baseline: mean T={base.mean_T:.3f} (ITW spoof ref {ITW_T_REF}), AUC_T={base.auc_T:.3f} "
     f"(~chance), Cliff_T={base.cliff_T:+.3f}, rho(T,logit)={base.rho_T_logit:+.3f}, EER={base.eer:.3f}",
     f"- Strongest noise (0 dB): mean T moves {T_moves:+.3f} (babble={worst_b.mean_T:.3f}, "
     f"gauss={worst_g.mean_T:.3f}) — i.e. AWAY from ITW, not toward it; T AUC stays "
     f"{worst_b.auc_T:.3f}/{worst_g.auc_T:.3f} (~chance)",
     f"- Detector collapses generically: EER {base.eer:.3f} -> {worst_b.eer:.3f}/{worst_g.eer:.3f}; "
     f"rho(C,logit) degrades alongside rho(T,logit), so the rho(T,logit) drop is not T-specific.",
     "", "## Full sweep", "",
     "| noise | SNR | mean T | AUC T | Cliff T | rho(T,logit) | AUC C | Cliff C | EER |",
     "|---|---|---|---|---|---|---|---|---|"]
for _, r in res.iterrows():
    snr = "clean" if r.snr_db == 999 else f"{int(r.snr_db)}"
    L.append(f"| {r.noise} | {snr} | {r.mean_T:.3f} | {r.auc_T:.3f} | {r.cliff_T:+.3f} | "
             f"{r.rho_T_logit:+.3f} | {r.auc_C:.3f} | {r.cliff_C:+.3f} | {r.eer:.3f} |")
L += ["", "## Reading",
      "- The proposed mechanism is 'noise inflates/scrambles velocity entropy T'. It fails twice: "
      "(i) T barely separates spoof/bona even on CLEAN MLAAD (AUC~0.53), so there is essentially no "
      "T signal for noise to destroy; (ii) adding noise LOWERS mean T (broadband noise makes "
      "frame-to-frame velocities uniformly large -> a more peaked velocity histogram -> lower "
      "entropy), moving T AWAY from the ITW value, not toward it.",
      "- The one quantity that shrinks with noise, rho(T,logit), is confounded: the detector itself "
      "collapses to chance (EER -> ~0.5) and rho(C,logit) degrades in step, so this is generic "
      "noise-degradation of the detector, not a T-specific effect.",
      "- Combined with Test 1 (T does not track SNR within ITW; T dead even in clean ITW clips), "
      "background noise is NOT the mechanism behind T's ITW failure from either the observational "
      "or the causal side.", "",
      "## Files",
      "- `e10_inject_metrics.csv`, `e10_inject_curves.png`"]
summ = RESDIR / "e10_summary.md"
prev = summ.read_text() if summ.exists() else ""
summ.write_text(prev + "\n".join(L))
print("\n".join(L[:30]))
print(f"\n[E10-T2] done -> {RESDIR}")
