#!/usr/bin/env python3
"""
E9 — Are C and T CAUSAL factors for detection, or just correlates?
==================================================================

E1–E8 established that C (=-rog@L12) and T (=vel_entropy@L9) *correlate* with
per-system detection hardness. Correlation is not causation: C/T could be
epiphenomena of some third property the detector actually uses. This experiment
intervenes directly on the detector's internal WavLM representation and asks
whether **moving C/T causally changes EER/AUC/Accuracy**.

Mechanism (verified against phoneme_GAT/modules.py):
  encoder_and_GAT runs `hidden = self.encoder(proj)[0]` -> that encoder OUTPUT is
  WavLM L12 (= where C is defined) and feeds the GAT directly; L9 (= where T is
  defined) is self.encoder.layers[8]. The encoder is frozen. We register forward
  hooks to perturb these representations for *every* utterance (spoof AND bona,
  equally -> no label leakage) and re-measure metrics.

Interventions:
  C-knob (L12, around per-utterance temporal centroid mu):
      h' = mu + alpha*(h - mu)   -> rog (hence C) scales by exactly alpha.
      alpha<1 => more compact (higher C); alpha>1 => less compact (lower C).
  T-knob (L9):
      smooth  (temporal moving-average, kernel k)  -> lowers velocity -> lowers T
      jitter  (add temporal noise, scale beta)     -> raises  velocity -> raises T
  CONTROL (L12), the key to isolating causation from generic fragility:
      shift   h' = h + d*unit_vec  with ||Δh|| matched to the C-scale at the same
      energy. A constant translation preserves rog AND velocities EXACTLY -> C and
      T unchanged. If C-scaling degrades metrics but the equal-energy shift does
      NOT, the effect is specific to C, not to perturbation magnitude.

Causal verdict: C/T are causal if (a) metrics move monotonically with the knob,
through the unperturbed baseline, AND (b) the matched-energy C-preserving control
is comparatively flat. Direction should match each corpus's correlational polarity
(ASVspoof eval: higher C -> EASIER, so LOWERING C should WORSEN metrics).

Detectors: robust_GOAT (ASVspoof in-distribution) + GOAT. Data: balanced ASVspoof
2019 LA eval (A07–A19) subset. Evaluation only; no retraining.
"""
from __future__ import annotations
import sys, warnings
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

BASE    = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
OUTDIR  = BASE / "outputs" / "ct_causal_intervention"
OUTDIR.mkdir(parents=True, exist_ok=True)
for _p in (str(BASE), str(EXP_DIR), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR     = 16_000
TARGET_LEN    = 3 * TARGET_SR
NF            = TARGET_LEN // 320 - 1
BS            = 16
CACHE_DIR     = BASE / "data" / "asvspoof_2019_la"
EVAL_ATTACKS  = [f"A{i:02d}" for i in range(7, 20)]
N_SPOOF, N_BONA = 800, 800
MODELS = {"robust_GOAT": BASE / "models" / "robust_goat.ckpt",
          "GOAT":        BASE / "models" / "goat.ckpt"}
HEADLINE = "robust_GOAT"

from _ablation_common import _all_metrics  # noqa: E402
print(f"[E9] device={DEVICE}  out={OUTDIR}")

# ─── trajectory metrics (same defs as C/T) ─────────────────────────────────────
def rog_t(h):                       # h: (B,T,D) -> (B,) radius of gyration
    mu = h.mean(1, keepdim=True)
    return torch.sqrt(((h - mu) ** 2).sum(-1).mean(1))

def vel_entropy_np(frames):
    vels = np.linalg.norm(frames[1:] - frames[:-1], axis=1)
    n = max(5, min(20, len(vels) // 4))
    h, _ = np.histogram(vels, bins=n); h = h.astype(float) + 1e-8; h /= h.sum()
    return float(-np.sum(h * np.log(h)))

def _crop(wav):
    if wav.ndim > 1: wav = wav.mean(0)
    if len(wav) < TARGET_LEN:
        wav = wav.repeat(-(-TARGET_LEN // len(wav)))
    mid = (len(wav) - TARGET_LEN) // 2
    return wav[mid:mid + TARGET_LEN]

# ═══ 1. Load balanced ASVspoof eval subset ═════════════════════════════════════
print("\n[1] Loading ASVspoof eval subset ...", flush=True)
from datasets import load_dataset, Audio as HFAudio
ds = load_dataset("Bisher/as_vspoof_2019_la", cache_dir=str(CACHE_DIR), trust_remote_code=True)["test"]
sysids = ds["system_id"]
rng = np.random.default_rng(SEED)
spoof_idx = [i for i, s in enumerate(sysids) if s in EVAL_ATTACKS]
bona_idx  = [i for i, s in enumerate(sysids) if s == "-"]
sel = sorted(rng.choice(spoof_idx, N_SPOOF, replace=False).tolist() +
             rng.choice(bona_idx,  N_BONA,  replace=False).tolist())
sub = ds.select(sel).cast_column("audio", HFAudio(sampling_rate=TARGET_SR))
labels = np.array([0 if sysids[i] == "-" else 1 for i in sel])
print("  caching waveforms ...", flush=True)
wavs = [_crop(torch.tensor(sub[i]["audio"]["array"], dtype=torch.float32)) for i in range(len(sub))]
print(f"  {len(wavs)} utts (spoof={int(labels.sum())} bona={int((labels==0).sum())})")

# ═══ 2. Model loader ═══════════════════════════════════════════════════════════
def patch():
    import phoneme_GAT.modules as mm, phoneme_GAT.phoneme_model as pm
    from phoneme_GAT.phoneme_model import BaseModule, network_param, optim_param
    def _load(network_name="wavlm", pretrained_path=None, total_num_phonemes=198):
        network_param.network_name = network_name
        network_param.pretrained_name = "microsoft/wavlm-base"
        network_param.vocab_size = total_num_phonemes
        return BaseModule(network_param, optim_param, tokenizer=None, total_num_phonemes=total_num_phonemes)
    pm.load_phoneme_model = _load; mm.load_phoneme_model = _load
patch()
torch.serialization.add_safe_globals([Namespace])
try:
    from pandas import Series as _PS
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination as _PTR
    torch.serialization.add_safe_globals([_PS, _PTR])
except Exception:
    pass
from phoneme_GAT.modules import Phoneme_GAT_lit

def _n_edges(ck):
    c = torch.load(str(ck), weights_only=False, map_location="cpu").get("hyper_parameters", {}).get("cfg", None)
    n = getattr(getattr(c, "PhonemeGAT", None), "n_edges", None) if c else None
    return int(n) if n is not None else 10

def load_model(ck):
    cfg = Namespace(PhonemeGAT=Namespace(backbone="wavlm", use_raw=False, use_GAT=True,
                    n_edges=_n_edges(ck), use_aug=True, use_pool=True, use_clip=True))
    lit = Phoneme_GAT_lit.load_from_checkpoint(str(ck), cfg=cfg, map_location=DEVICE, strict=True)
    lit.to(DEVICE).eval(); lit.freeze()
    return lit

# ═══ 3. Hook-based intervention machinery ══════════════════════════════════════
# STATE controls what each hook does on the current forward pass.
STATE = {"C_mode": "none", "alpha": 1.0, "shift_dir": None, "shift_norm": 0.0,
         "T_mode": "none", "k": 1, "beta": 0.0,
         "probe": False, "rec": {}}

def c_hook(mod, inp, out):
    h = out[0]
    if STATE["C_mode"] == "scale":
        mu = h.mean(1, keepdim=True)
        h = mu + STATE["alpha"] * (h - mu)
    elif STATE["C_mode"] == "shift":
        # constant per-utterance translation: preserves rog & velocities exactly
        d = STATE["shift_dir"].to(h.dtype)                      # (D,) unit
        h = h + STATE["shift_norm"] * d.view(1, 1, -1)
    if STATE["probe"]:
        STATE["rec"]["C"] = rog_t(h).mean().item()
    return (h,) + tuple(out[1:])

def t_hook(mod, inp, out):
    h = out[0]
    if STATE["T_mode"] == "smooth":
        k = STATE["k"]
        x = h.transpose(1, 2)                                   # (B,D,T)
        x = torch.nn.functional.avg_pool1d(x, k, stride=1, padding=k // 2)
        h = x[..., :h.shape[1]].transpose(1, 2)
    elif STATE["T_mode"] == "jitter":
        vel = (h[:, 1:] - h[:, :-1]).norm(dim=-1).mean()        # scalar velocity scale
        h = h + STATE["beta"] * vel * torch.randn_like(h) / (h.shape[-1] ** 0.5)
    if STATE["probe"]:
        hh = h.detach().float().cpu().numpy()
        STATE["rec"]["T"] = float(np.mean([vel_entropy_np(hh[i]) for i in range(min(8, len(hh)))]))
        STATE["rec"]["C_atL9"] = rog_t(h).mean().item()
    return (h,) + tuple(out[1:])

@torch.no_grad()
def run_eval(gm):
    logits = []
    for b in range(0, len(wavs), BS):
        wb = torch.stack(wavs[b:b + BS]).to(DEVICE)
        nf = torch.full((wb.shape[0],), NF, device=DEVICE)
        logits.extend(gm(wb, nf, use_aug=False, stage="val")["logit"].cpu().numpy().tolist())
    return np.array(logits)

@torch.no_grad()
def probe_ct(gm):
    """Measure mean C (L12 rog) and T (L9 vel_entropy) under current STATE on a probe batch."""
    STATE["probe"] = True; STATE["rec"] = {}
    wb = torch.stack(wavs[:64]).to(DEVICE)
    nf = torch.full((wb.shape[0],), NF, device=DEVICE)
    gm(wb, nf, use_aug=False, stage="val")
    STATE["probe"] = False
    return STATE["rec"].get("C", np.nan), STATE["rec"].get("T", np.nan), STATE["rec"].get("C_atL9", np.nan)

def metrics(logits):
    m = _all_metrics(labels, logits)
    return m["eer"], m["auc"], m["acc"]

# ═══ 4. Run sweeps per model ═══════════════════════════════════════════════════
C_ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
T_SMOOTH = [3, 5, 9, 15]
T_JITTER = [0.5, 1.0, 2.0, 4.0]

rows_C, rows_T = [], []
shift_dir = torch.tensor(rng.standard_normal(768), dtype=torch.float32, device=DEVICE)
shift_dir = shift_dir / shift_dir.norm()
STATE["shift_dir"] = shift_dir

for mname, ck in MODELS.items():
    if not Path(ck).exists():
        print(f"  [skip] {mname}"); continue
    print(f"\n[model] {mname}", flush=True)
    gm = load_model(ck).model
    hC = gm.encoder.register_forward_hook(c_hook)
    hT = gm.encoder.layers[8].register_forward_hook(t_hook)

    # baseline C/T (probe, no perturbation)
    STATE.update(C_mode="none", T_mode="none")
    baseC, baseT, _ = probe_ct(gm)
    print(f"  baseline measured  C(L12 rog)={baseC:.3f}  T(L9 vel_ent)={baseT:.3f}")

    # ---- C-scale sweep + matched-energy shift control ----
    for a in C_ALPHAS:
        STATE.update(C_mode="scale", alpha=a, T_mode="none")
        mC, _, _ = probe_ct(gm)
        e, au, ac = metrics(run_eval(gm))
        rows_C.append({"model": mname, "intervention": "C_scale", "alpha": a,
                       "energy": abs(a - 1.0), "measured_C": mC, "EER": e, "AUC": au, "Acc": ac})
        # matched-energy shift control: ||Δh||_frame_rms = |a-1| * rog = energy*baseC
        STATE.update(C_mode="shift", shift_norm=abs(a - 1.0) * baseC)
        mC2, _, _ = probe_ct(gm)
        e2, au2, ac2 = metrics(run_eval(gm))
        rows_C.append({"model": mname, "intervention": "shift_control", "alpha": a,
                       "energy": abs(a - 1.0), "measured_C": mC2, "EER": e2, "AUC": au2, "Acc": ac2})
        print(f"    alpha={a:<4} C={mC:6.2f} | scale EER={e:.3f} AUC={au:.3f} Acc={ac:.3f} "
              f"|| shift(C={mC2:6.2f}) EER={e2:.3f} AUC={au2:.3f} Acc={ac2:.3f}")
    STATE.update(C_mode="none", shift_norm=0.0)

    # ---- T sweep (smooth lowers T, jitter raises T) ----
    for k in T_SMOOTH:
        STATE.update(T_mode="smooth", k=k)
        _, mT, cAt9 = probe_ct(gm)
        e, au, ac = metrics(run_eval(gm))
        rows_T.append({"model": mname, "intervention": "T_smooth", "param": k,
                       "measured_T": mT, "C_atL9": cAt9, "EER": e, "AUC": au, "Acc": ac})
        print(f"    smooth k={k:<3} T={mT:.3f} (C@L9={cAt9:.2f}) EER={e:.3f} AUC={au:.3f} Acc={ac:.3f}")
    for be in T_JITTER:
        STATE.update(T_mode="jitter", beta=be)
        _, mT, cAt9 = probe_ct(gm)
        e, au, ac = metrics(run_eval(gm))
        rows_T.append({"model": mname, "intervention": "T_jitter", "param": be,
                       "measured_T": mT, "C_atL9": cAt9, "EER": e, "AUC": au, "Acc": ac})
        print(f"    jitter b={be:<4} T={mT:.3f} (C@L9={cAt9:.2f}) EER={e:.3f} AUC={au:.3f} Acc={ac:.3f}")
    STATE.update(T_mode="none")

    # baseline row (alpha=1 already in C_scale; add explicit)
    hC.remove(); hT.remove()
    del gm; torch.cuda.empty_cache()

C_df = pd.DataFrame(rows_C); C_df.to_csv(OUTDIR / "e9_causal_C.csv", index=False)
T_df = pd.DataFrame(rows_T); T_df.to_csv(OUTDIR / "e9_causal_T.csv", index=False)
print("\n  saved e9_causal_C.csv, e9_causal_T.csv")

# ═══ 5. Figures ════════════════════════════════════════════════════════════════
print("\n[5] Figures ...", flush=True)
for mname in [m for m in MODELS if m in C_df.model.unique()]:
    sc = C_df[(C_df.model == mname) & (C_df.intervention == "C_scale")].sort_values("measured_C")
    sh = C_df[(C_df.model == mname) & (C_df.intervention == "shift_control")].sort_values("energy")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, met in zip(axes, ["EER", "AUC", "Acc"]):
        ax.plot(sc["measured_C"], sc[met], "o-", c="#d62728", label="C-scale (changes C)")
        # shift control plotted at its (preserved) measured_C, but vary by energy on twin axis idea:
        ax.plot(sh["measured_C"], sh[met], "s--", c="#1f77b4", alpha=0.7, label="shift control (C fixed)")
        ax.axvline(sc["measured_C"][sc["alpha"] == 1.0].values[0] if (sc["alpha"] == 1.0).any() else np.nan,
                   c="k", ls=":", lw=0.8)
        ax.set_xlabel("measured C (L12 rog)"); ax.set_ylabel(met); ax.set_title(met)
    axes[0].legend(fontsize=8)
    plt.suptitle(f"E9 causal C-intervention — {mname} (ASVspoof eval)\n"
                 "if C-scale curve moves but shift control is flat ⇒ C is causal", fontweight="bold")
    plt.tight_layout(); plt.savefig(OUTDIR / f"e9_causal_C_{mname}.png", dpi=150); plt.close()

    tt = T_df[T_df.model == mname].sort_values("measured_T")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, met in zip(axes, ["EER", "AUC", "Acc"]):
        ax.plot(tt["measured_T"], tt[met], "o-", c="#2ca02c")
        ax.set_xlabel("measured T (L9 vel-entropy)"); ax.set_ylabel(met); ax.set_title(met)
    plt.suptitle(f"E9 causal T-intervention — {mname} (smooth↓T / jitter↑T)", fontweight="bold")
    plt.tight_layout(); plt.savefig(OUTDIR / f"e9_causal_T_{mname}.png", dpi=150); plt.close()
print("  figures saved.")

# ═══ 6. Verdict ════════════════════════════════════════════════════════════════
def slope(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[m], y[m], 1)[0]) if m.sum() >= 3 else np.nan

L = ["# E9 — Are C and T causal for detection, or just correlates?", "",
     "Direct intervention on the detector's frozen WavLM representation (C at L12 encoder "
     "output, T at L9 = layers[8]), applied equally to spoof+bona. ASVspoof eval (A07–A19), "
     f"{int(labels.sum())} spoof + {int((labels==0).sum())} bona. Headline: **{HEADLINE}** "
     "(in-distribution). C-scale changes C by exactly α; the **shift control** translates the "
     "representation by the same ‖Δh‖ while leaving C/T mathematically unchanged — so any "
     "metric move under shift is generic fragility, not a C effect.",
     "",
     "Reference (E8 correlation, ASVspoof): higher C → EASIER ⇒ **lowering C (α>1) should "
     "WORSEN** EER/AUC/Acc if C is causal.", ""]
for mname in [m for m in MODELS if m in C_df.model.unique()]:
    sc = C_df[(C_df.model == mname) & (C_df.intervention == "C_scale")]
    sh = C_df[(C_df.model == mname) & (C_df.intervention == "shift_control")]
    base = sc[sc.alpha == 1.0].iloc[0]
    lowC = sc[sc.alpha == sc.alpha.max()].iloc[0]   # most lowered C (α=4)
    hiC  = sc[sc.alpha == sc.alpha.min()].iloc[0]   # most raised C (α=0.25)
    sl_eer = slope(sc["measured_C"].values, sc["EER"].values)
    sl_eer_sh = slope(sh["measured_C"].values, sh["EER"].values)  # ~0 expected (C fixed) — use energy
    sl_eer_sh = slope(sh["energy"].values, sh["EER"].values)
    L += [f"## {mname}",
          f"- baseline (α=1): EER={base.EER:.3f} AUC={base.AUC:.3f} Acc={base.Acc:.3f}  (C={base.measured_C:.2f})",
          f"- lower C (α=4, C={lowC.measured_C:.2f}): EER={lowC.EER:.3f} AUC={lowC.AUC:.3f} Acc={lowC.Acc:.3f}  "
          f"(ΔEER={lowC.EER-base.EER:+.3f}, ΔAUC={lowC.AUC-base.AUC:+.3f})",
          f"- raise C (α=0.25, C={hiC.measured_C:.2f}): EER={hiC.EER:.3f} AUC={hiC.AUC:.3f} Acc={hiC.Acc:.3f}  "
          f"(ΔEER={hiC.EER-base.EER:+.3f})",
          f"- slope EER vs C (scale) = {sl_eer:+.4f} per unit C; shift-control slope EER vs energy = {sl_eer_sh:+.4f}",
          ""]
    # causal call for this model
    eff = abs(lowC.EER - base.EER) + abs(base.AUC - lowC.AUC)
    sh_eff = abs(sh["EER"].max() - sh["EER"].min())
    causal = (eff > 0.02) and (eff > 1.5 * sh_eff)
    L.append(f"  **C verdict ({mname}): {'CAUSAL' if causal else 'not clearly causal'}** "
             f"(C-scale effect {eff:.3f} vs shift-control range {sh_eff:.3f}; "
             f"direction {'consistent with E8 (lower C → worse)' if (lowC.EER>base.EER) else 'OPPOSITE to E8'}).")
    tt = T_df[T_df.model == mname]
    sl_t = slope(tt["measured_T"].values, tt["EER"].values)
    t_range = abs(tt["EER"].max() - tt["EER"].min())
    L += ["", f"- T sweep ({mname}): EER vs T slope={sl_t:+.4f}, EER range over T sweep={t_range:.3f} "
          f"⇒ T {'appears causal' if t_range > 0.03 else 'shows little/no causal effect'}.", ""]

L += ["## Files",
      "- `e9_causal_C.csv` (C-scale + shift control), `e9_causal_T.csv` (smooth/jitter)",
      "- figures: `e9_causal_C_<model>.png`, `e9_causal_T_<model>.png`"]
(OUTDIR / "e9_summary.md").write_text("\n".join(L))
print(f"  summary -> {OUTDIR/'e9_summary.md'}")
print("\n[E9] done.")
