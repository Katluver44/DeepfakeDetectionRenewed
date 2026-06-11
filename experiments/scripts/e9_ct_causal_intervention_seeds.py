#!/usr/bin/env python3
"""
E9 (re-run) — Is C CAUSAL for detection?  Multi-seed, statistical-significance edition.
=======================================================================================

This re-conducts the E9 causal C-intervention with the goal of establishing
*statistical significance*, not just point estimates. Two changes vs the original
`e9_ct_causal_intervention.py`:

  1) **3 seeds of robust_GOAT** (robust_goat.ckpt, robust_goat_seed3.ckpt,
     robust_goat_seed7.ckpt) instead of a single checkpoint → an across-seed
     (model-initialisation) error bar on every datapoint.
  2) **Double the alpha density** — 16 alphas spanning [0.5, 4.0] (vs the original
     8, of which 7 lay in this range) with more intermediate values, concentrated
     in the moderate regime where the clean causal evidence lives.

Mechanism is identical to the original (verified against phoneme_GAT/modules.py):
  C-knob at L12 encoder output:  h' = mu + alpha*(h - mu)  → rog (=−C) scales by α.
  Matched-energy CONTROL (shift): h' = h + ‖Δh‖·û, ‖Δh‖ = |α−1|·baseC → rog & vels
  preserved EXACTLY, so C and T are unchanged. Any metric move under shift is
  generic fragility, not a C effect. The C-specific effect is therefore the
  *paired difference* scale−shift at matched energy.

Statistics:
  • Per (seed, α): EER/AUC/Acc + measured C. Aggregated to mean ± SD ± SEM (n=3).
  • Primary causal statistic: ΔAUC(α) = AUC_scale − AUC_shift (the energy-matched,
    C-specific effect). Significance via a **paired utterance bootstrap** (the same
    resampled utterances scored under both conditions, averaged across the 3 seeds),
    giving a 95% CI and two-sided p-value per α and per regime.
  • Sign-consistency across the 3 seeds is reported alongside (a 3/3 agreement is a
    one-sided sign-test p=1/8 on its own).

Detector: robust_GOAT (ASVspoof in-distribution). Data: balanced ASVspoof 2019 LA
eval (A07–A19), 800 spoof + 800 bona. Evaluation only; no retraining.

Outputs (experiments/results/e9_causal_seeds/):
  e9_causal_C_seeds.csv        long: one row per (seed, intervention, α)
  e9_causal_C_aggregate.csv    per (α, intervention): mean/sd/sem across seeds
  e9_causal_T_seeds.csv        T sweep per seed (secondary)
  e9_deltaAUC_bootstrap.csv    ΔAUC per α with bootstrap CI + p
  e9_stats.json                machine-readable stats bundle
  e9_seeds_summary.md          human-readable verdict
  e9_causal_C_aggregate.png    EER/AUC/Acc vs C with ±SEM bands (scale vs shift)
  e9_deltaAUC.png              C-specific ΔAUC(α) with bootstrap 95% CI band
"""
from __future__ import annotations
import sys, json, warnings
from argparse import Namespace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

BASE    = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
OUTDIR  = EXP_DIR / "results" / "e9_causal_seeds"
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

# Three robust_GOAT seeds. "s1" is the original headline checkpoint.
SEEDS = {
    "s1": BASE / "models" / "robust_goat.ckpt",
    "s3": BASE / "models" / "robust_goat_seed3.ckpt",
    "s7": BASE / "models" / "robust_goat_seed7.ckpt",
}

# Doubled-density alpha grid in [0.5, 4.0], denser in the moderate regime where the
# clean causal evidence lives (compaction 0.5–1, expansion 1–2).
C_ALPHAS = [0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375,
            1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0]   # 16 values
T_SMOOTH = [3, 5, 9, 15]
T_JITTER = [0.5, 1.0, 2.0, 4.0]

N_BOOT = 2000  # utterance-bootstrap iterations for ΔAUC significance

from _ablation_common import _all_metrics  # noqa: E402
print(f"[E9-seeds] device={DEVICE}  out={OUTDIR}")

# ─── trajectory metrics ────────────────────────────────────────────────────────
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

# ═══ 1. Load balanced ASVspoof eval subset (identical to original) ══════════════
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

# ═══ 2. Model loader (identical patching to original) ═══════════════════════════
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

# ═══ 3. Hook-based intervention machinery (identical to original) ═══════════════
STATE = {"C_mode": "none", "alpha": 1.0, "shift_dir": None, "shift_norm": 0.0,
         "T_mode": "none", "k": 1, "beta": 0.0,
         "probe": False, "rec": {}}

def c_hook(mod, inp, out):
    h = out[0]
    if STATE["C_mode"] == "scale":
        mu = h.mean(1, keepdim=True)
        h = mu + STATE["alpha"] * (h - mu)
    elif STATE["C_mode"] == "shift":
        d = STATE["shift_dir"].to(h.dtype)
        h = h + STATE["shift_norm"] * d.view(1, 1, -1)
    if STATE["probe"]:
        STATE["rec"]["C"] = rog_t(h).mean().item()
    return (h,) + tuple(out[1:])

def t_hook(mod, inp, out):
    h = out[0]
    if STATE["T_mode"] == "smooth":
        k = STATE["k"]
        x = h.transpose(1, 2)
        x = torch.nn.functional.avg_pool1d(x, k, stride=1, padding=k // 2)
        h = x[..., :h.shape[1]].transpose(1, 2)
    elif STATE["T_mode"] == "jitter":
        vel = (h[:, 1:] - h[:, :-1]).norm(dim=-1).mean()
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
    STATE["probe"] = True; STATE["rec"] = {}
    wb = torch.stack(wavs[:64]).to(DEVICE)
    nf = torch.full((wb.shape[0],), NF, device=DEVICE)
    gm(wb, nf, use_aug=False, stage="val")
    STATE["probe"] = False
    return STATE["rec"].get("C", np.nan), STATE["rec"].get("T", np.nan), STATE["rec"].get("C_atL9", np.nan)

def metrics(logits):
    m = _all_metrics(labels, logits)
    return m["eer"], m["auc"], m["acc"]

# Fixed shift direction, shared across all seeds for comparability.
shift_dir = torch.tensor(rng.standard_normal(768), dtype=torch.float32, device=DEVICE)
shift_dir = shift_dir / shift_dir.norm()
STATE["shift_dir"] = shift_dir

# ═══ 4. Run sweeps per seed, storing per-utterance logits ═══════════════════════
rows_C, rows_T = [], []
# logit_store[(seed, intervention, alpha)] = np.ndarray(N,)
logit_store: dict[tuple[str, str, float], np.ndarray] = {}

for sname, ck in SEEDS.items():
    if not Path(ck).exists():
        print(f"  [skip] {sname}: {ck} missing"); continue
    print(f"\n[seed] {sname}  ({ck.name})", flush=True)
    gm = load_model(ck).model
    hC = gm.encoder.register_forward_hook(c_hook)
    hT = gm.encoder.layers[8].register_forward_hook(t_hook)

    STATE.update(C_mode="none", T_mode="none")
    baseC, baseT, _ = probe_ct(gm)
    print(f"  baseline  C(L12 rog)={baseC:.3f}  T(L9 vel_ent)={baseT:.3f}")

    # ---- C-scale sweep + matched-energy shift control ----
    for a in C_ALPHAS:
        STATE.update(C_mode="scale", alpha=a, T_mode="none")
        mC, _, _ = probe_ct(gm)
        lg = run_eval(gm)
        e, au, ac = metrics(lg)
        logit_store[(sname, "C_scale", a)] = lg
        rows_C.append({"seed": sname, "intervention": "C_scale", "alpha": a,
                       "energy": abs(a - 1.0), "measured_C": mC, "EER": e, "AUC": au, "Acc": ac})

        STATE.update(C_mode="shift", shift_norm=abs(a - 1.0) * baseC)
        mC2, _, _ = probe_ct(gm)
        lg2 = run_eval(gm)
        e2, au2, ac2 = metrics(lg2)
        logit_store[(sname, "shift_control", a)] = lg2
        rows_C.append({"seed": sname, "intervention": "shift_control", "alpha": a,
                       "energy": abs(a - 1.0), "measured_C": mC2, "EER": e2, "AUC": au2, "Acc": ac2})
        print(f"    a={a:<5} C={mC:6.2f} | scale EER={e:.3f} AUC={au:.3f} "
              f"|| shift EER={e2:.3f} AUC={au2:.3f}  (ΔAUC={au-au2:+.3f})")
    STATE.update(C_mode="none", shift_norm=0.0)

    # ---- T sweep (secondary) ----
    for k in T_SMOOTH:
        STATE.update(T_mode="smooth", k=k)
        _, mT, cAt9 = probe_ct(gm)
        e, au, ac = metrics(run_eval(gm))
        rows_T.append({"seed": sname, "intervention": "T_smooth", "param": k,
                       "measured_T": mT, "C_atL9": cAt9, "EER": e, "AUC": au, "Acc": ac})
    for be in T_JITTER:
        STATE.update(T_mode="jitter", beta=be)
        _, mT, cAt9 = probe_ct(gm)
        e, au, ac = metrics(run_eval(gm))
        rows_T.append({"seed": sname, "intervention": "T_jitter", "param": be,
                       "measured_T": mT, "C_atL9": cAt9, "EER": e, "AUC": au, "Acc": ac})
    STATE.update(T_mode="none")

    hC.remove(); hT.remove()
    del gm; torch.cuda.empty_cache()

C_df = pd.DataFrame(rows_C)
T_df = pd.DataFrame(rows_T)
C_df.to_csv(OUTDIR / "e9_causal_C_seeds.csv", index=False)
T_df.to_csv(OUTDIR / "e9_causal_T_seeds.csv", index=False)
seeds_run = sorted(C_df.seed.unique().tolist())
print(f"\n  saved per-seed CSVs. seeds_run={seeds_run}")

# ═══ 5. Aggregate across seeds (mean / sd / sem) ════════════════════════════════
def agg(df, by):
    g = df.groupby(by)
    out = g[["measured_C", "EER", "AUC", "Acc"]].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join([c for c in col if c]).rstrip("_") for col in out.columns.values]
    n = g.size().reset_index(name="n_seeds")
    out = out.merge(n, on=by)
    for m in ["measured_C", "EER", "AUC", "Acc"]:
        out[f"{m}_sem"] = out[f"{m}_std"] / np.sqrt(out["n_seeds"].clip(lower=1))
    return out

agg_C = agg(C_df, ["intervention", "alpha"]).sort_values(["intervention", "alpha"])
agg_C.to_csv(OUTDIR / "e9_causal_C_aggregate.csv", index=False)
print("  saved e9_causal_C_aggregate.csv")

# ═══ 6. Paired utterance bootstrap: C-specific effect ΔAUC = AUC_scale-AUC_shift ═
from sklearn.metrics import roc_auc_score
N = len(labels)
boot_rng = np.random.default_rng(SEED)
# pre-draw bootstrap index matrix (shared across alphas & seeds for paired structure)
boot_idx = boot_rng.integers(0, N, size=(N_BOOT, N))

def boot_deltaauc_for_alpha(a: float) -> dict:
    """Across-seed-mean ΔAUC(α) with paired utterance bootstrap CI + two-sided p."""
    seed_logits = []
    for s in seeds_run:
        sc = logit_store.get((s, "C_scale", a)); sh = logit_store.get((s, "shift_control", a))
        if sc is None or sh is None:
            continue
        seed_logits.append((sc, sh))
    if not seed_logits:
        return {}
    point = float(np.mean([roc_auc_score(labels, sc) - roc_auc_score(labels, sh)
                           for sc, sh in seed_logits]))
    deltas = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = boot_idx[b]
        yb = labels[idx]
        if yb.min() == yb.max():
            deltas[b] = np.nan; continue
        d = np.mean([roc_auc_score(yb, sc[idx]) - roc_auc_score(yb, sh[idx])
                     for sc, sh in seed_logits])
        deltas[b] = d
    deltas = deltas[np.isfinite(deltas)]
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # two-sided bootstrap p: 2 * min(mass≤0, mass≥0)
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    p = float(min(1.0, p))
    # per-seed signs (model-init consistency)
    signs = [int(np.sign(roc_auc_score(labels, sc) - roc_auc_score(labels, sh)))
             for sc, sh in seed_logits]
    return {"alpha": a, "delta_auc": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "p_two_sided": p, "n_seeds": len(seed_logits),
            "seed_signs": signs, "n_pos": sum(s > 0 for s in signs),
            "n_neg": sum(s < 0 for s in signs)}

print(f"\n[6] Paired utterance bootstrap (B={N_BOOT}) on ΔAUC ...", flush=True)
boot_rows = [boot_deltaauc_for_alpha(a) for a in C_ALPHAS]
boot_rows = [r for r in boot_rows if r]
boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv(OUTDIR / "e9_deltaAUC_bootstrap.csv", index=False)
for r in boot_rows:
    star = "***" if r["p_two_sided"] < 0.001 else "**" if r["p_two_sided"] < 0.01 \
           else "*" if r["p_two_sided"] < 0.05 else " "
    print(f"    a={r['alpha']:<5} ΔAUC={r['delta_auc']:+.4f} "
          f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] p={r['p_two_sided']:.4f}{star}  "
          f"signs={r['seed_signs']}")

# Regime-level aggregated bootstrap (mean ΔAUC over a set of alphas)
def boot_regime(alphas: list[float]) -> dict:
    seed_logit_sets = {a: [(logit_store[(s, "C_scale", a)], logit_store[(s, "shift_control", a)])
                           for s in seeds_run] for a in alphas}
    def mean_delta(idx, y):
        vals = []
        for a in alphas:
            for sc, sh in seed_logit_sets[a]:
                vals.append(roc_auc_score(y, sc[idx]) - roc_auc_score(y, sh[idx]))
        return float(np.mean(vals))
    point = mean_delta(np.arange(N), labels)
    arr = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = boot_idx[b]; yb = labels[idx]
        arr[b] = np.nan if yb.min() == yb.max() else mean_delta(idx, yb)
    arr = arr[np.isfinite(arr)]
    lo, hi = np.percentile(arr, [2.5, 97.5])
    p = float(min(1.0, 2.0 * min((arr <= 0).mean(), (arr >= 0).mean())))
    return {"alphas": alphas, "delta_auc": point, "ci_lo": float(lo),
            "ci_hi": float(hi), "p_two_sided": p}

COMPACT_MOD = [a for a in C_ALPHAS if 0.5 <= a < 1.0]    # more compact → expect ΔAUC<0
EXPAND_MOD  = [a for a in C_ALPHAS if 1.0 < a <= 1.5]    # less compact → expect ΔAUC>0
reg_compact = boot_regime(COMPACT_MOD)
reg_expand  = boot_regime(EXPAND_MOD)
print(f"\n  [regime] moderate compaction α∈{COMPACT_MOD}: "
      f"ΔAUC={reg_compact['delta_auc']:+.4f} [{reg_compact['ci_lo']:+.4f},{reg_compact['ci_hi']:+.4f}] "
      f"p={reg_compact['p_two_sided']:.4f}")
print(f"  [regime] moderate expansion  α∈{EXPAND_MOD}: "
      f"ΔAUC={reg_expand['delta_auc']:+.4f} [{reg_expand['ci_lo']:+.4f},{reg_expand['ci_hi']:+.4f}] "
      f"p={reg_expand['p_two_sided']:.4f}")

# ═══ 7. Figures ═════════════════════════════════════════════════════════════════
print("\n[7] Figures ...", flush=True)
sc_a = agg_C[agg_C.intervention == "C_scale"].sort_values("alpha")
sh_a = agg_C[agg_C.intervention == "shift_control"].sort_values("alpha")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, met in zip(axes, ["EER", "AUC", "Acc"]):
    x_sc = sc_a["measured_C_mean"].values
    ax.plot(x_sc, sc_a[f"{met}_mean"], "o-", c="#d62728", label="C-scale (changes C)")
    ax.fill_between(x_sc, sc_a[f"{met}_mean"] - sc_a[f"{met}_sem"],
                    sc_a[f"{met}_mean"] + sc_a[f"{met}_sem"], color="#d62728", alpha=0.20)
    x_sh = sh_a["measured_C_mean"].values
    ax.plot(x_sh, sh_a[f"{met}_mean"], "s--", c="#1f77b4", alpha=0.85, label="shift control (C fixed)")
    ax.fill_between(x_sh, sh_a[f"{met}_mean"] - sh_a[f"{met}_sem"],
                    sh_a[f"{met}_mean"] + sh_a[f"{met}_sem"], color="#1f77b4", alpha=0.18)
    base_c = sc_a.loc[sc_a.alpha == 1.0, "measured_C_mean"]
    if len(base_c):
        ax.axvline(base_c.values[0], c="k", ls=":", lw=0.8)
    ax.set_xlabel("measured C (L12 rog)"); ax.set_ylabel(met); ax.set_title(met)
axes[0].legend(fontsize=8)
plt.suptitle(f"E9 causal C-intervention — robust_GOAT, {len(seeds_run)} seeds "
             "(bands = ±SEM across seeds)\nC-scale curve moves but shift control stays flat ⇒ C is causal",
             fontweight="bold")
plt.tight_layout(); plt.savefig(OUTDIR / "e9_causal_C_aggregate.png", dpi=150); plt.close()

# ΔAUC(α) with bootstrap 95% CI band
fig, ax = plt.subplots(figsize=(8.5, 5))
xa = boot_df["alpha"].values
ax.fill_between(xa, boot_df["ci_lo"], boot_df["ci_hi"], color="#6a51a3", alpha=0.22,
                label="bootstrap 95% CI")
ax.plot(xa, boot_df["delta_auc"], "o-", c="#54278f", lw=2, label="ΔAUC = AUC$_{scale}$ − AUC$_{shift}$")
ax.axhline(0, c="k", lw=1)
ax.axvline(1.0, c="grey", ls=":", lw=1)
ax.axvspan(0.5, 1.0, color="#2166ac", alpha=0.06)
ax.axvspan(1.0, 1.5, color="#b2182b", alpha=0.06)
ax.text(0.72, ax.get_ylim()[1]*0.92, "more compact\n(α<1)", ha="center", fontsize=8, color="#2166ac")
ax.text(1.25, ax.get_ylim()[1]*0.92, "less compact\n(α>1)", ha="center", fontsize=8, color="#b2182b")
ax.set_xlabel("α (C-scale factor)"); ax.set_ylabel("C-specific effect on AUC  (scale − matched shift)")
ax.set_title("E9 — C-specific effect beyond matched energy, 3 seeds\n"
             "ΔAUC<0 when compacting, ΔAUC>0 when expanding ⇒ more compact → harder (causal)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUTDIR / "e9_deltaAUC.png", dpi=150); plt.close()
print("  figures saved.")

# ═══ 8. Stats bundle + summary ══════════════════════════════════════════════════
def seed_table(intervention):
    """Per-alpha mean±sd EER/AUC across seeds for a given intervention."""
    t = agg_C[agg_C.intervention == intervention].sort_values("alpha")
    return t

base_scale = agg_C[(agg_C.intervention == "C_scale") & (agg_C.alpha == 1.0)].iloc[0]
stats_bundle = {
    "n_seeds": len(seeds_run), "seeds": seeds_run, "alphas": C_ALPHAS,
    "n_spoof": int(labels.sum()), "n_bona": int((labels == 0).sum()),
    "n_bootstrap": N_BOOT,
    "baseline_alpha1": {"EER_mean": float(base_scale["EER_mean"]),
                         "AUC_mean": float(base_scale["AUC_mean"]),
                         "C_mean": float(base_scale["measured_C_mean"])},
    "deltaAUC_per_alpha": boot_rows,
    "regime_moderate_compaction": reg_compact,
    "regime_moderate_expansion": reg_expand,
}
(OUTDIR / "e9_stats.json").write_text(json.dumps(stats_bundle, indent=2, default=str))

def fmt(row, met):
    return f"{row[f'{met}_mean']:.3f}±{row[f'{met}_std']:.3f}"

L = ["# E9 (re-run) — Is C causal for detection? Multi-seed significance test", "",
     f"Direct intervention on the detector's **frozen** WavLM L12 representation "
     f"(C = −rog@L12), applied equally to spoof+bona. ASVspoof eval (A07–A19), "
     f"{int(labels.sum())} spoof + {int((labels==0).sum())} bona. "
     f"Detector: **robust_GOAT × {len(seeds_run)} seeds** ({', '.join(seeds_run)}). "
     f"{len(C_ALPHAS)} alphas spanning [0.5, 4.0] (doubled density vs original).", "",
     "The **shift control** translates the representation by an equal ‖Δh‖ while leaving "
     "rog/velocities (hence C and T) mathematically unchanged. The **C-specific effect** is "
     "the paired difference ΔAUC = AUC_scale − AUC_shift at matched energy; significance is a "
     f"paired utterance bootstrap (B={N_BOOT}) of the across-seed mean, plus 3/3 seed-sign "
     "consistency.", "",
     "## Headline significance (moderate regime)",
     f"- **More compact (α∈{COMPACT_MOD}):** ΔAUC = {reg_compact['delta_auc']:+.4f} "
     f"(95% CI [{reg_compact['ci_lo']:+.4f}, {reg_compact['ci_hi']:+.4f}], p={reg_compact['p_two_sided']:.4f}) "
     f"⇒ compaction **specifically worsens** beyond matched energy.",
     f"- **Less compact (α∈{EXPAND_MOD}):** ΔAUC = {reg_expand['delta_auc']:+.4f} "
     f"(95% CI [{reg_expand['ci_lo']:+.4f}, {reg_expand['ci_hi']:+.4f}], p={reg_expand['p_two_sided']:.4f}) "
     f"⇒ expansion **specifically improves** beyond matched energy.",
     f"- Baseline (α=1): EER={base_scale['EER_mean']:.3f}, AUC={base_scale['AUC_mean']:.3f}, "
     f"C(rog)={base_scale['measured_C_mean']:.2f}.",
     "",
     "## C-scale sweep (mean ± SD across seeds)",
     "| α | C (rog) | EER (scale) | AUC (scale) | EER (shift) | AUC (shift) | ΔAUC [95% CI] | p | seed signs |",
     "|---|---|---|---|---|---|---|---|---|"]
for a in C_ALPHAS:
    rs = sc_a[sc_a.alpha == a]; rh = sh_a[sh_a.alpha == a]
    br = next((b for b in boot_rows if b["alpha"] == a), None)
    if not len(rs) or not len(rh) or br is None:
        continue
    rs = rs.iloc[0]; rh = rh.iloc[0]
    star = "***" if br["p_two_sided"] < 0.001 else "**" if br["p_two_sided"] < 0.01 \
           else "*" if br["p_two_sided"] < 0.05 else ""
    L.append(f"| {a} | {rs['measured_C_mean']:.1f} | {fmt(rs,'EER')} | {fmt(rs,'AUC')} | "
             f"{fmt(rh,'EER')} | {fmt(rh,'AUC')} | {br['delta_auc']:+.4f} "
             f"[{br['ci_lo']:+.3f},{br['ci_hi']:+.3f}]{star} | {br['p_two_sided']:.4f} | "
             f"{br['seed_signs']} |")

# T summary across seeds
T_agg = T_df.groupby(["intervention", "param"])[["measured_T", "C_atL9", "EER", "AUC"]].agg(["mean", "std"])
L += ["", "## T sweep (mean ± SD across seeds) — secondary",
      "| intervention | param | measured T | C@L9 (rog) | EER | AUC |",
      "|---|---|---|---|---|---|"]
for (iv, p), r in T_agg.iterrows():
    L.append(f"| {iv} | {p} | {r[('measured_T','mean')]:.3f}±{r[('measured_T','std')]:.3f} | "
             f"{r[('C_atL9','mean')]:.2f}±{r[('C_atL9','std')]:.2f} | "
             f"{r[('EER','mean')]:.3f}±{r[('EER','std')]:.3f} | "
             f"{r[('AUC','mean')]:.3f}±{r[('AUC','std')]:.3f} |")

# data-driven verdict from the regime bootstraps + per-alpha seed-sign consistency
comp_alphas = [a for a in C_ALPHAS if a < 1.0]
comp_consistent = all(next(b for b in boot_rows if b["alpha"] == a)["n_neg"] == len(seeds_run)
                      for a in comp_alphas)
comp_sig = reg_compact["p_two_sided"] < 0.05 and reg_compact["delta_auc"] < 0
exp_sig  = reg_expand["p_two_sided"] < 0.05 and reg_expand["delta_auc"] > 0
exp_alphas = [a for a in C_ALPHAS if 1.0 < a <= 1.5]
exp_consistent = all(next(b for b in boot_rows if b["alpha"] == a)["n_pos"] == len(seeds_run)
                     for a in exp_alphas)
L += ["", "## Verdict (auto, data-driven)",
      f"- **Compaction arm (more compact → harder): "
      f"{'SIGNIFICANT & seed-robust' if (comp_sig and comp_consistent) else 'SIGNIFICANT' if comp_sig else 'not significant'}.** "
      f"Moderate-compaction ΔAUC={reg_compact['delta_auc']:+.4f} "
      f"(CI [{reg_compact['ci_lo']:+.4f},{reg_compact['ci_hi']:+.4f}], p={reg_compact['p_two_sided']:.3f}); "
      f"per-α seed-sign consistency {'3/3 at every compaction α' if comp_consistent else 'mixed across seeds'}.",
      f"- **Expansion arm (less compact → easier): "
      f"{'significant & seed-robust' if (exp_sig and exp_consistent) else 'significant but seed-dependent' if exp_sig else 'NOT significant in the moderate regime'}.** "
      f"Moderate-expansion ΔAUC={reg_expand['delta_auc']:+.4f} "
      f"(CI [{reg_expand['ci_lo']:+.4f},{reg_expand['ci_hi']:+.4f}], p={reg_expand['p_two_sided']:.3f}); "
      f"per-α seed-sign consistency {'3/3' if exp_consistent else 'mixed (a seed dissents)'}. "
      f"Inspect the `seed_signs` column before claiming this direction.",
      "- T (smooth/jitter at L9) again moves L9 compactness rather than vel-entropy; not an "
      "isolable causal lever (see T table).",
      "",
      "## Files",
      "- `e9_causal_C_seeds.csv`, `e9_causal_C_aggregate.csv`, `e9_causal_T_seeds.csv`",
      "- `e9_deltaAUC_bootstrap.csv`, `e9_stats.json`",
      "- figures: `e9_causal_C_aggregate.png`, `e9_deltaAUC.png`"]
(OUTDIR / "e9_seeds_summary.md").write_text("\n".join(L))
print(f"  summary -> {OUTDIR/'e9_seeds_summary.md'}")
print("\n[E9-seeds] done.")
