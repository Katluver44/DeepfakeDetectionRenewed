#!/usr/bin/env python3
"""
I1 — Geometry-resolved causal decomposition of the E9 compaction effect
========================================================================
E9 established (3 seeds, paired bootstrap) that isotropic compaction of the L12
representation harms detection beyond a matched-energy shift control. But the
isotropic map  h' = mu + a*(h-mu)  scales radius-of-gyration, ALL frame velocities,
and the whole covariance spectrum by the same factor — so "C (= -rog) is causal"
is underdetermined: the operative coordinate could be total dispersion (rog),
spectrum shape (effective rank), or temporal smoothness (velocity magnitude).

I1 dissociates them with matched interventions at the same site (trainable-encoder
output feeding phoneme-pool -> GAT, == E9's "L12"):

  iso(a)            rog x a, vel x a, spectrum-shape fixed     (E9 replication, gated)
  shift(n)          rog/vel/spectrum all fixed                 (E9 energy control)
  spec(gamma)       eff-rank moves, rog EXACTLY fixed          (shape at fixed size)
  smooth_rescale(k) vel down, rog EXACTLY restored             (smoothness at fixed size)
  jitter_rescale(b) vel up,   rog EXACTLY restored
  shuffle           temporal order destroyed, rog/spectrum exactly fixed
  sub_top / sub_res compaction confined to top / residual PCs, matched rog target
  iso_ungated(0.7)  E9's ORIGINAL ungated hook (also corrupts the frozen phoneme-ID
                    path, modules.py:417+576 share the encoder object) — audits how
                    much of E9's published effect rides on phoneme-assignment damage.

All other arms are GATED to the detection path only (px_common AxisProjector trick).

Decision logic:
  * iso harms but smooth_rescale/jitter_rescale ≈ flat  -> velocity is NOT the channel.
  * spec ≈ flat at fixed rog                            -> spectrum shape NOT the channel;
    rog (total dispersion) is the sufficient statistic  -> C genuinely causal (A).
  * smooth_rescale reproduces the harm                  -> C partially proxies smoothness (B/E).
  * spec harms at fixed rog                             -> C is a low-dim summary of richer
    spectral structure (B/E).
  * sub_top vs sub_res asymmetry                        -> directional structure matters.

Stats: per-utterance logits stored for every (seed, condition); paired utterance
bootstrap (B=2000) of across-seed mean AUC differences for the key contrasts.
Setup mirrors E9: ASVspoof 2019 LA eval A07-A19, 800 spoof + 800 bona,
robust_GOAT seeds s1/s3/s7.

Outputs -> experiments/results/i1_geometry_causal_decomp/
"""
from __future__ import annotations
import sys, json, time, warnings
from argparse import Namespace
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

BASE    = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
OUT     = EXP_DIR / "results" / "i1_geometry_causal_decomp"
OUT.mkdir(parents=True, exist_ok=True)
for _p in (str(BASE), str(EXP_DIR), str(SCRIPTS)):
    if _p not in sys.path: sys.path.insert(0, _p)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR, TARGET_LEN = 16_000, 48_000
NF = TARGET_LEN // 320 - 1
BS = 16
N_SPOOF = N_BONA = 800
EVAL_ATTACKS = [f"A{i:02d}" for i in range(7, 20)]
# seed7 checkpoint is unrecoverable (truncated at the Drive source), so I1 runs
# with the two available robust_GOAT seeds: s1 (robust_goat) and s3
# (robust_goat_seed3), both taken from models/good_models per user direction.
SEEDS = {"s1": BASE/"models"/"good_models"/"robust_goat.ckpt",
         "s3": BASE/"models"/"good_models"/"robust_goat_seed3.ckpt"}
N_BOOT = 2000

# ─── data (identical selection to E9) ─────────────────────────────────────────
def _crop(wav):
    if wav.ndim > 1: wav = wav.mean(0)
    if len(wav) < TARGET_LEN: wav = wav.repeat(-(-TARGET_LEN // len(wav)))
    mid = (len(wav) - TARGET_LEN) // 2
    return wav[mid:mid + TARGET_LEN]

print("[I1] loading ASVspoof eval subset ...", flush=True)
# Bisher/as_vspoof_2019_la no longer exists on the Hub and script-based datasets
# (trust_remote_code) are unsupported by datasets>=5. Load the locally-built cache
# (RohitGENAICODER/ASVspoofLADataset subset) saved via save_to_disk instead.
from datasets import load_from_disk, Audio as HFAudio
ds = load_from_disk(str(BASE/"data"/"asvspoof_2019_la"))["test"]
sysids = ds["system_id"]
rng = np.random.default_rng(SEED)
spoof_idx = [i for i, s in enumerate(sysids) if s in EVAL_ATTACKS]
bona_idx  = [i for i, s in enumerate(sysids) if s == "-"]
sel = sorted(rng.choice(spoof_idx, N_SPOOF, replace=False).tolist() +
             rng.choice(bona_idx,  N_BONA,  replace=False).tolist())
# audio is already stored as a decoded {array, sampling_rate=16000} struct in the
# local cache, so no Audio-feature cast/resample is needed (TARGET_SR == 16000).
sub = ds.select(sel)
labels = np.array([0 if sysids[i] == "-" else 1 for i in sel])
wavs = [_crop(torch.tensor(sub[i]["audio"]["array"], dtype=torch.float32)) for i in range(len(sub))]
print(f"  {len(wavs)} utts (spoof={int(labels.sum())})")

# ─── model loading (E9-identical) ─────────────────────────────────────────────
def patch():
    import phoneme_GAT.modules as mm, phoneme_GAT.phoneme_model as pm
    from phoneme_GAT.phoneme_model import BaseModule, network_param, optim_param
    def _load(network_name="wavlm", pretrained_path=None, total_num_phonemes=198):
        network_param.network_name = network_name
        network_param.pretrained_name = "microsoft/wavlm-base"
        network_param.vocab_size = total_num_phonemes
        return BaseModule(network_param, optim_param, tokenizer=None,
                          total_num_phonemes=total_num_phonemes)
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
    c = torch.load(str(ck), weights_only=False, map_location="cpu")\
        .get("hyper_parameters", {}).get("cfg", None)
    n = getattr(getattr(c, "PhonemeGAT", None), "n_edges", None) if c else None
    return int(n) if n is not None else 10

def load_model(ck):
    cfg = Namespace(PhonemeGAT=Namespace(backbone="wavlm", use_raw=False, use_GAT=True,
                    n_edges=_n_edges(ck), use_aug=True, use_pool=True, use_clip=True))
    lit = Phoneme_GAT_lit.load_from_checkpoint(str(ck), cfg=cfg, map_location=DEVICE, strict=True)
    lit.to(DEVICE).eval(); lit.freeze()
    return lit

# ─── transforms on (B,T,768) ──────────────────────────────────────────────────
def rog_bt(S):  # S centered (B,T,D) -> (B,)
    return torch.sqrt((S ** 2).sum(-1).mean(1))

def t_iso(h, a):
    mu = h.mean(1, keepdim=True)
    return mu + a * (h - mu)

def t_shift(h, n, direction):
    return h + n * direction.to(h.dtype).view(1, 1, -1)

def t_spec(h, gamma):
    mu = h.mean(1, keepdim=True); S = (h - mu).float()
    U, sv, Vh = torch.linalg.svd(S, full_matrices=False)
    sv2 = sv.clamp_min(1e-8) ** gamma
    c = torch.sqrt((sv ** 2).sum(-1) / (sv2 ** 2).sum(-1).clamp_min(1e-12))  # (B,)
    S2 = torch.einsum("btr,br,brd->btd", U, c.unsqueeze(-1) * sv2, Vh)
    return (mu + S2).to(h.dtype)

def t_smooth_rescale(h, k):
    mu = h.mean(1, keepdim=True); S = h - mu
    x = torch.nn.functional.avg_pool1d(S.transpose(1, 2), k, stride=1, padding=k // 2)
    St = x[..., :h.shape[1]].transpose(1, 2)
    St = St - St.mean(1, keepdim=True)
    scale = (rog_bt(S) / rog_bt(St).clamp_min(1e-12)).view(-1, 1, 1)
    return mu + scale * St

def t_jitter_rescale(h, b):
    mu = h.mean(1, keepdim=True); S = h - mu
    vel = (S[:, 1:] - S[:, :-1]).norm(dim=-1).mean(1).view(-1, 1, 1)
    St = S + b * vel * torch.randn_like(S) / (S.shape[-1] ** 0.5)
    St = St - St.mean(1, keepdim=True)
    scale = (rog_bt(S) / rog_bt(St).clamp_min(1e-12)).view(-1, 1, 1)
    return mu + scale * St

def t_shuffle(h, gen):
    B, T, D = h.shape
    out = h.clone()
    for i in range(B):
        perm = torch.randperm(T, generator=gen, device=h.device)
        out[i] = h[i, perm]
    return out

def t_subspace(h, target, which):
    """Compact only top (cumvar>=0.5) or residual PCs toward overall rog ratio `target`."""
    mu = h.mean(1, keepdim=True); S = (h - mu).float()
    U, sv, Vh = torch.linalg.svd(S, full_matrices=False)
    E = sv ** 2                                   # (B,r)
    tot = E.sum(-1, keepdim=True)
    cum = torch.cumsum(E, -1) / tot.clamp_min(1e-12)
    mask_top = (cum <= 0.5)
    mask_top[..., 0] = True                       # at least the first component
    mask = mask_top if which == "top" else ~mask_top
    E_in = (E * mask).sum(-1); E_out = (E * (~mask)).sum(-1)
    f2 = ((target ** 2) * tot.squeeze(-1) - E_out) / E_in.clamp_min(1e-12)
    f = torch.sqrt(f2.clamp_min(0.0))             # clamp if infeasible; record achieved rog
    fac = torch.where(mask, f.unsqueeze(-1), torch.ones_like(E))
    S2 = torch.einsum("btr,br,brd->btd", U, fac * sv, Vh)
    return (mu + S2).to(h.dtype)

# ─── hook machinery (gated to detection path) ─────────────────────────────────
STATE = {"mode": "none", "p": None, "probe": False, "rec": [], "gate": True,
         "shift_dir": None, "gen": None}

def apply_transform(h):
    m, p = STATE["mode"], STATE["p"]
    if m == "iso":            h = t_iso(h, p)
    elif m == "shift":        h = t_shift(h, p, STATE["shift_dir"])
    elif m == "spec":         h = t_spec(h, p)
    elif m == "smooth":       h = t_smooth_rescale(h, p)
    elif m == "jitter":       h = t_jitter_rescale(h, p)
    elif m == "shuffle":      h = t_shuffle(h, STATE["gen"])
    elif m == "sub_top":      h = t_subspace(h, p, "top")
    elif m == "sub_res":      h = t_subspace(h, p, "res")
    return h

def probe_stats(h):
    mu = h.mean(1, keepdim=True); S = (h - mu).float()
    rog = rog_bt(S).mean().item()
    vel = (h[:, 1:] - h[:, :-1]).float().norm(dim=-1).mean().item()
    sv = torch.linalg.svdvals(S)
    ev = (sv ** 2); ev = ev / ev.sum(-1, keepdim=True).clamp_min(1e-12)
    er = torch.exp(-(ev * (ev + 1e-12).log()).sum(-1)).mean().item()
    return rog, vel, er

def hook(mod, inp, out):
    if STATE["gate"] and not getattr(hook, "_tap_on", False):
        return None
    h = out[0]
    if STATE["mode"] != "none":
        h = apply_transform(h)
    if STATE["probe"]:
        STATE["rec"].append(probe_stats(h))
    return (h,) + tuple(out[1:])

def install(gm):
    import types
    orig = gm.encoder_and_GAT.__func__
    def wrapped(self_inner, *a, **k):
        hook._tap_on = True
        try: return orig(self_inner, *a, **k)
        finally: hook._tap_on = False
    gm.encoder_and_GAT = types.MethodType(wrapped, gm)
    return gm.encoder.register_forward_hook(hook)

@torch.no_grad()
def run_eval(gm, probe_batches=3):
    logits, STATE["rec"] = [], []
    for b in range(0, len(wavs), BS):
        STATE["probe"] = (b // BS) < probe_batches
        wb = torch.stack(wavs[b:b + BS]).to(DEVICE)
        nf = torch.full((wb.shape[0],), NF, device=DEVICE)
        logits.extend(gm(wb, nf, use_aug=False, stage="val")["logit"].cpu().numpy().tolist())
    STATE["probe"] = False
    rec = np.array(STATE["rec"]) if STATE["rec"] else np.full((1, 3), np.nan)
    return np.array(logits), rec.mean(0)  # logits, (rog, vel, eff_rank)

from sklearn.metrics import roc_auc_score, roc_curve
def eer_of(y, s):
    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)

# ─── conditions ───────────────────────────────────────────────────────────────
ISO    = [0.5, 0.7, 0.875, 1.25]
SPEC   = [0.6, 0.8, 1.25, 1.6]
SMOOTH = [3, 5, 9]
JITTER = [0.5, 1.0]
SUBT   = 0.7

def conditions(baseC):
    conds = [("baseline", "none", None, True)]
    conds += [(f"iso_{a}", "iso", a, True) for a in ISO]
    conds += [(f"shift_{a}", "shift", abs(a - 1.0) * baseC, True) for a in ISO]
    conds += [("iso_0.7_ungated", "iso", 0.7, False)]
    conds += [(f"spec_{g}", "spec", g, True) for g in SPEC]
    conds += [(f"smooth_{k}", "smooth", k, True) for k in SMOOTH]
    conds += [(f"jitter_{b}", "jitter", b, True) for b in JITTER]
    conds += [("shuffle", "shuffle", None, True)]
    conds += [("sub_top_0.7", "sub_top", SUBT, True), ("sub_res_0.7", "sub_res", SUBT, True)]
    return conds

shift_dir = torch.tensor(np.random.default_rng(SEED).standard_normal(768),
                         dtype=torch.float32, device=DEVICE)
STATE["shift_dir"] = shift_dir / shift_dir.norm()

rows = []
store: dict[tuple, np.ndarray] = {}
for sname, ck in SEEDS.items():
    if not Path(ck).exists():
        print(f"  [skip] {sname}"); continue
    print(f"\n[seed] {sname}", flush=True)
    gm = load_model(ck).model
    handle = install(gm)
    STATE["gen"] = torch.Generator(device=DEVICE); STATE["gen"].manual_seed(SEED)

    STATE.update(mode="none", gate=True)
    lg, (baseC, baseV, baseER) = run_eval(gm)
    print(f"  baseline rog={baseC:.2f} vel={baseV:.3f} effrank={baseER:.1f} "
          f"AUC={roc_auc_score(labels, lg):.4f}", flush=True)

    for cname, mode, p, gate in conditions(baseC):
        STATE.update(mode=mode, p=p, gate=gate)
        STATE["gen"].manual_seed(SEED)            # reproducible shuffle/jitter
        t0 = time.time()
        lg, (rg, vl, er) = run_eval(gm)
        auc = roc_auc_score(labels, lg); eer = eer_of(labels, lg)
        store[(sname, cname)] = lg
        rows.append({"seed": sname, "cond": cname, "mode": mode, "param": p,
                     "gated": gate, "rog": rg, "rog_ratio": rg / baseC,
                     "vel": vl, "vel_ratio": vl / baseV, "eff_rank": er,
                     "AUC": auc, "EER": eer, "sec": time.time() - t0})
        print(f"    {cname:<18} rog={rg/baseC:5.2f}x vel={vl/baseV:5.2f}x er={er:5.1f} "
              f"AUC={auc:.4f} EER={eer:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    STATE.update(mode="none")
    handle.remove()
    del gm; torch.cuda.empty_cache()

df = pd.DataFrame(rows)
df.to_csv(OUT / "i1_conditions.csv", index=False)
np.savez_compressed(OUT / "i1_logits.npz", labels=labels,
                    **{f"{s}__{c}": v for (s, c), v in store.items()})
seeds_run = sorted(df.seed.unique())

# ─── paired utterance bootstrap on key contrasts ──────────────────────────────
N = len(labels)
boot_idx = np.random.default_rng(SEED).integers(0, N, size=(N_BOOT, N))

def boot_contrast(condA, condB):
    """across-seed mean AUC(condA) - AUC(condB), paired utterance bootstrap."""
    pairs = [(store[(s, condA)], store[(s, condB)]) for s in seeds_run
             if (s, condA) in store and (s, condB) in store]
    if not pairs: return None
    point = float(np.mean([roc_auc_score(labels, a) - roc_auc_score(labels, b)
                           for a, b in pairs]))
    arr = np.empty(N_BOOT)
    for bi in range(N_BOOT):
        idx = boot_idx[bi]; yb = labels[idx]
        if yb.min() == yb.max(): arr[bi] = np.nan; continue
        arr[bi] = np.mean([roc_auc_score(yb, a[idx]) - roc_auc_score(yb, b[idx])
                           for a, b in pairs])
    arr = arr[np.isfinite(arr)]
    lo, hi = np.percentile(arr, [2.5, 97.5])
    p = float(min(1.0, 2 * min((arr <= 0).mean(), (arr >= 0).mean())))
    signs = [int(np.sign(roc_auc_score(labels, a) - roc_auc_score(labels, b))) for a, b in pairs]
    return {"A": condA, "B": condB, "dAUC": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "p": p, "seed_signs": signs}

# pick the smooth condition whose mean vel_ratio best matches iso_0.7's
sm_means = df[df["mode"] == "smooth"].groupby("cond")["vel_ratio"].mean()
iso07_vel = df[df.cond == "iso_0.7"]["vel_ratio"].mean()
smooth_star = (sm_means - iso07_vel).abs().idxmin()
print(f"\n  matched-velocity smooth condition: {smooth_star} "
      f"(vel {sm_means[smooth_star]:.2f}x vs iso_0.7 {iso07_vel:.2f}x)")

contrasts = []
for c in df.cond.unique():
    if c != "baseline":
        contrasts.append(boot_contrast(c, "baseline"))
contrasts += [boot_contrast("iso_0.7", "shift_0.7"),
              boot_contrast("iso_0.7", "iso_0.7_ungated"),
              boot_contrast("iso_0.7", smooth_star),
              boot_contrast("sub_top_0.7", "sub_res_0.7")]
cdf = pd.DataFrame([c for c in contrasts if c])
cdf.to_csv(OUT / "i1_contrasts.csv", index=False)

# ─── summary ──────────────────────────────────────────────────────────────────
agg = df.groupby("cond")[["rog_ratio", "vel_ratio", "eff_rank", "AUC", "EER"]].agg(["mean", "std"])
L = ["# I1 — Geometry-resolved causal decomposition of the compaction effect", "",
     f"ASVspoof eval A07–A19, 800+800, robust_GOAT seeds {seeds_run}. All arms gated to the "
     "detection path (phoneme-ID path untouched) except `iso_0.7_ungated` (E9's original "
     "ungated hook). Paired utterance bootstrap B=2000 on across-seed mean AUC.", "",
     "## Conditions (mean ± SD across seeds)",
     "| cond | rog ratio | vel ratio | eff rank | AUC | EER |", "|---|---|---|---|---|---|"]
for c in df.cond.unique():
    r = agg.loc[c]
    L.append(f"| {c} | {r[('rog_ratio','mean')]:.2f} | {r[('vel_ratio','mean')]:.2f} | "
             f"{r[('eff_rank','mean')]:.1f} | {r[('AUC','mean')]:.4f}±{r[('AUC','std')]:.4f} | "
             f"{r[('EER','mean')]:.3f}±{r[('EER','std')]:.3f} |")
L += ["", "## Key contrasts (ΔAUC = A − B, bootstrap 95% CI)",
      "| A | B | ΔAUC | 95% CI | p | seed signs |", "|---|---|---|---|---|---|"]
for _, r in cdf.iterrows():
    star = "***" if r.p < 0.001 else "**" if r.p < 0.01 else "*" if r.p < 0.05 else ""
    L.append(f"| {r.A} | {r.B} | {r.dAUC:+.4f}{star} | [{r.ci_lo:+.4f},{r.ci_hi:+.4f}] | "
             f"{r.p:.4f} | {r.seed_signs} |")
(OUT / "i1_summary.md").write_text("\n".join(L))
(OUT / "i1_meta.json").write_text(json.dumps(
    {"seeds": seeds_run, "n_boot": N_BOOT, "smooth_star": smooth_star,
     "n_utts": int(N)}, indent=2))
print(f"\n[I1] done -> {OUT}")
