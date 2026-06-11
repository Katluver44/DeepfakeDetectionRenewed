#!/usr/bin/env python3
"""
I6 — Test-time geometry interventions: turning the causal findings into performance
====================================================================================
I1 showed the detector's causal interface is dominant-subspace content + temporal
order: at FIXED rog, spectral sharpening (+0.0051***), temporal smoothing
(+0.0035***) and residual-subspace compaction (+0.0039***) all IMPROVE AUC on
ASVspoof (3/3 seeds). Mechanistic reading: within-utterance variance spread across
many minor directions dilutes the discriminative dominant subspace — test-time
"trajectory denoising" should therefore help, with the largest gains on HARD systems.

I6 evaluates zero-training, test-time-only interventions at the detector's encoder
output (gated to the detection path) on:
  (a) MLAAD test in-distribution, 3 detector seeds (main / s42 / s1024)
  (b) In-The-Wild (1500+1500 subset), MLAAD main detector (cross-domain)

Conditions: baseline, spec_{1.25,1.6,2.0}, smooth_{3,5}, sub_res_{0.7,0.5},
combo (smooth_3 then spec_1.6). Metrics: EER/AUC (+ paired utterance bootstrap on
deltas) and per-system AUC by baseline-hardness quartile.

Outputs -> experiments/results/i6_testtime_boost/
"""
from __future__ import annotations
import sys, json, time, types, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import px_common as px

OUT = px.EXP_DIR / "results" / "i6_testtime_boost"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = px.DEVICE
N_BOOT = 2000

SEED_CKPTS = {
    "main":  px.EXP_DIR / "checkpoints" / "mlaad_robust_goat.ckpt",
    "s42":   px.EXP_DIR / "checkpoints" / "mlaad_robust_goat_seed42-best-epoch=05-val-eer=0.3030.ckpt",
    "s1024": px.EXP_DIR / "checkpoints" / "mlaad_robust_goat_seed1024-best-epoch=03-val-eer=0.2976.ckpt",
}

# ─── transforms (same definitions as I1) ──────────────────────────────────────
def rog_bt(S): return torch.sqrt((S ** 2).sum(-1).mean(1))

def t_spec(h, gamma):
    mu = h.mean(1, keepdim=True); S = (h - mu).float()
    U, sv, Vh = torch.linalg.svd(S, full_matrices=False)
    sv2 = sv.clamp_min(1e-8) ** gamma
    c = torch.sqrt((sv ** 2).sum(-1) / (sv2 ** 2).sum(-1).clamp_min(1e-12))
    S2 = torch.einsum("btr,br,brd->btd", U, c.unsqueeze(-1) * sv2, Vh)
    return (mu + S2).to(h.dtype)

def t_smooth(h, k):
    mu = h.mean(1, keepdim=True); S = h - mu
    x = torch.nn.functional.avg_pool1d(S.transpose(1, 2), k, stride=1, padding=k // 2)
    St = x[..., :h.shape[1]].transpose(1, 2)
    St = St - St.mean(1, keepdim=True)
    scale = (rog_bt(S) / rog_bt(St).clamp_min(1e-12)).view(-1, 1, 1)
    return mu + scale * St

def t_sub_res(h, target):
    mu = h.mean(1, keepdim=True); S = (h - mu).float()
    U, sv, Vh = torch.linalg.svd(S, full_matrices=False)
    E = sv ** 2; tot = E.sum(-1, keepdim=True)
    cum = torch.cumsum(E, -1) / tot.clamp_min(1e-12)
    mask_top = (cum <= 0.5); mask_top[..., 0] = True
    mask = ~mask_top
    E_in = (E * mask).sum(-1); E_out = (E * (~mask)).sum(-1)
    f2 = ((target ** 2) * tot.squeeze(-1) - E_out) / E_in.clamp_min(1e-12)
    f = torch.sqrt(f2.clamp_min(0.0))
    fac = torch.where(mask, f.unsqueeze(-1), torch.ones_like(E))
    S2 = torch.einsum("btr,br,brd->btd", U, fac * sv, Vh)
    return (mu + S2).to(h.dtype)

CONDS = [("baseline",   lambda h: h),
         ("spec_1.25",  lambda h: t_spec(h, 1.25)),
         ("spec_1.6",   lambda h: t_spec(h, 1.6)),
         ("spec_2.0",   lambda h: t_spec(h, 2.0)),
         ("smooth_3",   lambda h: t_smooth(h, 3)),
         ("smooth_5",   lambda h: t_smooth(h, 5)),
         ("sub_res_0.7", lambda h: t_sub_res(h, 0.7)),
         ("sub_res_0.5", lambda h: t_sub_res(h, 0.5)),
         ("combo",      lambda h: t_spec(t_smooth(h, 3), 1.6))]

STATE = {"fn": None}
def hook(mod, inp, out):
    if not getattr(hook, "_tap_on", False) or STATE["fn"] is None:
        return None
    h = STATE["fn"](out[0])
    return (h,) + tuple(out[1:])

def install(gm):
    orig = gm.encoder_and_GAT.__func__
    def wrapped(self_inner, *a, **k):
        hook._tap_on = True
        try: return orig(self_inner, *a, **k)
        finally: hook._tap_on = False
    gm.encoder_and_GAT = types.MethodType(wrapped, gm)
    return gm.encoder.register_forward_hook(hook)

from sklearn.metrics import roc_auc_score, roc_curve
def eer_of(y, s):
    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)

# ─── data ─────────────────────────────────────────────────────────────────────
recs = json.loads(px.TEST_JSON.read_text())
z = np.load(px.WAVE_CACHE / "i2_full_test_waves.npz", allow_pickle=True)
mw, ok_idx = z["waves"].astype(np.float32), z["ok_idx"]
recs = [recs[i] for i in ok_idx]
mlab = np.array([1 if str(r["label"]).lower().startswith("spoof") else 0 for r in recs])
msys = np.array([f'{r["attack_system"]}|{r["language"]}' if mlab[i] else "bona"
                 for i, r in enumerate(recs)])

itw = px.load_itw_waves(n_per_class=3000)
rngi = np.random.default_rng(SEED)
sel_b = rngi.choice(np.where(itw["labels"] == 0)[0], 1500, replace=False)
sel_s = rngi.choice(np.where(itw["labels"] == 1)[0], 1500, replace=False)
isel = np.sort(np.r_[sel_b, sel_s])
iw, ilab = itw["waves"][isel], itw["labels"][isel]
print(f"[I6] MLAAD {len(mw)} utts | ITW subset {len(iw)} utts")

@torch.no_grad()
def score(gm, waves):
    out = []
    for b in range(0, len(waves), 16):
        xb = torch.as_tensor(np.stack(waves[b:b+16]), dtype=torch.float32, device=DEVICE)
        nf = torch.full((xb.shape[0],), px.NF_PER, device=DEVICE)
        out.extend(gm(xb, nf, profiler=None, use_aug=False, stage="val")["logit"]
                   .cpu().float().numpy().ravel().tolist())
    return np.asarray(out)

# ─── run sweeps ───────────────────────────────────────────────────────────────
store = {}     # (dataset, seed, cond) -> logits
rows = []
for sname, ck in SEED_CKPTS.items():
    gm = px.load_detector(ckpt=ck)
    handle = install(gm)
    for cname, fn in CONDS:
        STATE["fn"] = None if cname == "baseline" else fn
        t0 = time.time()
        lg = score(gm, mw)
        store[("mlaad", sname, cname)] = lg
        e, auc = eer_of(mlab, lg), roc_auc_score(mlab, lg)
        rows.append({"dataset": "mlaad", "seed": sname, "cond": cname,
                     "EER": e, "AUC": auc, "sec": time.time() - t0})
        print(f"  mlaad/{sname}/{cname:<11} EER={e:.4f} AUC={auc:.4f} ({time.time()-t0:.0f}s)",
              flush=True)
    if sname == "main":   # cross-domain ITW with the main detector
        for cname, fn in CONDS:
            STATE["fn"] = None if cname == "baseline" else fn
            lg = score(gm, iw)
            store[("itw", sname, cname)] = lg
            e, auc = eer_of(ilab, lg), roc_auc_score(ilab, lg)
            rows.append({"dataset": "itw", "seed": sname, "cond": cname,
                         "EER": e, "AUC": auc, "sec": np.nan})
            print(f"  itw/{sname}/{cname:<11} EER={e:.4f} AUC={auc:.4f}", flush=True)
    STATE["fn"] = None
    handle.remove()
    del gm; torch.cuda.empty_cache()

df = pd.DataFrame(rows)
df.to_csv(OUT / "i6_results.csv", index=False)
np.savez_compressed(OUT / "i6_logits.npz",
                    mlab=mlab, ilab=ilab,
                    **{f"{d}__{s}__{c}": v for (d, s, c), v in store.items()})

# ─── bootstrap deltas vs baseline ─────────────────────────────────────────────
def boot(dataset, cond, y, seeds):
    pairs = [(store[(dataset, s, cond)], store[(dataset, s, "baseline")]) for s in seeds]
    n = len(y); idx_mat = np.random.default_rng(SEED).integers(0, n, size=(N_BOOT, n))
    dA = float(np.mean([roc_auc_score(y, a) - roc_auc_score(y, b) for a, b in pairs]))
    dE = float(np.mean([eer_of(y, a) - eer_of(y, b) for a, b in pairs]))
    arrA = np.empty(N_BOOT); arrE = np.empty(N_BOOT)
    for i in range(N_BOOT):
        ix = idx_mat[i]; yb = y[ix]
        if yb.min() == yb.max(): arrA[i] = arrE[i] = np.nan; continue
        arrA[i] = np.mean([roc_auc_score(yb, a[ix]) - roc_auc_score(yb, b[ix]) for a, b in pairs])
        arrE[i] = np.mean([eer_of(yb, a[ix]) - eer_of(yb, b[ix]) for a, b in pairs])
    arrA = arrA[np.isfinite(arrA)]; arrE = arrE[np.isfinite(arrE)]
    pA = float(min(1, 2 * min((arrA <= 0).mean(), (arrA >= 0).mean())))
    pE = float(min(1, 2 * min((arrE <= 0).mean(), (arrE >= 0).mean())))
    return {"dataset": dataset, "cond": cond, "dAUC": dA,
            "dAUC_lo": float(np.percentile(arrA, 2.5)), "dAUC_hi": float(np.percentile(arrA, 97.5)),
            "p_AUC": pA, "dEER": dE,
            "dEER_lo": float(np.percentile(arrE, 2.5)), "dEER_hi": float(np.percentile(arrE, 97.5)),
            "p_EER": pE}

print("\n[boot] deltas vs baseline:")
brows = []
for cname, _ in CONDS[1:]:
    brows.append(boot("mlaad", cname, mlab, list(SEED_CKPTS)))
    brows.append(boot("itw", cname, ilab, ["main"]))
bdf = pd.DataFrame(brows)
bdf.to_csv(OUT / "i6_bootstrap.csv", index=False)
for _, r in bdf.iterrows():
    print(f"  {r.dataset:<6} {r.cond:<11} dEER={r.dEER:+.4f} [{r.dEER_lo:+.4f},{r.dEER_hi:+.4f}] "
          f"p={r.p_EER:.4f} | dAUC={r.dAUC:+.4f} p={r.p_AUC:.4f}")

# ─── hard-quartile breakdown (MLAAD, per-system AUC, mean over seeds) ─────────
sys_list = sorted({s for s in msys if s != "bona" and (msys == s).sum() >= 8})
def sys_auc(lg, s):
    bl = lg[mlab == 0]; sl = lg[msys == s]
    return roc_auc_score(np.r_[np.zeros(len(bl)), np.ones(len(sl))], np.r_[bl, sl])
base_h = {s: np.mean([1 - sys_auc(store[("mlaad", sd, "baseline")], s)
                      for sd in SEED_CKPTS]) for s in sys_list}
q = pd.qcut(pd.Series(base_h), 4, labels=["Q1_easy", "Q2", "Q3", "Q4_hard"])
qrows = []
for cname, _ in CONDS:
    for qq in ["Q1_easy", "Q2", "Q3", "Q4_hard"]:
        ss = [s for s in sys_list if q[s] == qq]
        v = np.mean([[sys_auc(store[("mlaad", sd, cname)], s) for s in ss]
                     for sd in SEED_CKPTS])
        qrows.append({"cond": cname, "quartile": qq, "mean_sys_AUC": float(v)})
qdf = pd.DataFrame(qrows).pivot(index="cond", columns="quartile", values="mean_sys_AUC")
qdf.to_csv(OUT / "i6_quartiles.csv")
print("\n[quartiles] mean per-system AUC (3 seeds):")
print(qdf.round(4).to_string())
print(f"\n[I6] done -> {OUT}")
