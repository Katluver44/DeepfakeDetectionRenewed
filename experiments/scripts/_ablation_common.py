#!/usr/bin/env python3
"""
_ablation_common.py
===================
Shared infrastructure for the C/T ablation roadmap (P5–P8 and re-usable by
P2/P3/P4). Factors out the boilerplate that was previously duplicated across
p1/p2/p4 scripts:

  - path constants + torch.load compat shim
  - per-system EER computation (matches p2_ct_feature_injection_train.py)
  - test-split evaluation loop
  - **C/T-quartile stratification** (the Verification Protocol requirement that
    the stock p2 summary omits — it stratifies by EER, not by C/T)
  - dataloader + Lightning trainer construction

Everything here is deliberately model-agnostic: callers pass a Lightning module
(or a factory) and get back a per-system EER dict + stratified report.

System-name keys are consistent across the three sources we join on:
    train.json `attack_system`  ∩  outputs/sig_layer_features.csv `system`  = 63
    p1 per_system_eer.csv `system`  ∩  sig_layer_features `system`          = 63
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPTS_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parents[1]
EXP_DIR      = PROJECT_ROOT / "experiments"
CKPT_DIR     = EXP_DIR / "checkpoints"
PROC_DIR     = EXP_DIR / "data" / "mlaad_tiny_processed"
BASE_CKPT    = CKPT_DIR / "mlaad_robust_goat.ckpt"
TEST_JSON    = EXP_DIR / "results" / "mlaad" / "baseline_eval" / "test_in_distribution.json"
SIG_FEATURES = PROJECT_ROOT / "outputs" / "sig_layer_features.csv"
BASELINE_EER = EXP_DIR / "results" / "mlaad" / "p1_ct_calibration" / "per_system_eer.csv"

for _p in (str(PROJECT_ROOT), str(EXP_DIR), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ─── A100 throughput knobs (safe; ~1.5-2× on transformer matmuls) ─────────────
# TF32 accelerates fp32 matmuls/convs with negligible accuracy change; cudnn
# autotuner picks the fastest kernels for our fixed 48000-sample inputs.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ─── torch.load compat (Lightning ckpts pickle argparse.Namespace etc.) ───────
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

from argparse import Namespace
try:
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination
    from pandas import Series
    torch.serialization.add_safe_globals([Namespace, Phonemer_Tokenizer_Recombination, Series])
except Exception:
    torch.serialization.add_safe_globals([Namespace])


# ─── EER primitives ──────────────────────────────────────────────────────────

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float | None:
    from scipy.interpolate import interp1d
    from scipy.optimize import brentq
    from sklearn.metrics import roc_curve
    if len(np.unique(labels)) < 2:
        return None
    try:
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        return float(brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0))
    except Exception:
        return None


def per_system_eer_from_dict(all_labels, all_logits, all_systems, min_n=5):
    """Each spoof system is scored against the *shared* bonafide pool."""
    bf_labels, bf_logits = [], []
    sp_groups = defaultdict(lambda: ([], []))
    for y, s, sysname in zip(all_labels, all_logits, all_systems):
        if y == 0:
            bf_labels.append(y); bf_logits.append(s)
        else:
            sp_groups[sysname][0].append(y); sp_groups[sysname][1].append(s)
    results = {}
    for sysname, (sp_lab, sp_log) in sorted(sp_groups.items()):
        if len(sp_lab) < min_n:
            results[sysname] = None; continue
        comb_labels = np.array(sp_lab + bf_labels)
        comb_scores = np.array(sp_log + bf_logits)
        results[sysname] = compute_eer(comb_labels, comb_scores)
    return results


# Metric directions: True = higher is better. Used by the consistency check so a
# real improvement must move EER down AND AUC/acc/bal_acc up together.
METRIC_HIGHER_BETTER = {"eer": False, "auc": True, "acc": True, "bal_acc": True}
METRIC_KEYS = ("eer", "auc", "bal_acc", "acc")


def _all_metrics(labels, scores) -> dict | None:
    """EER, AUC, accuracy + balanced accuracy (at the Youden-J threshold) for one
    bonafide-vs-one-system comparison. None if a class is missing."""
    from sklearn.metrics import roc_auc_score, roc_curve
    labels = np.asarray(labels); scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return None
    eer = compute_eer(labels, scores)
    try:
        auc = float(roc_auc_score(labels, scores))
    except Exception:
        auc = None
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    j = np.argmax(tpr - fpr)
    t = thr[j]
    pred = (scores >= t).astype(int)
    acc = float((pred == labels).mean())
    tp = int(((pred == 1) & (labels == 1)).sum()); fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum()); fp = int(((pred == 1) & (labels == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    bal_acc = float((sens + spec) / 2)
    return {"eer": eer, "auc": auc, "acc": acc, "bal_acc": bal_acc, "n_spoof": int((labels == 1).sum())}


def per_system_metrics_from_dict(all_labels, all_logits, all_systems, min_n=5) -> dict:
    """Like per_system_eer_from_dict but returns the full metric bundle per system.
    {system: {'eer','auc','acc','bal_acc','n_spoof'} | None}."""
    bf_labels, bf_logits = [], []
    sp_groups = defaultdict(lambda: ([], []))
    for y, s, sysname in zip(all_labels, all_logits, all_systems):
        if y == 0:
            bf_labels.append(y); bf_logits.append(s)
        else:
            sp_groups[sysname][0].append(y); sp_groups[sysname][1].append(s)
    results = {}
    for sysname, (sp_lab, sp_log) in sorted(sp_groups.items()):
        if len(sp_lab) < min_n:
            results[sysname] = None; continue
        results[sysname] = _all_metrics(np.array(sp_lab + bf_labels),
                                        np.array(sp_log + bf_logits))
    return results


# ─── Test-split evaluation ───────────────────────────────────────────────────

@torch.no_grad()
def evaluate_on_test(model, device, out_csv: Path | None = None, batch_size: int = 16):
    """Run model.model(audio, num_frames, use_aug=False, stage='val') over the
    locked test split; return {system: {'eer','auc','acc','bal_acc','n_spoof'}}."""

    class _TestDS(torch.utils.data.Dataset):
        def __init__(self, records): self.records = records
        def __len__(self): return len(self.records)
        def __getitem__(self, i):
            r = self.records[i]
            wav = torch.load(PROC_DIR / r["audio_path"]).unsqueeze(0)
            return {"audio": wav,
                    "label": 0 if r["label"] == "bonafide" else 1,
                    "system": r["attack_system"]}

    def _collate(batch):
        return {
            "audio":  torch.stack([b["audio"] for b in batch]),
            "label":  [b["label"] for b in batch],
            "system": [b["system"] for b in batch],
        }

    records = json.loads(TEST_JSON.read_text())
    dl = torch.utils.data.DataLoader(
        _TestDS(records), batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=_collate, pin_memory=True)

    model.eval().to(device)
    all_labels, all_logits, all_systems = [], [], []
    NF = 48000 // 320 - 1
    for batch in dl:
        audio = batch["audio"].to(device)
        num_frames = torch.full((audio.shape[0],), NF, device=device)
        out = model.model(audio, num_frames, use_aug=False, stage="val")
        all_logits.extend(out["logit"].cpu().tolist())
        all_labels.extend(batch["label"])
        all_systems.extend(batch["system"])

    sys_metrics = per_system_metrics_from_dict(all_labels, all_logits, all_systems)

    if out_csv is not None:
        import pandas as pd
        rows = [{"system": s, **m} for s, m in sys_metrics.items() if m is not None]
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).sort_values("eer", ascending=False).to_csv(out_csv, index=False)
    return sys_metrics


# ─── C/T loading + quartile stratification ───────────────────────────────────

def load_system_ct() -> dict[str, dict]:
    """{system: {'C': −rog_L12, 'T': vel_entropy_L9, 'residual': r}} for 63 systems.

    C uses the roadmap convention C = −rog@L12, so HIGHER C = more compact =
    harder. With T also higher = harder, the top quartile (Q4) on either axis is
    consistently the hard-system bucket the roadmap targets.
    """
    import csv
    out = {}
    with open(SIG_FEATURES) as f:
        for row in csv.DictReader(f):
            out[row["system"]] = {
                "C": -float(row["rog_L12"]),
                "T": float(row["vel_entropy_L9"]),
                "residual": float(row["residual"]),
            }
    return out


def load_baseline_metrics(csv_path: Path = BASELINE_EER) -> dict[str, dict]:
    """Baseline per-system metrics {system: {'eer','auc','acc','bal_acc'}}.

    Accepts the P1 per_system_eer.csv schema (eer_before/auc_before/acc_before/
    bal_acc_before) or a plain eval CSV (eer/auc/acc/bal_acc)."""
    import pandas as pd
    if not Path(csv_path).exists():
        return {}
    df = pd.read_csv(csv_path)
    suffix = "_before" if "eer_before" in df.columns else ""
    out = {}
    for _, r in df.iterrows():
        m = {}
        for k in ("eer", "auc", "acc", "bal_acc"):
            col = k + suffix
            if col in df.columns:
                m[k] = float(r[col])
        out[r["system"]] = m
    return out


def load_baseline_eer(csv_path: Path = BASELINE_EER) -> dict[str, float]:
    """Back-compat: baseline per-system EER only."""
    return {s: m["eer"] for s, m in load_baseline_metrics(csv_path).items() if "eer" in m}


def _quartile_buckets(systems, ct, axis):
    vals = np.array([ct[s][axis] for s in systems])
    qs = np.quantile(vals, [0.25, 0.50, 0.75])
    buckets = [[] for _ in range(4)]
    for s in systems:
        v = ct[s][axis]
        q = 0 if v <= qs[0] else 1 if v <= qs[1] else 2 if v <= qs[2] else 3
        buckets[q].append(s)
    return buckets


def stratify_by_ct(metrics_model: dict[str, dict | float],
                   metrics_baseline: dict[str, dict | float] | None = None,
                   ct: dict[str, dict] | None = None,
                   metric_keys=METRIC_KEYS) -> dict:
    """
    Stratify per-system metrics by C-quartile and T-quartile. Q4 = highest C/T =
    hardest systems, where the roadmap predicts C/T gains should concentrate.

    Accepts either the full metric bundle ({system:{'eer','auc',...}}) or a bare
    {system: eer} float dict (back-compat — only EER is then reported).

    Returns {'n_systems', 'metric_keys', 'C':[4 rows], 'T':[4 rows], 'overall'}.
    Each quartile row carries model/baseline/delta for every available metric.
    """
    ct = ct or load_system_ct()

    def _bundle(d):
        if not d:
            return {}
        sample = next(iter(d.values()))
        if isinstance(sample, dict) or sample is None:
            return {s: m for s, m in d.items() if m is not None}
        return {s: {"eer": v} for s, v in d.items() if v is not None}

    mm = _bundle(metrics_model)
    bm = _bundle(metrics_baseline or {})
    keys = [k for k in metric_keys if any(k in v for v in mm.values())]

    systems = [s for s in mm if s in ct]
    report = {"n_systems": len(systems), "metric_keys": keys}

    for axis in ("C", "T"):
        rows = []
        for qi, members in enumerate(_quartile_buckets(systems, ct, axis)):
            row = {"quartile": f"Q{qi+1}", "n": len(members)}
            for k in keys:
                mvals = [mm[s][k] for s in members if k in mm[s] and mm[s][k] is not None]
                bvals = [bm[s][k] for s in members if s in bm and k in bm[s] and bm[s][k] is not None]
                row[f"model_{k}"] = float(np.mean(mvals)) if mvals else float("nan")
                row[f"baseline_{k}"] = float(np.mean(bvals)) if bvals else float("nan")
                row[f"delta_{k}"] = (row[f"model_{k}"] - row[f"baseline_{k}"]) \
                    if (mvals and bvals) else float("nan")
            rows.append(row)
        report[axis] = rows

    overall = {}
    for k in keys:
        mvals = [mm[s][k] for s in systems if k in mm[s] and mm[s][k] is not None]
        overall[f"model_mean_{k}"] = float(np.mean(mvals)) if mvals else float("nan")
        common = [s for s in systems if s in bm and k in bm[s] and bm[s][k] is not None
                  and k in mm[s] and mm[s][k] is not None]
        if common:
            overall[f"baseline_mean_{k}"] = float(np.mean([bm[s][k] for s in common]))
            overall[f"delta_mean_{k}"] = float(np.mean([mm[s][k] - bm[s][k] for s in common]))
            overall["n_common"] = len(common)
    # Back-compat aliases used by the per-P verdict functions.
    overall["model_mean_eer"] = overall.get("model_mean_eer", float("nan"))
    overall["model_median_eer"] = float(np.median(
        [mm[s]["eer"] for s in systems if "eer" in mm[s] and mm[s]["eer"] is not None]) or float("nan")) \
        if systems else float("nan")
    overall["delta_mean_eer"] = overall.get("delta_mean_eer", float("nan"))
    report["overall"] = overall
    return report


def consistency_note(report: dict, axis: str = "C", quartile_idx: int = 3) -> str:
    """Cross-metric agreement on the hard (Q4) bucket: a genuine improvement moves
    EER DOWN while AUC/acc/bal_acc move UP. Returns a one-line verdict."""
    row = report[axis][quartile_idx]
    keys = [k for k in report["metric_keys"] if not np.isnan(row.get(f"delta_{k}", float("nan")))]
    if not keys:
        return "consistency: no baseline deltas available"
    votes = {}
    for k in keys:
        d = row[f"delta_{k}"]
        improved = (d < 0) if not METRIC_HIGHER_BETTER[k] else (d > 0)
        votes[k] = improved
    n_improve = sum(votes.values())
    parts = ", ".join(f"{k}{'↑' if METRIC_HIGHER_BETTER[k] else '↓'}"
                      f"={'better' if votes[k] else 'worse/flat'}" for k in keys)
    if all(votes.values()):
        tag = "CONSISTENT improvement"
    elif not any(votes.values()):
        tag = "CONSISTENT no-improvement/regression"
    else:
        tag = f"MIXED ({n_improve}/{len(keys)} metrics agree)"
    return f"{axis}-Q4 cross-metric: {tag} — {parts}"


def stratified_markdown(report: dict, title: str) -> str:
    """Render a stratify_by_ct() report as markdown with EER/AUC/bal_acc/acc
    tables. Q4 rows are the hard systems the roadmap targets; a trend is only
    credible if EER and AUC (and accuracy) agree (see consistency line)."""
    keys = report["metric_keys"]
    ov = report["overall"]
    L = [f"## {title}", "", f"Systems with C/T + metrics: {report['n_systems']}", "",
         "### Overall (model | Δ vs baseline)"]
    for k in keys:
        mm = ov.get(f"model_mean_{k}", float("nan"))
        dm = ov.get(f"delta_mean_{k}", float("nan"))
        arrow = "↓ better" if not METRIC_HIGHER_BETTER[k] else "↑ better"
        dstr = f"{dm:+.4f}" if not np.isnan(dm) else "—"
        L.append(f"- {k} ({arrow}): {mm:.4f}   Δ={dstr}")
    for axis, name in (("C", "C = −rog@L12 (Q4 = most compact = hardest)"),
                       ("T", "T = vel_entropy@L9 (Q4 = burstiest = hardest)")):
        header = "| Quartile | n | " + " | ".join(
            f"{k} (Δ)" for k in keys) + " |"
        sep = "|" + "---|" * (2 + len(keys))
        L += ["", f"### Stratified by {name}", header, sep]
        for r in report[axis]:
            cells = [r["quartile"], str(r["n"])]
            for k in keys:
                mv = r.get(f"model_{k}", float("nan"))
                dv = r.get(f"delta_{k}", float("nan"))
                ds = f"{dv:+.3f}" if not np.isnan(dv) else "—"
                cells.append(f"{mv:.3f} ({ds})" if not np.isnan(mv) else "—")
            L.append("| " + " | ".join(cells) + " |")
        L += ["", f"_{consistency_note(report, axis)}_"]
    return "\n".join(L)


# ─── Dataloaders + trainer (mirrors p2 setup) ────────────────────────────────

def build_dataloaders(seed: int, batch_size: int | None = None):
    """Balanced train + val loaders identical to the p2 ablation setup."""
    from train_mlaad_adversarial import HP, AUG_FNS, MAALDSplitDataset
    from torch.utils.data import DataLoader
    bs = batch_size or HP["batch_size"]
    splits_dir = PROC_DIR / "splits"
    train_ds = MAALDSplitDataset(
        splits_dir / "train.json", PROC_DIR, mode="train",
        balance=True, seed=seed, aug_prob=HP["aug_prob"],
        aug_fns=AUG_FNS, aug_weights=HP["aug_weights"])
    val_ds = MAALDSplitDataset(
        splits_dir / "val.json", PROC_DIR, mode="eval", balance=True, seed=seed)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                          num_workers=HP["num_workers"], pin_memory=True,
                          drop_last=HP["drop_last"], persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False,
                        num_workers=HP["num_workers"], pin_memory=True,
                        drop_last=False, persistent_workers=True)
    return train_dl, val_dl


def make_trainer(max_epochs: int, log_dir: Path, ckpt_stem: str):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from callbacks import EER_Callback
    from callbacks_rational import BinaryACC_Callback, BinaryAUC_Callback

    log_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EER_Callback(batch_key="label", output_key="logit"),
        BinaryACC_Callback(batch_key="label", output_key="logit"),
        BinaryAUC_Callback(batch_key="label", output_key="logit"),
        ModelCheckpoint(dirpath=str(CKPT_DIR),
                        filename=ckpt_stem + "-best-{epoch:02d}-{val-eer:.4f}",
                        monitor="val-eer", mode="min", save_last=False, verbose=True),
    ]
    logger = CSVLogger(save_dir=str(log_dir), name="", version="",
                       flush_logs_every_n_steps=10)
    # bf16-mixed: A100 runs the frozen WavLM forward (dominant cost) much faster.
    # save_last is off so concurrent runs don't fight over a shared last.ckpt.
    return pl.Trainer(accelerator="gpu", devices=1, max_epochs=max_epochs,
                      precision="bf16-mixed",
                      logger=logger, callbacks=callbacks,
                      log_every_n_steps=10, deterministic=False)


def base_cfg():
    """Standard robust_goat config Namespace used by every ablation."""
    from train_mlaad_adversarial import HP
    return Namespace(PhonemeGAT=Namespace(
        backbone=HP["backbone"], use_raw=False,
        use_GAT=HP["use_GAT"], n_edges=HP["n_edges"],
        use_aug=HP["use_aug"], use_pool=HP["use_pool"], use_clip=HP["use_clip"]))


def run_seeds(model_factory, name: str, out_dir: Path, seeds, max_epochs: int,
              lr: float, criteria_fn=None) -> dict:
    """
    Generic fine-tune → evaluate → stratify driver shared by P5/P6/P7.

    model_factory(seed) -> a ready Lightning module (warm-started + ablation
    configured). This function owns the training loop, per-system EER eval, the
    C/T-quartile stratification, and summary.md. Returns {seed: eer_dict}.
    """
    import numpy as np
    import pandas as pd
    from train_mlaad_adversarial import set_seed

    out_dir.mkdir(parents=True, exist_ok=True)
    all_seed_eers = {}

    for seed in seeds:
        set_seed(seed)
        print(f"\n{'='*60}\n{name} — seed {seed}  (epochs={max_epochs}, lr={lr})\n{'='*60}")
        train_dl, val_dl = build_dataloaders(seed)
        model = model_factory(seed)
        model.lr = lr

        ckpt_stem = f"{name}_seed{seed}"
        trainer = make_trainer(max_epochs, out_dir / f"logs_seed{seed}", ckpt_stem)
        trainer.fit(model, train_dl, val_dl)
        trainer.save_checkpoint(str(CKPT_DIR / f"{ckpt_stem}.ckpt"))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metrics = evaluate_on_test(model, device,
                                   out_csv=out_dir / f"per_system_metrics_seed{seed}.csv")
        valid = [m for m in metrics.values() if m is not None]
        print(f"  [{name} seed {seed}] test mean EER: "
              f"{np.mean([m['eer'] for m in valid]):.4f}  "
              f"AUC: {np.mean([m['auc'] for m in valid]):.4f}  "
              f"bal_acc: {np.mean([m['bal_acc'] for m in valid]):.4f} ({len(valid)} systems)")
        all_seed_eers[seed] = metrics

    # Mean of each metric per system across seeds
    systems = sorted({s for d in all_seed_eers.values() for s in d})
    mean_metrics = {}
    for s in systems:
        ms = [d[s] for d in all_seed_eers.values() if d.get(s) is not None]
        if not ms:
            mean_metrics[s] = None; continue
        mean_metrics[s] = {k: float(np.mean([m[k] for m in ms if m.get(k) is not None]))
                           for k in METRIC_KEYS if any(m.get(k) is not None for m in ms)}

    baseline = load_baseline_metrics()
    ct = load_system_ct()
    report = stratify_by_ct(mean_metrics, baseline, ct)

    # Verdict against pre-registered criteria (operates on the stratified report)
    verdict = criteria_fn(report) if criteria_fn else ""

    md = [f"# {name} — Summary", "",
          f"seeds={list(seeds)}  epochs={max_epochs}  lr={lr}", "",
          stratified_markdown(report, "Per-system metrics vs mlaad_robust_goat baseline")]
    if verdict:
        md += ["", "## Verdict (EER-based, cross-checked against AUC/accuracy above)", verdict]
    (out_dir / "summary.md").write_text("\n".join(md))

    # Machine-readable per-system table (all metrics + baseline + C/T)
    rows = []
    for s in systems:
        if mean_metrics[s] is None:
            continue
        row = {"system": s, "C": ct.get(s, {}).get("C"), "T": ct.get(s, {}).get("T")}
        for k in METRIC_KEYS:
            row[f"model_{k}"] = mean_metrics[s].get(k)
            row[f"baseline_{k}"] = baseline.get(s, {}).get(k)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "per_system_metrics_mean.csv", index=False)

    print(f"\n[{name}] summary → {out_dir/'summary.md'}")
    if verdict:
        print(verdict)
    return all_seed_eers


def load_lit_state_from_base(lit, base_ckpt: Path = BASE_CKPT, strict: bool = False):
    """Load a robust_goat checkpoint's state_dict into a (same-shape) Lightning
    module. For shape-compatible variants (P5/P6/P7 don't change cls_head) this
    is an exact warm-start."""
    ckpt = torch.load(str(base_ckpt), map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    missing, unexpected = lit.load_state_dict(sd, strict=strict)
    if missing:
        print(f"  [warm-start] missing keys (random init): {list(missing)[:5]}")
    if unexpected:
        print(f"  [warm-start] unexpected keys (ignored): {list(unexpected)[:5]}")
    print(f"  [warm-start] loaded base weights from {base_ckpt.name}")
    return lit
