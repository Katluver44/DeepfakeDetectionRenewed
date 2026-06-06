#!/usr/bin/env python3
"""
Baseline evaluation of both MLAAD checkpoints on in-distribution and
cross-language test sets.

Usage:
    python experiments/scripts/eval_mlaad_baseline.py

Reads:
    experiments/checkpoints/mlaad_goat.ckpt
    experiments/checkpoints/mlaad_robust_goat.ckpt
    experiments/data/mlaad_tiny_processed/splits/test.json

Writes:
    experiments/results/mlaad/baseline_eval/
        eval_in_distribution.json   pooled + per-system EER on English test
        eval_cross_language.json    pooled + per-system EER on German test
        summary.json                headline EERs and cross-language gap per checkpoint
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ─── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_orig_torch_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)
torch.load = _patched_load

import torch.serialization
from argparse import Namespace
from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination
from pandas import Series
torch.serialization.add_safe_globals([Namespace, Phonemer_Tokenizer_Recombination, Series])

from phoneme_GAT.modules import Phoneme_GAT_lit
from loader import TARGET_SR
from callbacks import compute_eer

# ─── Constants ───────────────────────────────────────────────────────────────
def _best_or_last(stem: str) -> Path:
    """Return best-epoch checkpoint if present, else fall back to stem.ckpt."""
    ckpt_dir = PROJECT_ROOT / "experiments/checkpoints"
    candidates = sorted(ckpt_dir.glob(f"{stem}-best-*.ckpt"))
    return candidates[0] if candidates else ckpt_dir / f"{stem}.ckpt"

CHECKPOINTS = {
    "mlaad_goat": _best_or_last("mlaad_goat"),
    "mlaad_robust_goat": _best_or_last("mlaad_robust_goat"),
}
PROCESSED_DIR = PROJECT_ROOT / "experiments/data/mlaad_tiny_processed"
OUT_DIR = PROJECT_ROOT / "experiments/results/mlaad/baseline_eval"

# ─── Dataset ─────────────────────────────────────────────────────────────────

class MAALDEvalDataset(Dataset):
    """Loads preprocessed .pt tensors for evaluation (no augmentation)."""

    def __init__(self, records: list[dict], processed_dir: Path):
        self.records = records
        self.processed_dir = processed_dir

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        tensor = torch.load(self.processed_dir / rec["audio_path"])  # (48000,)
        wav = tensor.unsqueeze(0)  # (1, 48000)
        y = 0 if rec["label"] == "bonafide" else 1
        return {
            "audio": wav,
            "label": torch.tensor(y, dtype=torch.long),
            "sample_rate": TARGET_SR,
            "attack_system": rec.get("attack_system", "unknown"),
            "language": rec.get("language", "unknown"),
            "sample_id": rec["sample_id"],
        }


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate that keeps string fields as lists."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        else:
            out[k] = vals
    return out


# ─── Inference ───────────────────────────────────────────────────────────────

def run_inference(
    model: Phoneme_GAT_lit,
    records: list[dict],
    processed_dir: Path,
    batch_size: int = 32,
    device: str = "cuda",
) -> tuple[list[int], list[float], list[str], list[str]]:
    """
    Returns (labels, logits, attack_systems, languages) for all records.
    """
    ds = MAALDEvalDataset(records, processed_dir)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = model.to(device)
    model.eval()

    all_labels: list[int] = []
    all_logits: list[float] = []
    all_systems: list[str] = []
    all_langs: list[str] = []

    with torch.no_grad():
        for batch in dl:
            audio = batch["audio"].to(device)
            B = audio.shape[0]
            num_frames = torch.full((B,), 48000 // 320 - 1, device=device)

            out = model.model(
                audio, num_frames,
                profiler=None,
                use_aug=False,
                stage="val",
                ground_truth_labels=batch["label"].to(device),
            )
            logits = out["logit"].cpu().float()

            all_labels.extend(batch["label"].tolist())
            all_logits.extend(logits.tolist())
            all_systems.extend(batch["attack_system"])
            all_langs.extend(batch["language"])

    return all_labels, all_logits, all_systems, all_langs


# ─── EER computation ──────────────────────────────────────────────────────────

def compute_eer_safe(labels: list[int], scores: list[float]) -> float | None:
    """Compute EER; returns None if either class is absent."""
    y = np.array(labels)
    s = np.array(scores)
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(compute_eer(y, s, positive_label=1))
    except Exception:
        return None


def per_system_eer(
    labels: list[int],
    logits: list[float],
    systems: list[str],
    bonafide_labels: list[int],
    bonafide_logits: list[float],
) -> dict[str, float | None]:
    """
    For each attack system: system spoof + ALL bonafide → EER.
    Returns {system_name: eer}.
    """
    # Group spoof by system
    system_groups: dict[str, tuple[list[int], list[float]]] = defaultdict(
        lambda: ([], [])
    )
    for y, s, sys in zip(labels, logits, systems):
        if sys == "bonafide":
            continue
        system_groups[sys][0].append(y)
        system_groups[sys][1].append(s)

    results = {}
    for sys_name, (sp_labels, sp_logits) in sorted(system_groups.items()):
        # Combine system spoof + all bonafide
        combined_labels = sp_labels + bonafide_labels
        combined_logits = sp_logits + bonafide_logits
        results[sys_name] = compute_eer_safe(combined_labels, combined_logits)

    return results


# ─── Main evaluation ─────────────────────────────────────────────────────────

def evaluate_checkpoint(
    ckpt_name: str,
    ckpt_path: Path,
    in_dist_records: list[dict],
    cross_lang_records: list[dict],
    processed_dir: Path,
    device: str = "cuda",
) -> dict:
    """
    Evaluate one checkpoint on both test sets.
    Returns a dict with all results.
    """
    print(f"\n  Loading {ckpt_name} from {ckpt_path} …")
    cfg = Namespace(
        PhonemeGAT=Namespace(
            backbone="wavlm",
            use_raw=False,
            use_GAT=True,
            n_edges=10,
            use_aug=True,
            use_pool=True,
            use_clip=True,
        )
    )
    model = Phoneme_GAT_lit.load_from_checkpoint(str(ckpt_path), cfg=cfg)
    model.eval()

    result = {"checkpoint": ckpt_name, "checkpoint_path": str(ckpt_path)}

    for split_name, records in [
        ("in_distribution", in_dist_records),
        ("cross_language", cross_lang_records),
    ]:
        print(f"  Evaluating {ckpt_name} on {split_name} ({len(records)} samples) …")
        t0 = time.time()
        labels, logits, systems, langs = run_inference(
            model, records, processed_dir, device=device
        )
        elapsed = time.time() - t0

        # Sanity check (d): cross-language records should contain German files
        if split_name == "cross_language":
            sample_langs = set(langs)
            has_german = "de" in sample_langs
            if not has_german:
                print(f"    WARNING: cross_language split has no German files! Languages: {sample_langs}")
            else:
                n_de = sum(1 for l in langs if l == "de")
                print(f"    German files in cross_language: {n_de}/{len(langs)} ✓")

        # Separate bonafide for per-system computation
        bf_idx = [i for i, y in enumerate(labels) if y == 0]
        bf_labels = [labels[i] for i in bf_idx]
        bf_logits = [logits[i] for i in bf_idx]

        # Pooled EER
        pooled_eer = compute_eer_safe(labels, logits)

        # Per-system EER
        sys_eers = per_system_eer(labels, logits, systems, bf_labels, bf_logits)

        result[split_name] = {
            "n_samples": len(records),
            "n_bonafide": len(bf_idx),
            "n_spoof": len(records) - len(bf_idx),
            "pooled_eer": pooled_eer,
            "per_system_eer": sys_eers,
            "inference_time_seconds": elapsed,
        }
        print(
            f"    Pooled EER: {pooled_eer:.4f} | "
            f"bonafide={len(bf_idx)} spoof={len(records)-len(bf_idx)} | "
            f"{elapsed:.1f}s"
        )

    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--processed-dir", type=Path, default=PROCESSED_DIR,
    )
    p.add_argument(
        "--goat-ckpt", type=Path, default=CHECKPOINTS["mlaad_goat"],
    )
    p.add_argument(
        "--robust-ckpt", type=Path, default=CHECKPOINTS["mlaad_robust_goat"],
    )
    p.add_argument(
        "--out-dir", type=Path, default=OUT_DIR,
    )
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load test records ────────────────────────────────────────────────────
    test_records = json.loads((args.processed_dir / "splits/test.json").read_text())
    print(f"Test set: {len(test_records)} records total")

    # ── Sanity check (d) language assertion ──────────────────────────────────
    in_dist = [r for r in test_records if r["language"] == "en"]
    cross_lang_spoof = [r for r in test_records if r["language"] == "de"]
    cross_lang_bonafide = [r for r in test_records if r["language"] == "en" and r["label"] == "bonafide"]

    # Cross-language: German spoof + English bonafide (as negative class reference)
    cross_lang = cross_lang_spoof + cross_lang_bonafide

    print(f"In-distribution (EN):  {len(in_dist)} records "
          f"({sum(1 for r in in_dist if r['label']=='bonafide')} bonafide, "
          f"{sum(1 for r in in_dist if r['label']=='spoof')} spoof)")
    print(f"Cross-language (DE spoof + EN bonafide): {len(cross_lang)} records "
          f"({len(cross_lang_bonafide)} bonafide, {len(cross_lang_spoof)} DE spoof)")

    # Write out split files for reference
    (args.out_dir / "test_in_distribution.json").write_text(json.dumps(in_dist, indent=2))
    (args.out_dir / "test_cross_language.json").write_text(json.dumps(cross_lang, indent=2))
    print("Wrote test split files to out_dir.")

    # ── Check both checkpoints exist ─────────────────────────────────────────
    for name, path in [("mlaad_goat", args.goat_ckpt), ("mlaad_robust_goat", args.robust_ckpt)]:
        if not path.exists():
            print(f"ERROR: Checkpoint not found: {path}")
            print("Run training scripts first.")
            sys.exit(1)

    # ── Evaluate both checkpoints ────────────────────────────────────────────
    all_results = {}
    for name, ckpt_path in [
        ("mlaad_goat", args.goat_ckpt),
        ("mlaad_robust_goat", args.robust_ckpt),
    ]:
        result = evaluate_checkpoint(
            name, ckpt_path, in_dist, cross_lang, args.processed_dir, device=device
        )
        all_results[name] = result

    # ── Write per-split eval files ────────────────────────────────────────────
    in_dist_out = {}
    cross_lang_out = {}
    for name, res in all_results.items():
        in_dist_out[name] = res["in_distribution"]
        cross_lang_out[name] = res["cross_language"]

    (args.out_dir / "eval_in_distribution.json").write_text(
        json.dumps(in_dist_out, indent=2)
    )
    (args.out_dir / "eval_cross_language.json").write_text(
        json.dumps(cross_lang_out, indent=2)
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = {}
    for name, res in all_results.items():
        in_eer = res["in_distribution"]["pooled_eer"]
        cl_eer = res["cross_language"]["pooled_eer"]
        gap = (cl_eer - in_eer) if (in_eer is not None and cl_eer is not None) else None
        summary[name] = {
            "in_distribution_eer": in_eer,
            "cross_language_eer": cl_eer,
            "cross_language_gap": gap,
        }

    # Sanity check (a): non-trivial in-distribution EER
    for name, s in summary.items():
        eer = s["in_distribution_eer"]
        if eer is None or eer >= 0.30:
            print(f"  [FAIL] (a) {name} in-dist EER={eer} — training failure threshold exceeded (>= 0.30)")
        else:
            print(f"  [PASS] (a) {name} in-dist EER={eer:.4f} < 0.30")

    # Sanity check (b): checkpoints differ
    try:
        sd1 = torch.load(str(args.goat_ckpt), map_location="cpu")["state_dict"]
        sd2 = torch.load(str(args.robust_ckpt), map_location="cpu")["state_dict"]
        keys = [k for k in sd1 if k in sd2 and sd1[k].is_floating_point()]
        diffs = [(sd1[k] - sd2[k]).abs().max().item() for k in keys]
        weights_differ = any(d > 1e-6 for d in diffs)
        n_differing = sum(1 for d in diffs if d > 1e-6)
        if weights_differ:
            print(f"  [PASS] (b) checkpoints differ in {n_differing}/{len(keys)} param tensors (GAT/head layers)")
        else:
            print("  [FAIL] (b) checkpoints appear identical — training may not have run correctly")
    except Exception as e:
        print(f"  [WARN] (b) Could not compare weights: {e}")

    # Sanity check (c): convergence — check training curves
    for name in ["mlaad_goat", "mlaad_robust_goat"]:
        csv_path = PROJECT_ROOT / f"experiments/results/mlaad/training_logs/{name}/training_curves.csv"
        if csv_path.exists():
            import csv
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            if len(rows) >= 2:
                eers = [float(r["val_eer"]) for r in rows if r.get("val_eer")]
                if len(eers) >= 2:
                    converged = eers[-1] <= eers[0]  # EER should decrease or plateau
                    status = "PASS" if converged else "WARN"
                    print(f"  [{status}] (c) {name} val EER: {eers[0]:.4f} → {eers[-1]:.4f}")

    # Sanity check (e): speaker disjoint in test splits
    train_records = json.loads((args.processed_dir / "splits/train.json").read_text())
    train_speakers = {r["speaker_id"] for r in train_records}
    test_speakers = {r["speaker_id"] for r in test_records}
    leaked = train_speakers & test_speakers
    if not leaked:
        print(f"  [PASS] (e) speaker IDs disjoint: no train speakers in test")
    else:
        print(f"  [FAIL] (e) speaker leak: {sorted(leaked)}")

    # Hypothesis check: does robust_goat have smaller gap?
    goat_gap = summary.get("mlaad_goat", {}).get("cross_language_gap")
    robust_gap = summary.get("mlaad_robust_goat", {}).get("cross_language_gap")
    hypothesis_confirmed = (
        goat_gap is not None
        and robust_gap is not None
        and robust_gap < goat_gap
    )

    summary["hypothesis"] = {
        "prediction": "mlaad_robust_goat has smaller cross-language EER gap than mlaad_goat",
        "confirmed": hypothesis_confirmed,
        "mlaad_goat_gap": goat_gap,
        "mlaad_robust_goat_gap": robust_gap,
        "note": (
            "Smaller gap = better language-invariant generalization "
            "(abstraction-over-discriminability story)"
        ),
    }

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ── Print headline results ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MLAAD BASELINE EVALUATION SUMMARY")
    print("=" * 60)
    for name in ["mlaad_goat", "mlaad_robust_goat"]:
        s = summary[name]
        print(f"\n  {name}:")
        print(f"    In-distribution EER:  {s['in_distribution_eer']:.4f}")
        print(f"    Cross-language EER:   {s['cross_language_eer']:.4f}")
        print(f"    Cross-language gap:   {s['cross_language_gap']:+.4f}")

    print(f"\n  Hypothesis (robust_goat gap < goat gap):")
    print(f"    mlaad_goat gap:        {goat_gap:+.4f}")
    print(f"    mlaad_robust_goat gap: {robust_gap:+.4f}")
    confirmed_str = "YES ✓" if hypothesis_confirmed else "NO ✗"
    print(f"    Confirmed: {confirmed_str}")
    print("=" * 60)
    print(f"\nResults written to: {args.out_dir}")


if __name__ == "__main__":
    main()
