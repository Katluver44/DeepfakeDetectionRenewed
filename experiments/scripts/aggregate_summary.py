#!/usr/bin/env python3
"""
aggregate_summary.py
====================
Build an experiment's multi-metric summary.md from its per-seed metric CSVs
(per_system_metrics_seed*.csv). This decouples summary generation from training
so the 3 seeds of an experiment can run as independent parallel jobs and be
aggregated afterwards.

Usage:
    python aggregate_summary.py --dir <results_dir> --name "P5 ..." \
        [--baseline-csv <csv>] [--motivation "text"]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _ablation_common import (
    load_baseline_metrics, load_system_ct, stratify_by_ct, stratified_markdown,
    consistency_note, METRIC_KEYS,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--baseline-csv", type=Path, default=None)
    ap.add_argument("--motivation", type=str, default="")
    args = ap.parse_args()

    seed_csvs = sorted(args.dir.glob("per_system_metrics_seed*.csv"))
    if not seed_csvs:
        raise SystemExit(f"No per_system_metrics_seed*.csv in {args.dir}")

    # Mean of each metric per system across the available seeds.
    per_seed = {}
    for c in seed_csvs:
        df = pd.read_csv(c)
        for _, r in df.iterrows():
            d = per_seed.setdefault(r["system"], {k: [] for k in METRIC_KEYS})
            for k in METRIC_KEYS:
                if k in df.columns and not pd.isna(r[k]):
                    d[k].append(float(r[k]))
    mean_metrics = {s: {k: float(np.mean(v)) for k, v in d.items() if v}
                    for s, d in per_seed.items()}

    baseline = load_baseline_metrics(args.baseline_csv) if args.baseline_csv \
        else load_baseline_metrics()
    ct = load_system_ct()
    report = stratify_by_ct(mean_metrics, baseline, ct)

    md = [f"# {args.name} — Summary", "",
          f"seeds aggregated: {[c.name.split('seed')[-1].split('.')[0] for c in seed_csvs]}", ""]
    if args.motivation:
        md += ["## Motivation", args.motivation, ""]
    md += [stratified_markdown(report, "Per-system metrics vs mlaad_robust_goat baseline")]
    (args.dir / "summary.md").write_text("\n".join(md))

    rows = []
    for s, m in mean_metrics.items():
        row = {"system": s, "C": ct.get(s, {}).get("C"), "T": ct.get(s, {}).get("T")}
        for k in METRIC_KEYS:
            row[f"model_{k}"] = m.get(k)
            row[f"baseline_{k}"] = baseline.get(s, {}).get(k)
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.dir / "per_system_metrics_mean.csv", index=False)
    print(f"[aggregate] {args.name}: {len(seed_csvs)} seeds → {args.dir/'summary.md'}")
    print("  " + consistency_note(report, "C"))


if __name__ == "__main__":
    main()
