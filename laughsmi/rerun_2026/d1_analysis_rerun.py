"""Reproduce the D1 evasion table for the 2026-07-11 rerun, using the exact
pairing/statistics logic of scripts/d1_analysis.py::analyze(), pointed at the
rerun's own base/aug score CSVs and manifest.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_curve

RERUN = Path(__file__).resolve().parent
LAUGHSMI = RERUN.parent


def load_scores(path):
    d = {}
    for r in csv.DictReader(open(path)):
        d[Path(r["file_id"]).name] = (float(r["score"]), r["label"])
    return d


def clean_eer_threshold(base):
    bona = [s for s, l in base.values() if l == "bona-fide"]
    spoof = [s for s, l in base.values() if l == "spoof"]
    s = np.array(bona + spoof); y = np.array([0]*len(bona) + [1]*len(spoof))
    fpr, tpr, thr = roc_curve(y, s); fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float(thr[i]), float((fpr[i] + fnr[i]) / 2)


def rank_biserial_from_wilcoxon(deltas):
    d = deltas[deltas != 0]
    if len(d) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    r_plus = ranks[d > 0].sum(); r_minus = ranks[d < 0].sum()
    total = ranks.sum()
    return float((r_plus - r_minus) / total)


def analyze(name, base_csv, aug_csv, manifest_csv):
    base = load_scores(base_csv); aug = load_scores(aug_csv)
    manifest = {Path(r["file"]).name: r for r in csv.DictReader(open(manifest_csv))}
    thr, eer = clean_eer_threshold(base)

    aug_ids = [k for k, r in manifest.items() if r["augmented"] == "1"]
    base_s = np.array([base[k][0] for k in aug_ids])
    aug_s = np.array([aug[k][0] for k in aug_ids])
    delta = aug_s - base_s

    W = wilcoxon(base_s, aug_s) if np.any(delta != 0) else None
    rb = rank_biserial_from_wilcoxon(delta)

    evasion_rate = float(np.mean(aug_s < thr))
    base_catch = float(np.mean(base_s >= thr))
    aug_catch = float(np.mean(aug_s >= thr))
    newly_evaded = int(np.sum((base_s >= thr) & (aug_s < thr)))

    by_pos = {}
    for pos in ["start", "mid", "end"]:
        idx = [i for i, k in enumerate(aug_ids) if manifest[k]["position"] == pos]
        if idx:
            by_pos[pos] = (float(np.mean(delta[idx])), len(idx))

    res = {
        "dataset": name, "n_aug_fakes": len(aug_ids), "clean_eer_pct": round(eer*100, 2),
        "thr_at_eer": round(thr, 4),
        "base_spoof_score_mean": round(float(base_s.mean()), 4),
        "aug_spoof_score_mean": round(float(aug_s.mean()), 4),
        "delta_mean": round(float(delta.mean()), 4), "delta_median": round(float(np.median(delta)), 4),
        "wilcoxon_p": (round(float(W.pvalue), 6) if W else None),
        "rank_biserial": round(rb, 3),
        "base_catch_rate": round(base_catch, 3), "aug_catch_rate": round(aug_catch, 3),
        "evasion_rate_after": round(evasion_rate, 3), "newly_evaded_n": newly_evaded,
        "delta_start": by_pos.get("start", (None, 0))[0],
        "delta_mid": by_pos.get("mid", (None, 0))[0],
        "delta_end": by_pos.get("end", (None, 0))[0],
    }
    return res


def main():
    out = RERUN / "tables" / "table_d1_rerun.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res = analyze(
        "ASVspoof19 (asv19 model) -- rerun 2026-07-11",
        RERUN / "base_asv19.csv", RERUN / "aug_asv19.csv",
        LAUGHSMI / "data" / "eval_asv19_aug" / "manifest.csv",
    )
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res.keys())); w.writeheader(); w.writerow(res)
    print("\n===", res["dataset"], "===")
    for k, v in res.items():
        if k != "dataset":
            print(f"  {k}: {v}")
    print(f"\nwrote {out}")

    # comparison line vs prior D1 result
    print("\n--- vs prior (RESULTS.md D1) ---")
    print("prior: clean EER 5.33%, base 0.899 -> aug 0.604, delta=-0.295, evasion=0.20 (Bark inserts)")
    print(f"rerun: clean EER {res['clean_eer_pct']}%, base {res['base_spoof_score_mean']} -> "
          f"aug {res['aug_spoof_score_mean']}, delta={res['delta_mean']}, evasion={res['evasion_rate_after']} (Bark inserts)")


if __name__ == "__main__":
    main()
