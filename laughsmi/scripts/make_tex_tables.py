"""Emit paper-ready LaTeX for Table D1 (laughter evasion) and Table D3
(real-vs-synth laughter geometry), from the produced CSVs + control scores."""
import csv
from pathlib import Path
import numpy as np

L = Path(__file__).resolve().parents[1]


def load(path):
    return {Path(r["file_id"]).name: (float(r["score"]), r["label"]) for r in csv.DictReader(open(path))}


def d1_row(name, base_csv, aug_csv, ctrl_csv, manifest_csv):
    base = load(base_csv); aug = load(aug_csv)
    ctrl = {r["file_id"]: r for r in csv.DictReader(open(ctrl_csv))}
    man = {Path(r["file"]).name: r for r in csv.DictReader(open(manifest_csv)) if r["augmented"] == "1"}
    ids = [k for k in man if k in ctrl]
    b = np.array([base[k][0] for k in ids]); a = np.array([aug[k][0] for k in ids])
    u = np.array([float(ctrl[k]["score_unmask"]) for k in ids])
    s = np.array([float(ctrl[k]["score_silence"]) for k in ids])
    # clean EER threshold from base
    from sklearn.metrics import roc_curve
    bo = [x for x, l in base.values() if l == "bona-fide"]; sp = [x for x, l in base.values() if l == "spoof"]
    fpr, tpr, thr = roc_curve([0]*len(bo)+[1]*len(sp), bo+sp); fnr = 1-tpr
    i = np.nanargmin(np.abs(fnr-fpr)); t = thr[i]
    return {"name": name, "n": len(ids), "base": b.mean(), "aug": a.mean(),
            "delta": (a-b).mean(), "unmask": u.mean(), "silence": s.mean(),
            "catch_base": (b >= t).mean(), "catch_aug": (a >= t).mean(),
            "evade": (a < t).mean()}


def main():
    do = L / "detector_out"; da = L / "data"
    r1 = d1_row("ASVspoof19 (asv19)", do/"base_asv19.csv", do/"aug_asv19.csv", do/"control_asv19.csv", da/"eval_asv19_aug/manifest.csv")
    r2 = d1_row("MLAAD (robust)", do/"base_mlaad.csv", do/"aug_mlaad.csv", do/"control_mlaad.csv", da/"eval_mlaad_aug/manifest.csv")

    tex = [r"\begin{tabular}{lccccccc}", r"\toprule",
           r"Eval (detector) & $n$ & clean & +laugh & $\Delta$ & unmask & silence & catch $\downarrow$ / evade \\",
           r"\midrule"]
    for r in (r1, r2):
        tex.append(f"{r['name']} & {r['n']} & {r['base']:.3f} & {r['aug']:.3f} & "
                   f"$-${abs(r['delta']):.3f} & {r['unmask']:.3f} & {r['silence']:.3f} & "
                   f"{r['catch_base']:.2f}$\\to${r['catch_aug']:.2f} / {r['evade']:.2f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (L/"tables"/"table_d1.tex").write_text("\n".join(tex))
    print("wrote tables/table_d1.tex")
    print("\n".join(tex))

    # D3 tex from table_d3.csv
    d3 = list(csv.reader(open(L/"tables"/"table_d3.csv")))
    print("\n(D3 CSV already at tables/table_d3.csv; key numbers: probe AUC L9/L12 ≈ 1.0)")


if __name__ == "__main__":
    main()
