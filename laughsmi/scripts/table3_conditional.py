"""Build Stage 3 conditional-detector results (LaugHSMI plan §6).

The input score must contain one probability per utterance, where a larger
value means more likely spoof.  Laughter presence is defined from the Stage-1
CSV at its native 0.5 operating point (``n_laugh_segs >= 1``).

For each of the laughter-present and laughter-absent subsets the script
reports ROC AUC and EER only when both classes have at least ``--min-per-class``
examples.  Otherwise it reports the bona-fide-versus-spoof score comparison
(Mann--Whitney U, including its AUC-equivalent effect size).  If masked scores
are supplied, it also reports paired changes in spoof probability for
laughter-containing files.  An optional Bark score CSV is assessed at the
ITW equal-error threshold.

Example:
  python scripts/table3_conditional.py \
    --scores detector_out/itw_scores.csv \
    --laughter detector_out/itw_laughter.csv \
    --masked detector_out/itw_scores_masked.csv \
    --out-csv tables/table3.csv --out-tex tables/table3.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve


LABELS = {"bona-fide": 0, "spoof": 1}


def _validate_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def eer_threshold(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Return (EER, threshold) by choosing the ROC point with min |FPR-FNR|."""
    fpr, tpr, thresholds = roc_curve(y_true, score)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2.0), float(thresholds[i])


def score_subset(frame: pd.DataFrame, subset: str, min_per_class: int) -> dict:
    """Compute the pre-specified conditional result for one subset."""
    bona = frame.loc[frame.label == "bona-fide", "score"].to_numpy(float)
    spoof = frame.loc[frame.label == "spoof", "score"].to_numpy(float)
    row = {
        "analysis": "conditional_scores", "subset": subset,
        "n_bona": len(bona), "n_spoof": len(spoof),
        "mean_score_bona": np.mean(bona) if len(bona) else np.nan,
        "mean_score_spoof": np.mean(spoof) if len(spoof) else np.nan,
        "auc": np.nan, "eer": np.nan, "eer_threshold": np.nan,
        "mannwhitney_u": np.nan, "mannwhitney_p": np.nan,
        "auc_equivalent": np.nan, "status": "insufficient_classes",
    }
    if not len(bona) or not len(spoof):
        return row
    y = frame.label.map(LABELS).to_numpy(int)
    s = frame.score.to_numpy(float)
    auc = float(roc_auc_score(y, s))
    u, p = stats.mannwhitneyu(spoof, bona, alternative="two-sided")
    row.update({"auc": auc, "mannwhitney_u": float(u), "mannwhitney_p": float(p),
                "auc_equivalent": float(u / (len(bona) * len(spoof)))})
    if len(bona) >= min_per_class and len(spoof) >= min_per_class:
        eer, threshold = eer_threshold(y, s)
        row.update({"eer": eer, "eer_threshold": threshold, "status": "auc_eer"})
    else:
        row["status"] = "mann_whitney_only"
    return row


def masking_result(scores: pd.DataFrame, masked: pd.DataFrame, laugh_ids: set[str]) -> dict:
    masked = masked.rename(columns={"score_masked": "masked_score"})
    merged = scores.merge(masked[["file_id", "masked_score"]], on="file_id", how="inner")
    merged = merged[merged.file_id.isin(laugh_ids)].dropna(subset=["score", "masked_score"])
    delta = (merged.masked_score - merged.score).to_numpy(float)
    row = {"analysis": "masking_delta", "subset": "laughter_present",
           "n_bona": int((merged.label == "bona-fide").sum()),
           "n_spoof": int((merged.label == "spoof").sum()),
           "n_pairs": len(delta), "mean_delta_masked_minus_original": np.mean(delta) if len(delta) else np.nan,
           "median_delta_masked_minus_original": np.median(delta) if len(delta) else np.nan,
           "wilcoxon_statistic": np.nan, "wilcoxon_p": np.nan, "status": "no_pairs"}
    if len(delta) and np.any(delta != 0):
        statistic, p = stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
        row.update({"wilcoxon_statistic": float(statistic), "wilcoxon_p": float(p), "status": "paired_wilcoxon"})
    elif len(delta):
        row["status"] = "all_deltas_zero"
    return row


def bark_result(bark: pd.DataFrame, threshold: float) -> dict:
    score_col = "score" if "score" in bark.columns else "score_masked" if "score_masked" in bark.columns else None
    if score_col is None:
        raise ValueError("Bark score CSV needs a score or score_masked column")
    scores = pd.to_numeric(bark[score_col], errors="coerce").dropna().to_numpy(float)
    return {"analysis": "bark_detection_at_itw_eer", "subset": "bark_probe",
            "n_bark": len(scores), "eer_threshold": threshold,
            "detection_rate": np.mean(scores >= threshold) if len(scores) else np.nan,
            "mean_score": np.mean(scores) if len(scores) else np.nan,
            "status": "ok" if len(scores) else "no_scores"}


def write_latex(table: pd.DataFrame, path: Path) -> None:
    conditional = table[table.analysis == "conditional_scores"]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Conditional deepfake-detector scores. Scores are spoof probabilities. EER is reported only when each class has at least the pre-specified sample minimum.}",
             r"\label{tab:table3}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Subset & $N_B$ & $N_S$ & AUC & EER & $p_{MWU}$ & Result \\", r"\midrule"]
    for _, r in conditional.iterrows():
        def num(x, digits=3): return "--" if pd.isna(x) else f"{x:.{digits}f}"
        p = num(r.mannwhitney_p)
        if not pd.isna(r.mannwhitney_p) and r.mannwhitney_p < .001: p = "$<$0.001"
        result = "AUC/EER" if r.status == "auc_eer" else "MWU only"
        lines.append(f"{str(r.subset).replace('_', ' ')} & {r.n_bona} & {r.n_spoof} & {num(r.auc)} & {num(r.eer)} & {p} & {result} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--laughter", type=Path, required=True)
    parser.add_argument("--masked", type=Path, default=None)
    parser.add_argument("--bark-scores", type=Path, default=None)
    parser.add_argument("--min-per-class", type=int, default=100)
    parser.add_argument("--out-csv", type=Path, default=Path("tables/table3.csv"))
    parser.add_argument("--out-tex", type=Path, default=Path("tables/table3.tex"))
    args = parser.parse_args()

    scores, laughter = pd.read_csv(args.scores), pd.read_csv(args.laughter)
    _validate_columns(scores, {"file_id", "label", "score"}, args.scores)
    _validate_columns(laughter, {"file_id", "n_laugh_segs"}, args.laughter)
    scores = scores[scores.label.isin(LABELS)].copy()
    scores.score = pd.to_numeric(scores.score, errors="coerce")
    scores = scores.dropna(subset=["score"]).drop_duplicates("file_id", keep="last")
    laugh_ids = set(laughter.loc[pd.to_numeric(laughter.n_laugh_segs, errors="coerce").fillna(0) >= 1, "file_id"])
    scores["laughter_present"] = scores.file_id.isin(laugh_ids)

    rows = [score_subset(scores[scores.laughter_present], "laughter_present", args.min_per_class),
            score_subset(scores[~scores.laughter_present], "laughter_absent", args.min_per_class)]
    # The all-ITW operating threshold is the principled threshold for the Bark probe.
    all_result = score_subset(scores, "all_itw", args.min_per_class)
    rows.append(all_result)
    if args.masked:
        rows.append(masking_result(scores, pd.read_csv(args.masked), laugh_ids))
    if args.bark_scores:
        if np.isnan(all_result["eer_threshold"]):
            raise ValueError("Bark detection requires an all-ITW EER threshold; score both ITW classes first.")
        rows.append(bark_result(pd.read_csv(args.bark_scores), all_result["eer_threshold"]))

    table = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False)
    write_latex(table, args.out_tex)
    print(f"Wrote {args.out_csv} ({len(table)} rows)")
    print(f"Wrote {args.out_tex}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
