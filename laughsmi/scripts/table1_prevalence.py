"""Table 1: laughter prevalence by class (bona-fide vs spoof) at multiple thresholds.

Implements laughsmi_plan.md §4 deliverable ("Table 1"):
For thresholds {0.3, 0.5, 0.7}, compute per class (bona-fide, spoof):
  - % files with laughter present (at thr=0.5 this uses n_laugh_segs >= 1;
    at thr=0.3/0.7 this is derived from max_prob >= thr, since the detector
    was only run once at thr=0.5 and max_prob lets us derive other operating
    points without re-running the detector, per plan §4).
  - mean laugh_ratio per class (restricted to files with laughter present at
    that threshold, since laugh_ratio/laugh_dur_s were computed at the
    original thr=0.5 segmentation; see --laugh-ratio-scope to change this).
  - Fisher's exact test (bona-fide "has laughter" vs spoof "has laughter").
  - Odds ratio (bona-fide relative to spoof) with 95% CI via the Woolf /
    Haldane-Anscombe corrected log-odds method (manual, no statsmodels
    dependency required, though statsmodels.stats.contingency_tables.Table2x2
    is used when available for a cross-check).

Input CSV schema (one row per ITW file), per plan §4:
    file_id,label,speaker,dur_s,n_laugh_segs,laugh_dur_s,laugh_ratio,
    max_prob,seg_starts,seg_ends
where label in {bona-fide, spoof}; seg_starts/seg_ends are ';'-joined floats;
max_prob is the per-file max frame probability from the laughter detector.

Outputs:
    tables/table1.csv  -- one row per (threshold, class) with all stats plus
                           the Fisher test / odds-ratio columns duplicated
                           across the two class-rows of the same threshold
                           for convenience.
    tables/table1.tex  -- a LaTeX tabular rendering of the same information.

Usage:
    python table1_prevalence.py --csv detector_out/itw_laughter.csv \
        --out-csv tables/table1.csv --out-tex tables/table1.tex
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

THRESHOLDS = (0.3, 0.5, 0.7)
CLASSES = ("bona-fide", "spoof")


def has_laughter_mask(df: pd.DataFrame, thr: float) -> pd.Series:
    """Boolean mask of 'file has laughter' at the given threshold.

    At thr == 0.5 uses n_laugh_segs >= 1 (the detector's native operating
    point). At other thresholds, derives presence from max_prob >= thr,
    since re-running the detector at other thresholds is unnecessary (plan §4).
    """
    if math.isclose(thr, 0.5):
        return df["n_laugh_segs"].fillna(0) >= 1
    return df["max_prob"].fillna(0.0) >= thr


def woolf_odds_ratio_ci(a: int, b: int, c: int, d: int, alpha: float = 0.05):
    """Odds ratio + Woolf (log-odds / Haldane-Anscombe corrected) 95% CI.

    2x2 table:
                 laughter present   laughter absent
        bona-fide      a                  b
        spoof          c                  d

    OR = (a*d) / (b*c), relative risk of bona-fide vs spoof having laughter.
    Uses Haldane-Anscombe correction (+0.5 to every cell) when any cell is 0,
    which is the standard fix for the Woolf logit-CI method.

    Returns:
        (odds_ratio, ci_low, ci_high)
    """
    cells = [a, b, c, d]
    if any(x == 0 for x in cells):
        a, b, c, d = (x + 0.5 for x in cells)
    odds_ratio = (a * d) / (b * c)
    log_or = math.log(odds_ratio)
    se_log_or = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = stats.norm.ppf(1 - alpha / 2)
    ci_low = math.exp(log_or - z * se_log_or)
    ci_high = math.exp(log_or + z * se_log_or)
    return odds_ratio, ci_low, ci_high


def statsmodels_cross_check(a: int, b: int, c: int, d: int):
    """Optional cross-check of the odds ratio + CI via statsmodels, if installed.

    Returns (or, ci_low, ci_high) or None if statsmodels is unavailable.
    """
    try:
        from statsmodels.stats.contingency_tables import Table2x2
    except ImportError:
        return None
    table = Table2x2([[a, b], [c, d]])
    or_est = table.oddsratio
    ci_low, ci_high = table.oddsratio_confint()
    return float(or_est), float(ci_low), float(ci_high)


def compute_threshold_stats(df: pd.DataFrame, thr: float) -> list[dict]:
    """Compute per-class + test stats for one threshold. Returns list of row dicts."""
    mask_present = has_laughter_mask(df, thr)
    rows = []

    counts = {}
    for cls in CLASSES:
        cls_df = df[df["label"] == cls]
        n_total = len(cls_df)
        n_present = int(mask_present.loc[cls_df.index].sum())
        counts[cls] = (n_present, n_total - n_present, n_total)

    a, b, _ = counts["bona-fide"]
    c, d, _ = counts["spoof"]

    if (a + b) > 0 and (c + d) > 0:
        oddsratio_fisher, p_value = stats.fisher_exact([[a, b], [c, d]])
    else:
        oddsratio_fisher, p_value = float("nan"), float("nan")

    if a and b and c and d:
        odds_ratio, ci_low, ci_high = woolf_odds_ratio_ci(a, b, c, d)
    else:
        odds_ratio, ci_low, ci_high = woolf_odds_ratio_ci(a, b, c, d)  # HA-corrected internally

    sm_check = statsmodels_cross_check(a, b, c, d)

    for cls in CLASSES:
        n_present, n_absent, n_total = counts[cls]
        pct_present = 100.0 * n_present / n_total if n_total else float("nan")
        cls_df = df[df["label"] == cls]
        present_mask_cls = mask_present.loc[cls_df.index]
        if present_mask_cls.any():
            mean_laugh_ratio = float(cls_df.loc[present_mask_cls, "laugh_ratio"].mean())
        else:
            mean_laugh_ratio = float("nan")
        mean_laugh_ratio_all = float(cls_df["laugh_ratio"].fillna(0.0).mean()) if n_total else float("nan")

        rows.append(
            {
                "threshold": thr,
                "class": cls,
                "n_total": n_total,
                "n_with_laughter": n_present,
                "pct_with_laughter": round(pct_present, 3),
                "mean_laugh_ratio_among_laughing": (
                    round(mean_laugh_ratio, 5) if not math.isnan(mean_laugh_ratio) else float("nan")
                ),
                "mean_laugh_ratio_all_files": round(mean_laugh_ratio_all, 5),
                "fisher_p_value": p_value,
                "fisher_odds_ratio": oddsratio_fisher,
                "odds_ratio_bona_vs_spoof": odds_ratio,
                "or_ci_low": ci_low,
                "or_ci_high": ci_high,
                "or_ci_method": "woolf_logit_haldane_anscombe",
                "statsmodels_or": sm_check[0] if sm_check else float("nan"),
                "statsmodels_ci_low": sm_check[1] if sm_check else float("nan"),
                "statsmodels_ci_high": sm_check[2] if sm_check else float("nan"),
            }
        )
    return rows


def build_table1(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full Table 1 dataframe across all thresholds."""
    required_cols = {
        "file_id", "label", "speaker", "dur_s", "n_laugh_segs",
        "laugh_dur_s", "laugh_ratio", "max_prob", "seg_starts", "seg_ends",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    bad_labels = set(df["label"].unique()) - set(CLASSES)
    if bad_labels:
        raise ValueError(f"Unexpected label values {bad_labels}; expected {CLASSES}")

    all_rows = []
    for thr in THRESHOLDS:
        all_rows.extend(compute_threshold_stats(df, thr))
    return pd.DataFrame(all_rows)


def fmt_p(p: float) -> str:
    if p != p:  # NaN
        return "--"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def write_latex(table1: pd.DataFrame, out_tex: Path) -> None:
    """Render Table 1 as a LaTeX tabular, one block per threshold."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Laughter prevalence in ITW by class, at three operating thresholds. "
                 r"Odds ratio (OR) is bona-fide relative to spoof, with 95\% Woolf-logit CI.}")
    lines.append(r"\label{tab:table1}")
    lines.append(r"\begin{tabular}{llrrrrl}")
    lines.append(r"\toprule")
    lines.append(r"Thr & Class & N & \% w/ laughter & Mean laugh ratio & Fisher $p$ & OR [95\% CI] \\")
    lines.append(r"\midrule")
    for thr in THRESHOLDS:
        sub = table1[table1["threshold"] == thr].reset_index(drop=True)
        for i, row in sub.iterrows():
            p_str = fmt_p(row["fisher_p_value"]) if i == 0 else ""
            or_str = (
                f"{row['odds_ratio_bona_vs_spoof']:.2f} "
                f"[{row['or_ci_low']:.2f}, {row['or_ci_high']:.2f}]"
                if i == 0 else ""
            )
            thr_str = f"{thr:.1f}" if i == 0 else ""
            lines.append(
                f"{thr_str} & {row['class']} & {row['n_total']} & "
                f"{row['pct_with_laughter']:.1f} & "
                f"{row['mean_laugh_ratio_among_laughing']:.4f} & "
                f"{p_str} & {or_str} \\\\"
            )
        lines.append(r"\addlinespace")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_tex.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Table 1: ITW laughter prevalence by class (bona-fide vs spoof)."
    )
    parser.add_argument("--csv", type=Path, default=Path("detector_out/itw_laughter.csv"),
                        help="Path to itw_laughter.csv (Stage-1 output).")
    parser.add_argument("--out-csv", type=Path, default=Path("tables/table1.csv"),
                        help="Output CSV path.")
    parser.add_argument("--out-tex", type=Path, default=Path("tables/table1.tex"),
                        help="Output LaTeX path.")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: input csv not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    table1 = build_table1(df)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table1.to_csv(args.out_csv, index=False)

    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    write_latex(table1, args.out_tex)

    print(f"Wrote {args.out_csv} ({len(table1)} rows)")
    print(f"Wrote {args.out_tex}")
    print(table1.to_string(index=False))


if __name__ == "__main__":
    main()
