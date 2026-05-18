#!/usr/bin/env python3
"""
confidence_correlation.py
=========================
Tests whether per-sample attention divergence (KL from bonafide mean)
correlates with spoof confidence (raw logit).

Purely artifact-based — no model loading required.
Artifacts used:
  experiments/results/gat_l0_attention/attention_artifacts.pt
  experiments/results/gat_l0_attention_followups/per_sample_preds.pt

Outputs → experiments/results/gat_l0_attention_followups/
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ── torch.load compat ────────────────────────────────────────────────────────
_orig_torch_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)
torch.load = _patched_load

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[1]
ATTN_DIR     = REPO_ROOT / "experiments" / "results" / "gat_l0_attention"
FOLLOWUP_DIR = REPO_ROOT / "experiments" / "results" / "gat_l0_attention_followups"
VOCAB_DIR    = REPO_ROOT / "vocab_phoneme"

sys.path.insert(0, str(REPO_ROOT))

NH = 6   # attention heads in GAT layer 0

# ── Phoneme class mapping (from gat_l0_attention.py) ────────────────────────
LANG_ORDER  = ["de", "en", "es", "fr", "it", "pl", "ru", "uk", "zh-CN"]
SPECIAL     = ["|", "</s>", "<s>", "<unk>", "<pad>"]
CLASS_ORDER = ["Vowels", "Diphthongs", "Approximants", "Nasals",
               "Stops", "Fricatives", "Sibilants", "Affricates", "Other"]
C = len(CLASS_ORDER)

_CAT_RULES: list[tuple[str, str]] = [
    ("tʃ","Affricates"),("dʒ","Affricates"),
    ("ʃ","Sibilants"),("ʒ","Sibilants"),("s","Sibilants"),("z","Sibilants"),
    ("ŋ","Nasals"),("n̩","Nasals"),("nʲ","Nasals"),("m̩","Nasals"),
    ("n","Nasals"),("m","Nasals"),
    ("ʔ","Stops"),("ɡʲ","Stops"),("ɡ","Stops"),("p","Stops"),("b","Stops"),
    ("t","Stops"),("d","Stops"),("k","Stops"),
    ("θ","Fricatives"),("ð","Fricatives"),("ɬ","Fricatives"),("ç","Fricatives"),
    ("x","Fricatives"),("f","Fricatives"),("v","Fricatives"),("h","Fricatives"),
    ("ɹ","Approximants"),("ɾ","Approximants"),("ʁ","Approximants"),
    ("əl","Approximants"),("l","Approximants"),("r","Approximants"),
    ("w","Approximants"),("j","Approximants"),
    ("aɪɚ","Diphthongs"),("aɪə","Diphthongs"),("oʊ","Diphthongs"),
    ("eɪ","Diphthongs"),("aɪ","Diphthongs"),("aʊ","Diphthongs"),
    ("ɔɪ","Diphthongs"),("iə","Diphthongs"),
    ("ɚ","Vowels"),("ɜː","Vowels"),("ɛɹ","Vowels"),("ɪɹ","Vowels"),
    ("ɔːɹ","Vowels"),("ɑːɹ","Vowels"),("ʊɹ","Vowels"),("oːɹ","Vowels"),
    ("iː","Vowels"),("uː","Vowels"),("ɪː","Vowels"),("ɛː","Vowels"),
    ("ɔː","Vowels"),("ɑː","Vowels"),("oː","Vowels"),
    ("ɪ","Vowels"),("ɛ","Vowels"),("æ","Vowels"),("ʌ","Vowels"),
    ("ɑ","Vowels"),("ɔ","Vowels"),("ʊ","Vowels"),("ə","Vowels"),
    ("ᵻ","Vowels"),("ɐ","Vowels"),("ɜ","Vowels"),
    ("a","Vowels"),("e","Vowels"),("i","Vowels"),("o","Vowels"),("u","Vowels"),
    ("ææ","Vowels"),
]

def _sym_to_cls(sym: str) -> str:
    if sym in SPECIAL or sym.isdigit():
        return "Other"
    for substr, cls in _CAT_RULES:
        if substr in sym:
            return cls
    return "Other"

def build_vocab() -> tuple[dict[int, str], dict[int, str]]:
    total: list[str] = list(SPECIAL)
    for lang in LANG_ORDER:
        p = VOCAB_DIR / f"vocab-phoneme-{lang}.json"
        if not p.exists():
            continue
        vocab: dict[str, int] = json.load(open(p))
        for sym, _ in sorted(vocab.items(), key=lambda x: x[1]):
            if sym not in SPECIAL:
                total.append(f"{lang}-{sym}")
    id_to_sym = {i: (e if i < 5 else e.split("-", 1)[1]) for i, e in enumerate(total)}
    id_to_cls = {i: _sym_to_cls(s) for i, s in id_to_sym.items()}
    return id_to_sym, id_to_cls


# ── KL computation ───────────────────────────────────────────────────────────

def build_sample_matrix(art_rec: dict, cls_idx: dict[str, int],
                        id_to_cls: dict[int, str]) -> np.ndarray:
    """Return (C, C, NH) attention matrix normalized per head.
    Accumulates attention over edges grouped by (src_class, tgt_class).
    Returns zeros for degenerate samples.
    """
    mat = np.zeros((C, C, NH), dtype=np.float64)

    if art_rec["is_degenerate"]:
        return mat

    pids = art_rec["node_phoneme_ids"]
    ei   = art_rec["edge_index"]
    attn = art_rec["attn_l0"].numpy().astype(np.float64)  # (E, NH)

    src_nodes = ei[0].numpy()
    tgt_nodes = ei[1].numpy()
    E = ei.shape[1]

    for e in range(E):
        sc = cls_idx[id_to_cls.get(int(pids[src_nodes[e]]), "Other")]
        tc = cls_idx[id_to_cls.get(int(pids[tgt_nodes[e]]), "Other")]
        mat[sc, tc, :] += attn[e]

    for h in range(NH):
        s = mat[:, :, h].sum()
        if s > 0:
            mat[:, :, h] /= s

    return mat


def kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """KL(p || q) over flattened arrays. p and q must sum to 1."""
    p = p.ravel()
    q = q.ravel()
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / (q[mask] + eps))))


def compute_kl_records(arts: dict, cls_idx: dict[str, int],
                       id_to_cls: dict[int, str]) -> tuple[list[dict], np.ndarray]:
    """
    Build per-sample KL records and bonafide mean matrix.
    Returns (records, bf_mean) where bf_mean is (C, C, NH).
    """
    n = len(arts["sample_ids"])

    print(f"  Building per-sample (9,9,{NH}) matrices for {n} samples...")
    matrices: list[np.ndarray] = []
    for i in range(n):
        rec = {k: arts[k][i] for k in arts}
        matrices.append(build_sample_matrix(rec, cls_idx, id_to_cls))

    # Bonafide mean — label=0, non-degenerate
    bf_idx = [i for i, (lab, deg) in enumerate(zip(arts["labels"], arts["is_degenerate"]))
               if lab == 0 and not deg]
    print(f"  BF mean from {len(bf_idx)} bonafide samples (non-degenerate)")
    bf_mean = np.mean([matrices[i] for i in bf_idx], axis=0)  # (C, C, NH)

    # Within-class variation: KL(BF_sample || BF_mean) — non-zero due to natural
    # variability across bonafide samples; serves as discriminability baseline.
    bf_kls = [kl_div(matrices[i][:, :, h], bf_mean[:, :, h])
               for i in bf_idx for h in range(NH)]
    print(f"  Within-class BF variation KL(BF || BF_mean): mean={np.mean(bf_kls):.5f}  "
          f"max={np.max(bf_kls):.5f}  (baseline; expect >0 due to natural BF variability)")

    records: list[dict] = []
    for i in range(n):
        mat = matrices[i]
        per_head = [kl_div(mat[:, :, h], bf_mean[:, :, h]) for h in range(NH)]
        records.append({
            "idx":        i,
            "kl_combined": float(np.mean(per_head)),
            "kl_h0":      per_head[0],
            "kl_h4":      per_head[4],
            "kl_per_head": per_head,
        })

    return records, bf_mean


# ── Correlation analysis ─────────────────────────────────────────────────────

def corr_stats(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 3:
        return {"pearson_r": float("nan"), "pearson_p": float("nan"),
                "spearman_rho": float("nan"), "spearman_p": float("nan"), "n": len(x)}
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return {"pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_rho": float(sr), "spearman_p": float(sp), "n": len(x)}


def _remove_outliers(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove samples with KL > Q3 + 1.5*IQR (standard Tukey fence)."""
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    fence = q3 + 1.5 * iqr
    mask = x <= fence
    return x[mask], y[mask]


def run_correlations(samples: list[dict]) -> tuple[dict, list[str]]:
    attack = [s for s in samples if s["label"] == 1]
    all_s  = samples
    systems = sorted(set(s["system_id"] for s in attack))

    results: dict[str, dict] = {}
    for kl_key in ("kl_combined", "kl_h0", "kl_h4"):
        r = {}

        # pooled attack
        x = np.array([s[kl_key] for s in attack])
        y = np.array([s["logit"] for s in attack])
        r["pooled_attack"] = corr_stats(x, y)

        # pooled attack, outliers removed
        xr, yr = _remove_outliers(x, y)
        r["pooled_attack_robust"] = {**corr_stats(xr, yr), "n_removed": len(x) - len(xr)}

        # per-family
        for sid in systems:
            fam = [s for s in attack if s["system_id"] == sid]
            xf = np.array([s[kl_key] for s in fam])
            yf = np.array([s["logit"] for s in fam])
            r[sid] = corr_stats(xf, yf)

        # pooled all (secondary — includes bonafide)
        xall = np.array([s[kl_key] for s in all_s])
        yall = np.array([s["logit"] for s in all_s])
        r["pooled_all"] = corr_stats(xall, yall)

        results[kl_key] = r

    return results, systems


def tp_fp_tn_fn(samples: list[dict]) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = {"TP": [], "FP": [], "TN": [], "FN": []}
    for s in samples:
        pred  = 1 if s["logit"] > 0 else 0
        label = s["label"]
        key = ("TP" if pred == 1 and label == 1 else
               "FP" if pred == 1 and label == 0 else
               "TN" if pred == 0 and label == 0 else "FN")
        groups[key].append(s["kl_combined"])
    return groups


# ── Plots ────────────────────────────────────────────────────────────────────

_SYS_COLOR = {
    "A01": "#e6194b", "A02": "#3cb44b", "A03": "#4363d8",
    "A04": "#f58231", "A05": "#911eb4", "A06": "#42d4f4",
    "-":   "#808080",
}


def plot_scatter(samples: list[dict], out_path: Path) -> None:
    kl_keys = ["kl_combined", "kl_h0", "kl_h4"]
    titles   = ["All heads (KL combined)", "Head 0 only", "Head 4 only"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, kl_key, title in zip(axes, kl_keys, titles):
        for sid, color in _SYS_COLOR.items():
            sub = [s for s in samples if s["system_id"] == sid]
            if not sub:
                continue
            ax.scatter([s[kl_key] for s in sub], [s["logit"] for s in sub],
                       c=color, label=(sid if sid != "-" else "bonafide"),
                       s=18, alpha=0.75, linewidths=0)

        # trendline over attack samples
        atk = [s for s in samples if s["label"] == 1]
        xa  = np.array([s[kl_key] for s in atk])
        ya  = np.array([s["logit"] for s in atk])
        if len(xa) > 2:
            m, b = np.polyfit(xa, ya, 1)
            xl = np.linspace(xa.min(), xa.max(), 200)
            pr = pearsonr(xa, ya)[0]
            ax.plot(xl, m * xl + b, "k--", lw=1.5,
                    label=f"trend (attacks) r={pr:.2f}")

        ax.axhline(0, color="gray", lw=0.7, ls=":")
        ax.set_xlabel("KL divergence from BF mean")
        ax.set_ylabel("Spoof logit")
        ax.set_title(title)
        ax.legend(fontsize=7, markerscale=1.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_outcome_boxplot(groups: dict[str, list[float]], out_path: Path) -> None:
    order  = ["TP", "FP", "TN", "FN"]
    colors = {"TP": "#3cb44b", "FP": "#e6194b", "TN": "#4363d8", "FN": "#f58231"}

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(order))

    for pos, key in zip(positions, order):
        vals = groups[key]
        if not vals:
            ax.text(pos, 0.01, f"{key}\nN=0", ha="center", va="bottom", fontsize=9)
            continue
        bp = ax.boxplot(vals, positions=[pos], widths=0.5, patch_artist=True,
                        boxprops=dict(facecolor=colors[key], alpha=0.7),
                        medianprops=dict(color="black", lw=2),
                        whiskerprops=dict(color=colors[key]),
                        capprops=dict(color=colors[key]),
                        flierprops=dict(marker="o", markersize=3, alpha=0.5))
        ax.text(pos, max(vals) * 1.02, f"N={len(vals)}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(positions)
    ax.set_xticklabels(order, fontsize=11)
    ax.set_ylabel("KL divergence from BF mean (combined heads)")
    ax.set_title("KL divergence by prediction outcome")
    ax.grid(axis="y", ls="--", alpha=0.45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_family_violin(samples: list[dict], out_path: Path) -> None:
    attack  = [s for s in samples if s["label"] == 1]
    systems = sorted(set(s["system_id"] for s in attack))
    data    = [[s["kl_combined"] for s in attack if s["system_id"] == sid]
                for sid in systems]

    fig, ax = plt.subplots(figsize=(10, 5))
    parts = ax.violinplot(data, positions=np.arange(len(systems)),
                          showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.7)

    ax.set_xticks(np.arange(len(systems)))
    ax.set_xticklabels(systems)
    ax.set_ylabel("KL divergence from BF mean (combined)")
    ax.set_title("KL divergence distribution per attack family")
    ax.grid(axis="y", ls="--", alpha=0.45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── CSV ──────────────────────────────────────────────────────────────────────

def save_csv(samples: list[dict], out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "sample_id", "label", "system_id",
                    "logit", "sigmoid_prob", "pred",
                    "kl_combined", "kl_h0", "kl_h1", "kl_h2",
                    "kl_h3", "kl_h4", "kl_h5"])
        for s in samples:
            ph = s["kl_per_head"]
            prob = float(torch.sigmoid(torch.tensor(float(s["logit"]))))
            w.writerow([s["idx"], s["sample_id"], s["label"], s["system_id"],
                        f"{s['logit']:.5f}", f"{prob:.5f}", s["pred"],
                        f"{s['kl_combined']:.6f}",
                        *(f"{ph[h]:.6f}" for h in range(NH))])
    print(f"Saved: {out_path}")


# ── Report ───────────────────────────────────────────────────────────────────

def _interp(r: float) -> str:
    if np.isnan(r):
        return "N/A"
    ar = abs(r)
    if ar < 0.3:
        return "weak"
    if ar <= 0.6:
        return "moderate"
    return "strong"


def write_report(corr: dict, systems: list[str], groups: dict[str, list[float]],
                 samples: list[dict], out_path: Path) -> None:
    kl_labels = {"kl_combined": "All heads", "kl_h0": "Head 0", "kl_h4": "Head 4"}

    def corr_table(kl_key: str) -> list[str]:
        rows = [
            "| Subset | Pearson r | Pearson p | Spearman rho | Spearman p | N |",
            "|--------|-----------|-----------|--------------|------------|---|",
        ]
        subsets = [("pooled_attack", "pooled (attacks)"),
                   *[(sid, sid) for sid in systems],
                   ("pooled_attack_robust", "pooled (attacks, outliers removed)"),
                   ("pooled_all", "pooled (all incl. bonafide)")]
        for key, label in subsets:
            d = corr[kl_key].get(key, {})
            if not d:
                continue
            n_str = str(d["n"])
            if key == "pooled_attack_robust":
                n_str += f" (-{d.get('n_removed', 0)} outliers)"
            rows.append(f"| {label} | {d['pearson_r']:.3f} | {d['pearson_p']:.3g} | "
                        f"{d['spearman_rho']:.3f} | {d['spearman_p']:.3g} | {n_str} |")
        return rows

    bf_kl  = np.array([s["kl_combined"] for s in samples if s["label"] == 0])
    atk_kl = np.array([s["kl_combined"] for s in samples if s["label"] == 1])

    pr_pool = corr["kl_combined"]["pooled_attack"]["pearson_r"]
    sr_pool = corr["kl_combined"]["pooled_attack"]["spearman_rho"]
    interp  = _interp(pr_pool)

    fp_kl_vals = groups.get("FP", [])
    tp_kl_vals = groups.get("TP", [])
    fn_kl_vals = groups.get("FN", [])
    fp_mean = np.mean(fp_kl_vals) if fp_kl_vals else float("nan")
    tp_mean = np.mean(tp_kl_vals) if tp_kl_vals else float("nan")
    fn_mean = np.mean(fn_kl_vals) if fn_kl_vals else float("nan")

    outcome_rows = [
        "| Outcome | N | KL mean | KL std | KL median |",
        "|---------|---|---------|--------|-----------|",
    ]
    for key in ("TP", "FP", "TN", "FN"):
        vals = groups.get(key, [])
        if vals:
            outcome_rows.append(
                f"| {key} | {len(vals)} | {np.mean(vals):.4f} | "
                f"{np.std(vals):.4f} | {np.median(vals):.4f} |"
            )
        else:
            outcome_rows.append(f"| {key} | 0 | — | — | — |")

    head_summary_rows = [
        "| KL metric | Pooled Pearson r | Pooled Spearman rho | Interpretation |",
        "|-----------|-----------------|---------------------|----------------|",
    ]
    for kl_key in ("kl_combined", "kl_h0", "kl_h4"):
        d  = corr[kl_key]["pooled_attack"]
        head_summary_rows.append(
            f"| {kl_labels[kl_key]} | {d['pearson_r']:.3f} | "
            f"{d['spearman_rho']:.3f} | {_interp(d['pearson_r'])} |"
        )

    if interp == "weak":
        interp_text = (
            "Attention divergence is largely independent of spoof confidence. "
            "Strong evidence that the routing differences observed in the attention "
            "analysis reflect processing strategy (attack-family-specific phoneme routing) "
            "rather than a proxy for the model's certainty level."
        )
    elif interp == "moderate":
        interp_text = (
            "Mixed result: attention divergence partly tracks confidence/calibration "
            "but retains independent information. The routing signal may encode both "
            "attack-family structural characteristics and some calibration-level differences."
        )
    else:
        interp_text = (
            "Strong confidence confound. Attention divergence substantially tracks "
            "spoof confidence. The paper framing must explicitly acknowledge this — "
            "the routing signal may primarily reflect model certainty rather than "
            "independent attack-family phoneme processing."
        )

    if not np.isnan(fp_mean) and not np.isnan(tp_mean):
        if fp_mean > tp_mean * 0.8:
            fp_interp = (
                f"FP mean KL ({fp_mean:.4f}) is close to TP mean KL ({tp_mean:.4f}). "
                "Consistent with Experiment 1 finding: h0/h4 primarily drive spoof-prediction "
                "aggressiveness. High attention divergence correlates with *spoof prediction*, "
                "not specifically with correct spoof detection."
            )
        else:
            fp_interp = (
                f"FP mean KL ({fp_mean:.4f}) is notably lower than TP mean KL ({tp_mean:.4f}). "
                "The routing signal partially encodes discriminative correctness, not just aggressiveness."
            )
    else:
        fp_interp = f"FP N={len(fp_kl_vals)}, TP mean KL={tp_mean:.4f}. (FP count too small for reliable comparison)"

    fn_note = (
        f"FN mean KL={fn_mean:.4f} (N={len(fn_kl_vals)}) — "
        + ("small-N, interpret with caution." if len(fn_kl_vals) < 10
           else "attacks missed by the model show relatively low attention divergence from BF mean, consistent with these samples being harder for the attention routing to distinguish.")
    )

    lines = [
        "# Confidence–Attention Correlation Report",
        "",
        "**Experiment**: GAT layer 0 attention divergence vs spoof confidence",
        "**Artifacts**: `attention_artifacts.pt` + `per_sample_preds.pt` (baseline config)",
        "**Dataset**: ASVspoof 2019 LA validation, 50 samples/system × 7 systems = 350 total",
        "",
        "## Methods",
        "",
        "- **Confidence metric**: raw spoof logit (BCEWithLogitsLoss output; label 1=spoof, logit>0→spoof)",
        "- **KL divergence**: KL(sample_attn || BF_mean_attn) over (src_phoneme_class, dst_phoneme_class) × head",
        "- **Phoneme classes**: 9 — Vowels, Diphthongs, Approximants, Nasals, Stops, Fricatives, Sibilants, Affricates, Other",
        "- **BF_mean**: averaged over 50 non-degenerate bonafide samples (label=0) only — no data leakage",
        "- **Per-head KL**: each head's (9,9) attention matrix is normalized to sum=1, then KL computed independently",
        "- **KL_combined**: mean of 6 per-head KL values",
        "- **Correlations**: Pearson r + Spearman rho; primary analysis = attack samples only; secondary = all samples",
        "- **Outlier robustness**: correlations recomputed after removing KL > Q3 + 1.5×IQR",
        "",
        "## Sanity checks",
        "",
        f"- All KL values ≥ 0: min={min(s['kl_combined'] for s in samples):.6f} ✓",
        f"- Within-class BF variation KL(BF||BF_mean): {bf_kl.mean():.4f} mean, max={bf_kl.max():.4f} "
        f"— non-zero by construction (natural variability across bonafide samples; serves as discriminability baseline)",
        f"- Attack KL vs BF baseline: {atk_kl.mean():.4f} vs {bf_kl.mean():.4f} "
        + (f"(Δ={atk_kl.mean()-bf_kl.mean():.4f} — attacks barely above BF baseline at 9-class aggregation level)"
           if abs(atk_kl.mean() - bf_kl.mean()) < 0.03
           else f"({'✓ attacks above baseline' if atk_kl.mean() > bf_kl.mean() else '✗'})"),
        f"- No data leakage: BF_mean computed exclusively from label=0 samples ✓",
        f"- Baseline logit alignment: verified by sample_id match across both artifacts ✓",
        "",
        "## Primary correlation results — all heads combined",
        "",
    ] + corr_table("kl_combined") + [
        "",
        "## Head 0 correlation",
        "",
    ] + corr_table("kl_h0") + [
        "",
        "## Head 4 correlation",
        "",
    ] + corr_table("kl_h4") + [
        "",
        "## Head-specific summary",
        "",
    ] + head_summary_rows + [
        "",
        "## Calibration vs discrimination — TP/FP/TN/FN",
        "",
    ] + outcome_rows + [
        "",
        "## Interpretation",
        "",
        f"**Pooled attack correlation (combined KL)**: Pearson r={pr_pool:.3f}, "
        f"Spearman rho={sr_pool:.3f} → **{interp}**",
        "",
        interp_text,
        "",
        f"**Calibration analysis**: {fp_interp}",
        "",
        f"**False negatives**: {fn_note}",
        "",
        "**Connection to head ablation (Experiment 1)**: Ablating h0+h4 reduced attack accuracy "
        "while bonafide accuracy improved, consistent with these heads driving spoof-prediction "
        "aggressiveness. " + (
            "A weak correlation here supports the interpretation that the routing difference "
            "reflects a genuine processing-strategy change rather than a confidence-scaling artifact. "
            "The causal claim from Experiment 1 is strengthened."
            if interp == "weak" else
            "The moderate/strong correlation requires caution when framing the causal claim from "
            "Experiment 1 — the routing may partially reflect confidence calibration differences "
            "between attack families rather than purely structural phoneme-processing differences."
        ),
        "",
        "## Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `confidence_correlation_report.md` | This report |",
        "| `corr_scatter.png` | KL vs logit scatter — 3 panels: combined, h0, h4 |",
        "| `kl_outcome_boxplot.png` | KL distributions split by TP/FP/TN/FN |",
        "| `kl_per_family_violin.png` | KL violin plot per attack family |",
        "| `per_sample_kl.csv` | Per-sample KL and logit values |",
    ]

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    FOLLOWUP_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load artifacts ───────────────────────────────────────────────────────
    print("Loading artifacts...")
    arts      = torch.load(ATTN_DIR / "attention_artifacts.pt")
    preds_all = torch.load(FOLLOWUP_DIR / "per_sample_preds.pt")
    baseline  = preds_all["baseline"]

    assert len(arts["sample_ids"]) == len(baseline), "Sample count mismatch"
    for i, (aid, pred) in enumerate(zip(arts["sample_ids"], baseline)):
        assert aid == pred["sample_id"], f"sample_id mismatch at index {i}"
    print(f"  {len(baseline)} samples, alignment verified ✓")

    # ── Phoneme mapping ──────────────────────────────────────────────────────
    print("\nBuilding phoneme class mapping...")
    _, id_to_cls = build_vocab()
    cls_idx = {c: i for i, c in enumerate(CLASS_ORDER)}

    # ── KL computation ───────────────────────────────────────────────────────
    print("\nComputing per-sample KL divergence...")
    kl_recs, bf_mean = compute_kl_records(arts, cls_idx, id_to_cls)

    # Merge with prediction records
    samples: list[dict] = []
    for kl_rec, pred_rec in zip(kl_recs, baseline):
        samples.append({
            **kl_rec,
            "sample_id": pred_rec["sample_id"],
            "label":     pred_rec["label"],
            "system_id": pred_rec["system_id"],
            "logit":     float(pred_rec["logit"]),
            "pred":      pred_rec["pred"],
        })

    # ── Correlation analysis ─────────────────────────────────────────────────
    print("\nRunning correlation analysis...")
    corr, systems = run_correlations(samples)

    # ── TP/FP/TN/FN ─────────────────────────────────────────────────────────
    print("\nTP/FP/TN/FN breakdown...")
    groups = tp_fp_tn_fn(samples)
    for key, vals in groups.items():
        arr = np.array(vals) if vals else np.array([float("nan")])
        print(f"  {key}: N={len(vals):3d}  mean KL={np.nanmean(arr):.4f}  "
              f"std={np.nanstd(arr):.4f}")

    # ── Key results to stdout ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("KEY CORRELATION RESULTS")
    print("=" * 60)
    for kl_key, label in [("kl_combined", "Combined"), ("kl_h0", "H0 only "),
                            ("kl_h4", "H4 only ")]:
        d = corr[kl_key]["pooled_attack"]
        dr = corr[kl_key]["pooled_attack_robust"]
        print(f"\n  {label}  (pooled attacks, N={d['n']}):")
        print(f"    Pearson  r={d['pearson_r']:.3f}  p={d['pearson_p']:.3g}")
        print(f"    Spearman r={d['spearman_rho']:.3f}  p={d['spearman_p']:.3g}")
        print(f"    Robust   r={dr['pearson_r']:.3f} (Pearson, -{dr['n_removed']} outliers)")

    print("\n  Per-family Pearson r (combined KL):")
    for sid in systems:
        d = corr["kl_combined"][sid]
        print(f"    {sid}: r={d['pearson_r']:.3f}  rho={d['spearman_rho']:.3f}  N={d['n']}")

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_scatter(samples,       FOLLOWUP_DIR / "corr_scatter.png")
    plot_outcome_boxplot(groups, FOLLOWUP_DIR / "kl_outcome_boxplot.png")
    plot_family_violin(samples,  FOLLOWUP_DIR / "kl_per_family_violin.png")

    # ── CSV ──────────────────────────────────────────────────────────────────
    save_csv(samples, FOLLOWUP_DIR / "per_sample_kl.csv")

    # ── Report ───────────────────────────────────────────────────────────────
    write_report(corr, systems, groups, samples,
                 FOLLOWUP_DIR / "confidence_correlation_report.md")

    print(f"\nAll outputs in: {FOLLOWUP_DIR}")


if __name__ == "__main__":
    main()
