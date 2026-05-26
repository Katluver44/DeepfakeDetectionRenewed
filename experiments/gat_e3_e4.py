#!/usr/bin/env python3
"""
gat_e3_e4.py
============
E3: Head-Decomposed Hub-Mass Analysis
    Build A_agg_critical (h0,h4 only per layer) and A_agg_control (h1,2,3,5 per layer).
    Mirror E2 hub-mass permutation for each subset.
    Outcome: critical-vs-control family p-value table per attack system.

E4: Per-System EER Under Attention Ablation (All 3 Layers)
    3 conditions:
      baseline     — no ablation
      attn_ablated — zero ALL 6 heads at ALL 3 layers (skip path only)
      critical_only — zero h0,h4 at ALL 3 layers
    Per-system EER with bootstrap CIs (1000 resamples).

Run order:
  venv/bin/python3 experiments/gat_e3_e4.py          # E3 only
  venv/bin/python3 experiments/gat_e3_e4.py --run-e4 # E3 + E4

Outputs → experiments/results/gat_e3_e4/
"""
from __future__ import annotations

import csv
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# ── torch.load compat (before any torch.load calls) ──────────────────────────
_orig_torch_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)
torch.load = _patched_load

from argparse import Namespace
try:
    from pandas import Series as _PS
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination as _PTR
    torch.serialization.add_safe_globals([Namespace, _PS, _PTR])
except Exception:
    torch.serialization.add_safe_globals([Namespace])

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[1]
EXP_DIR    = Path(__file__).resolve().parent
OUT_BASE   = EXP_DIR / "results" / "gat_e3_e4"
OUT_E3     = OUT_BASE / "e3"
OUT_E4     = OUT_BASE / "e4"
ALL_LAYERS = EXP_DIR / "results" / "gat_attn_graphs" / "all_layers_artifacts.pt"
VOCAB_DIR  = REPO_ROOT / "vocab_phoneme"
CKPT       = REPO_ROOT / "models" / "robust_goat.ckpt"

for p in [OUT_BASE, OUT_E3, OUT_E4]:
    p.mkdir(parents=True, exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Constants ─────────────────────────────────────────────────────────────────
CRITICAL_HEADS = [0, 4]
CONTROL_HEADS  = [1, 2, 3, 5]
N_PERM         = 10_000
SEED           = 42
NF_PER_SAMPLE  = 3 * 16_000 // 320 - 1   # 149
N_BOOTSTRAP    = 1_000
MIN_NODES      = 3

LANG_ORDER = ["de", "en", "es", "fr", "it", "pl", "ru", "uk", "zh-CN"]
SPECIAL    = ["|", "</s>", "<s>", "<unk>", "<pad>"]


# ── Vocab (same logic as gat_attn_graphs.py) ──────────────────────────────────

import json

def build_vocab() -> tuple[dict[int, str], dict[int, str]]:
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
        if sym in SPECIAL or sym.isdigit(): return "Other"
        for substr, cls in _CAT_RULES:
            if substr in sym: return cls
        return "Other"

    total: list[str] = list(SPECIAL)
    for lang in LANG_ORDER:
        p = VOCAB_DIR / f"vocab-phoneme-{lang}.json"
        if not p.exists(): continue
        vocab: dict[str, int] = json.load(open(p))
        for sym, _ in sorted(vocab.items(), key=lambda x: x[1]):
            if sym not in SPECIAL:
                total.append(f"{lang}-{sym}")
    id_to_sym = {i: (e if i < 5 else e.split("-", 1)[1]) for i, e in enumerate(total)}
    id_to_cls = {i: _sym_to_cls(s) for i, s in id_to_sym.items()}
    return id_to_sym, id_to_cls


# ── Holm-Bonferroni ───────────────────────────────────────────────────────────

def holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    n     = len(pvals)
    order = np.argsort(pvals)
    adj   = np.minimum(1.0, pvals[order] * np.arange(n, 0, -1, dtype=float))
    for i in range(1, n):
        adj[i] = max(adj[i], adj[i - 1])
    result        = np.empty(n)
    result[order] = adj
    return np.clip(result, 0.0, 1.0)


# ── Load all-layer artifacts ──────────────────────────────────────────────────

def load_all_layers(path: Path) -> list[dict]:
    data = torch.load(str(path), map_location="cpu", weights_only=False)
    n    = len(data["sample_ids"])
    recs = []
    for i in range(n):
        recs.append({
            "sample_id":        data["sample_ids"][i],
            "label":            data["labels"][i],
            "system_id":        data["system_ids"][i],
            "n_nodes":          data["n_nodes"][i],
            "n_edges":          data["n_edges"][i],
            "is_degenerate":    data["is_degenerate"][i],
            "node_phoneme_ids": data["node_phoneme_ids"][i],
            "edge_index":       data["edge_index"][i],
            "attn_l0":          data["attn_l0"][i],
            "attn_l1":          data["attn_l1"][i],
            "attn_l2":          data["attn_l2"][i],
        })
    return recs


# ── A_agg with head_subset ────────────────────────────────────────────────────

def sparse_to_dense(attn_mean: np.ndarray, edge_index: np.ndarray, N: int) -> np.ndarray:
    A   = np.zeros((N, N), dtype=np.float64)
    src = edge_index[0]
    tgt = edge_index[1]
    np.add.at(A, (tgt, src), attn_mean)
    row_sums = A.sum(axis=1)
    no_in    = np.where(row_sums < 1e-10)[0]
    A[no_in, no_in] = 1.0
    return A


def build_agg_graph(rec: dict, head_subset: list[int] | None = None) -> np.ndarray:
    """
    A_agg = A^{L3} @ A^{L2} @ A^{L1}, row-normalised.
    head_subset: None=all 6, [0,4]=critical, [1,2,3,5]=control.
    Mean is taken over the selected heads before building each layer's dense matrix.
    """
    N  = rec["n_nodes"]
    ei = rec["edge_index"].numpy()   # (2, E)
    layers: list[np.ndarray] = []
    for li in range(3):
        attn_np = rec[f"attn_l{li}"].float().numpy()   # (E, NH)
        if head_subset is not None:
            attn_np = attn_np[:, head_subset]
        attn_mean = attn_np.mean(axis=1)               # (E,)
        layers.append(sparse_to_dense(attn_mean, ei, N))

    A1, A2, A3 = layers
    A_agg = A3 @ A2 @ A1
    rs = A_agg.sum(axis=1, keepdims=True)
    return A_agg / np.where(rs > 1e-10, rs, 1.0)


# ── Per-sample pair matrix → column-sum ───────────────────────────────────────

def build_symbol_vocab(records: list[dict],
                       id_to_sym: dict[int, str]) -> tuple[list[str], dict[str, int]]:
    seen: set[str] = set()
    for rec in records:
        if rec["is_degenerate"]: continue
        for pid in rec["node_phoneme_ids"].tolist():
            seen.add(id_to_sym.get(int(pid), "?"))
    sym_list   = sorted(seen)
    sym_to_idx = {s: i for i, s in enumerate(sym_list)}
    return sym_list, sym_to_idx


def per_sample_colsum(rec: dict, A_agg: np.ndarray,
                      id_to_sym: dict[int, str],
                      sym_to_idx: dict[str, int], V: int) -> np.ndarray:
    """
    Column-sum d_in(p) = sum_t A_agg[t,p] / sum over all (t,p) normalisation.
    Returns (V,) vector of per-phoneme in-degree for this sample's A_agg.
    Mirrors _precompute_sample_colsums in gat_attn_graphs.py but computes
    directly from A_agg rather than via the VxV pair matrix detour.
    """
    pids    = rec["node_phoneme_ids"].numpy()
    N       = rec["n_nodes"]
    sym_ids = np.array([sym_to_idx.get(id_to_sym.get(int(p), "?"), 0)
                        for p in pids], dtype=np.int64)

    H = np.zeros((N, V), dtype=np.float64)
    H[np.arange(N), sym_ids] = 1.0

    # pair_sum[t, s] = sum of A_agg[i,j] over (i,j) with sym[i]=t, sym[j]=s
    pair_sum = H.T @ A_agg @ H                      # (V, V)
    counts   = H.sum(axis=0)                         # (V,)
    outer    = counts[:, None] * counts[None, :]     # (V, V)

    with np.errstate(invalid="ignore", divide="ignore"):
        pair_mean = np.where(outer > 0, pair_sum / outer, np.nan)

    colsum = np.nansum(pair_mean, axis=0)    # (V,)  — column sum = d_in per phoneme
    return colsum


# ── E3 permutation test (mirrors E2 exactly) ─────────────────────────────────

def _e2_test_stat(atk_cs: np.ndarray, bon_cs: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanmean(atk_cs, axis=0) - np.nanmean(bon_cs, axis=0)


def run_hub_mass_perm(sample_colsums: dict[str, np.ndarray],
                      attack_systems: list[str], V: int,
                      sym_list: list[str],
                      n_perm: int = N_PERM,
                      label: str = "") -> dict[str, dict]:
    """
    Mirrors run_e2_permutation from gat_attn_graphs.py.
    sample_colsums: {system_id: (n_samples, V)}
    Returns {sid: {p_raw, p_holm, p_family, n_sig, obs}}
    """
    bon_cs  = sample_colsums["-"]
    results = {}

    for sid in sorted(attack_systems):
        atk_cs  = sample_colsums[sid]
        pool_cs = np.concatenate([atk_cs, bon_cs], axis=0)
        n_atk   = len(atk_cs)

        obs    = _e2_test_stat(atk_cs, bon_cs)   # (V,)
        exceed = np.zeros(V, dtype=np.int64)
        t0     = time.time()
        for _ in range(n_perm):
            perm  = np.random.permutation(len(pool_cs))
            pstat = _e2_test_stat(pool_cs[perm[:n_atk]], pool_cs[perm[n_atk:]])
            exceed += (np.abs(pstat) >= np.abs(obs))

        p_raw  = (exceed + 1) / (n_perm + 1)
        p_holm = holm_bonferroni(p_raw)
        n_sig  = int((p_holm < 0.05).sum())

        # Family-level: fraction of perms where max |perm_stat| >= max |obs|
        max_obs     = np.abs(obs).max()
        max_exceed  = 0
        for _ in range(n_perm):
            perm  = np.random.permutation(len(pool_cs))
            pstat = _e2_test_stat(pool_cs[perm[:n_atk]], pool_cs[perm[n_atk:]])
            if np.abs(pstat).max() >= max_obs:
                max_exceed += 1
        p_family = (max_exceed + 1) / (n_perm + 1)

        tag = f"[{label}] " if label else ""
        print(f"  {tag}{sid} ({n_perm} perms) "
              f"p_family={p_family:.5f}  n_sig={n_sig}/{V}  "
              f"({time.time()-t0:.1f}s)")
        results[sid] = {"obs": obs, "p_raw": p_raw, "p_holm": p_holm,
                        "p_family": p_family, "n_sig": n_sig}

    return results


# ── E3 main logic ─────────────────────────────────────────────────────────────

def run_e3(records: list[dict], id_to_sym: dict[int, str]) -> None:
    """Build critical/control A_agg sets, run hub-mass permutation, print table."""
    valid  = [r for r in records if not r["is_degenerate"]]
    sym_list, sym_to_idx = build_symbol_vocab(records, id_to_sym)
    V      = len(sym_list)
    systems = sorted(set(r["system_id"] for r in records))
    attack_systems = [s for s in systems if s != "-"]

    print(f"\n  Symbol vocab: V={V}  |  attack systems: {attack_systems}")

    for subset_name, head_subset in [("critical", CRITICAL_HEADS),
                                      ("control",  CONTROL_HEADS)]:
        print(f"\n--- E3 [{subset_name}] heads={head_subset}: building A_agg ---")
        t0 = time.time()
        sample_colsums: dict[str, list[np.ndarray]] = defaultdict(list)

        for i, rec in enumerate(valid):
            A_agg  = build_agg_graph(rec, head_subset=head_subset)
            colsum = per_sample_colsum(rec, A_agg, id_to_sym, sym_to_idx, V)
            sample_colsums[rec["system_id"]].append(colsum)

        stacked = {s: np.stack(arrs) for s, arrs in sample_colsums.items()}
        print(f"  Built {len(valid)} A_agg matrices in {time.time()-t0:.1f}s")

        print(f"\n--- E3 [{subset_name}]: hub-mass permutation ({N_PERM} perms per system) ---")
        perm_results = run_hub_mass_perm(stacked, attack_systems, V, sym_list,
                                         n_perm=N_PERM, label=subset_name)

        # Save CSV
        rows = []
        for sid in sorted(attack_systems):
            r     = perm_results[sid]
            sig   = [sym_list[i] for i in range(V) if r["p_holm"][i] < 0.05]
            rows.append({"system": sid, "subset": subset_name,
                         "p_family": r["p_family"], "n_sig": r["n_sig"],
                         "surviving_phonemes": "|".join(sig) if sig else "none"})
        csv_path = OUT_E3 / f"e3_{subset_name}_hub_mass.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"  Saved: {csv_path}")

        # Store for comparison table
        if subset_name == "critical":
            crit_results = perm_results
        else:
            ctrl_results = perm_results

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("E3: Critical-vs-Control family p-value table")
    print("=" * 70)
    header = f"{'system':8s}  {'p_family (critical h0,h4)':>25s}  {'p_family (control h1-3,5)':>25s}  {'verdict':>20s}"
    print(header)
    print("-" * len(header))

    table_rows = []
    for sid in sorted(attack_systems):
        p_crit = crit_results[sid]["p_family"]
        p_ctrl = ctrl_results[sid]["p_family"]
        sig_crit = p_crit < 0.05
        sig_ctrl = p_ctrl < 0.05
        if sig_crit and not sig_ctrl:
            verdict = "critical only"
        elif sig_crit and sig_ctrl:
            verdict = "both"
        elif not sig_crit and sig_ctrl:
            verdict = "control only (!)"
        else:
            verdict = "null"
        star_c = "*" if sig_crit else " "
        star_t = "*" if sig_ctrl else " "
        print(f"  {sid:6s}  {p_crit:>22.5f}{star_c}  {p_ctrl:>22.5f}{star_t}  {verdict:>20s}")
        table_rows.append({"system": sid,
                           "p_family_critical": p_crit,
                           "p_family_control": p_ctrl,
                           "sig_critical": sig_crit,
                           "sig_control": sig_ctrl,
                           "verdict": verdict})

    print("=" * 70)
    print("  * p < 0.05")

    # Write markdown report
    lines = ["# E3: Head-Decomposed Hub-Mass — Critical vs Control\n",
             "## Family p-value table\n",
             "| system | p_family (critical h0,h4) | p_family (control h1-3,5) | verdict |",
             "|--------|--------------------------|--------------------------|---------|"]
    for row in table_rows:
        sc = "**{:.5f}**".format(row["p_family_critical"]) if row["sig_critical"] \
             else "{:.5f}".format(row["p_family_critical"])
        st = "**{:.5f}**".format(row["p_family_control"]) if row["sig_control"] \
             else "{:.5f}".format(row["p_family_control"])
        lines.append(f"| {row['system']} | {sc} | {st} | {row['verdict']} |")
    lines += ["",
              "## Interpretation\n",
              "critical heads = h0, h4 (identified as high-KL heads in prior gat_l0_attention analysis)",
              "control heads  = h1, h2, h3, h5\n",
              "Outcomes:",
              "- **critical only**: signal mediated by h0/h4 across all 3 layers",
              "- **both**: graph-wide redistribution not specific to critical heads",
              "- **control only**: unexpected — critical heads not carrying the hub signal",
              "- **null**: system does not produce detectable hub-mass shift in either subset",
              ""]
    (OUT_E3 / "e3_report.md").write_text("\n".join(lines))
    print(f"\n  Saved: {OUT_E3 / 'e3_report.md'}")


# ── E4: Ablation across all 3 GAT layers ─────────────────────────────────────

def install_ablation_all_layers(gat_net, heads_to_zero: list[int]) -> callable:
    """
    Extends head_ablation.install_ablation to all 3 gat_net layers simultaneously.
    Mode is always 'zero' (post-softmax, attn[:,h,0]=0.0).
    Returns a remove_all() callable that restores all 3 layers.
    """
    removers = []
    for layer in gat_net:
        orig_nas = layer.neighborhood_aware_softmax  # bound method

        def _make_ablated(orig, heads):
            def _ablated_nas(scores_per_edge, trg_index, num_of_nodes):
                attn = orig(scores_per_edge, trg_index, num_of_nodes)  # (E, NH, 1)
                if not heads:
                    return attn
                attn = attn.clone()
                for h in heads:
                    attn[:, h, 0] = 0.0
                return attn
            return _ablated_nas

        ablated = _make_ablated(orig_nas, heads_to_zero)
        layer.neighborhood_aware_softmax = ablated

        def _make_remover(lay):
            def remove():
                if "neighborhood_aware_softmax" in lay.__dict__:
                    del lay.__dict__["neighborhood_aware_softmax"]
            return remove

        removers.append(_make_remover(layer))

    def remove_all():
        for r in removers: r()

    return remove_all


def run_eval_condition(lit, loader, device,
                       heads_to_zero: list[int]) -> list[dict]:
    """
    Run full dataset inference under a given zero-ablation condition.
    Returns per-sample records: {sample_id, label, system_id, logit}.
    """
    gat_model = lit.model
    gat_net   = gat_model.GAT.gat_net

    remove_all = install_ablation_all_layers(gat_net, heads_to_zero)
    records: list[dict] = []
    sample_id = 0

    try:
        with torch.no_grad():
            for batch in loader:
                audio   = batch["audio"].to(device)
                labels  = batch["label"].tolist()
                sys_ids = batch["system_id"]
                B       = len(labels)
                num_f   = torch.full((B,), NF_PER_SAMPLE, device=device)

                hs, pids = _run_frozen_frontend(audio, gat_model, device)
                result   = gat_model.encoder_and_GAT(hs, num_f, pids)
                logits   = result[5].cpu()   # (B,)

                for i in range(B):
                    records.append({
                        "sample_id": sample_id + i,
                        "label":     labels[i],
                        "system_id": sys_ids[i],
                        "logit":     float(logits[i].item()),
                    })
                sample_id += B
    finally:
        remove_all()
        # Verify all layers restored
        for idx, layer in enumerate(gat_net):
            if "neighborhood_aware_softmax" in layer.__dict__:
                raise AssertionError(f"Layer {idx} not restored after ablation!")

    return records


def _run_frozen_frontend(audio, gat_model, device):
    """Frozen WavLM feature extraction + phoneme prediction."""
    import torch.nn.functional as F
    x = audio
    if x.ndim == 3 and x.size(1) == 1:
        x = x[:, 0, :]
    with torch.no_grad():
        feat1         = gat_model.transformer_in_phoneme_model.feature_extractor(x).transpose(1, 2)
        hidden_states, _ = gat_model.transformer_in_phoneme_model.feature_projection(feat1)
        phoneme_feat  = gat_model.transformer_in_phoneme_model.encoder(hidden_states)[0]
        phoneme_logits = gat_model.phoneme_model.model.model.lm_head(phoneme_feat)
        phoneme_ids   = torch.argmax(phoneme_logits, dim=-1)
    return hidden_states, phoneme_ids


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """EER via linear threshold sweep.  labels: 1=spoof, scores=sigmoid(logit)."""
    thresholds = np.unique(scores)
    n_bon      = (labels == 0).sum()
    n_spoof    = (labels == 1).sum()
    best_eer, best_diff = 1.0, float("inf")
    for t in thresholds:
        preds = (scores >= t).astype(int)
        fp    = int(((preds == 1) & (labels == 0)).sum())
        fn    = int(((preds == 0) & (labels == 1)).sum())
        far   = fp / max(n_bon, 1)
        frr   = fn / max(n_spoof, 1)
        diff  = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_eer  = (far + frr) / 2
    return best_eer


def per_system_eer(records: list[dict],
                   attack_systems: list[str]) -> dict[str, float]:
    """
    For each attack system, compute EER using:
      - all bonafide samples as negatives
      - that system's samples as positives
    """
    bon_recs = [r for r in records if r["system_id"] == "-"]
    result   = {}
    for sid in attack_systems:
        atk_recs = [r for r in records if r["system_id"] == sid]
        combined = bon_recs + atk_recs
        labels   = np.array([r["label"] for r in combined])
        logits   = np.array([r["logit"] for r in combined])
        scores   = 1.0 / (1.0 + np.exp(-logits))
        result[sid] = compute_eer(labels, scores)
    return result


def bootstrap_eer_ci(records: list[dict], attack_systems: list[str],
                     n_bootstrap: int = N_BOOTSTRAP,
                     rng: np.random.Generator | None = None) -> dict[str, tuple[float, float]]:
    """
    Bootstrap 95% CIs for per-system EER.
    Returns {sid: (ci_low, ci_high)}.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    bon_recs = [r for r in records if r["system_id"] == "-"]
    result   = {}
    for sid in attack_systems:
        atk_recs = [r for r in records if r["system_id"] == sid]
        combined = bon_recs + atk_recs
        n        = len(combined)
        boot_eers = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_recs = [combined[i] for i in idx]
            labels    = np.array([r["label"] for r in boot_recs])
            logits    = np.array([r["logit"] for r in boot_recs])
            scores    = 1.0 / (1.0 + np.exp(-logits))
            if labels.sum() == 0 or labels.sum() == len(labels):
                continue
            boot_eers.append(compute_eer(labels, scores))
        if boot_eers:
            result[sid] = (float(np.percentile(boot_eers, 2.5)),
                           float(np.percentile(boot_eers, 97.5)))
        else:
            result[sid] = (float("nan"), float("nan"))
    return result


def run_e4(records_cache: list[dict]) -> None:
    """Load model, run 3 ablation conditions, compute per-system EER with bootstrap CIs."""
    import io as _io
    import soundfile as sf
    import torchaudio.transforms as T
    from collections import defaultdict

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  E4 device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    def _patch_phoneme_loader():
        import phoneme_GAT.modules as mm
        import phoneme_GAT.phoneme_model as pm
        from phoneme_GAT.phoneme_model import BaseModule, network_param, optim_param

        def _load(network_name="wavlm", pretrained_path=None, total_num_phonemes=198):
            network_param.network_name = network_name
            network_param.pretrained_name = (
                "microsoft/wavlm-base" if network_name.lower() == "wavlm"
                else "facebook/wav2vec2-base-960h")
            network_param.vocab_size = total_num_phonemes
            if pretrained_path and Path(pretrained_path).exists():
                return BaseModule.load_from_checkpoint(
                    str(pretrained_path), network_param=network_param,
                    optim_param=optim_param, tokenizer=None,
                    total_num_phonemes=total_num_phonemes, weights_only=False).cpu()
            return BaseModule(network_param, optim_param, tokenizer=None,
                              total_num_phonemes=total_num_phonemes)

        pm.load_phoneme_model = _load
        mm.load_phoneme_model = _load

    _patch_phoneme_loader()
    from phoneme_GAT.modules import Phoneme_GAT_lit
    cfg = Namespace(PhonemeGAT=Namespace(
        backbone="wavlm", use_raw=False, use_GAT=True,
        n_edges=10, use_aug=True, use_pool=True, use_clip=True))
    lit = Phoneme_GAT_lit.load_from_checkpoint(
        str(CKPT), cfg=cfg, map_location=device, strict=True)
    lit.to(device); lit.eval(); lit.freeze()
    print(f"  Model loaded: {CKPT.name}")

    # ── Build dataloader using the SAME samples as all_layers_artifacts ───────
    TARGET_SR      = 16_000
    TARGET_SAMPLES = 3 * TARGET_SR
    N_PER_CLASS    = int(os.environ.get("N_PER_CLASS", 50))
    BATCH_SIZE     = int(os.environ.get("BATCH_SIZE", 8))
    HF_DATASET     = "Bisher/ASVspoof_2019_LA"
    CACHE_DIR      = REPO_ROOT / "data" / "asvspoof_2019_la"
    HF_TOKEN_PATH  = REPO_ROOT / "secret.txt"

    def _decode(entry: dict) -> torch.Tensor:
        raw = entry.get("bytes"); path = entry.get("path")
        arr, sr = (sf.read(_io.BytesIO(raw), dtype="float32", always_2d=False)
                   if raw is not None else sf.read(path, dtype="float32", always_2d=False))
        w = torch.tensor(arr)
        if w.ndim == 1: w = w.unsqueeze(0)
        elif w.ndim == 2: w = w.mean(0, keepdim=True)
        if sr != TARGET_SR: w = T.Resample(sr, TARGET_SR)(w)
        return w

    def _crop(w: torch.Tensor) -> torch.Tensor:
        n = w.shape[-1]
        if n < TARGET_SAMPLES: w = w.repeat(1, -(-TARGET_SAMPLES // n))
        s = (w.shape[-1] - TARGET_SAMPLES) // 2
        return w[:, s : s + TARGET_SAMPLES]

    def _lbl(raw) -> int:
        if isinstance(raw, str):
            s = raw.strip().lower()
            return 0 if s in ("0", "bonafide", "real", "genuine") else 1
        return int(raw)

    class BalancedDataset(torch.utils.data.Dataset):
        def __init__(self, hf_name, split, cache_dir, token, n_per_class, seed=42):
            from datasets import load_dataset, Audio as HFAudio
            ds = load_dataset(hf_name, split=split, cache_dir=str(cache_dir),
                              token=token)
            self.ds = ds.cast_column("audio", HFAudio(decode=False))
            ex0 = self.ds[0]
            self.label_key = next(
                (k for k in ex0 if k != "audio" and ("label" in k.lower()
                                                       or k.lower() == "key")),
                "label")
            by_system: dict[str, list[int]] = defaultdict(list)
            for i in range(len(self.ds)):
                sid = self.ds[i].get("system_id", "unknown")
                by_system[sid].append(i)
            rng = random.Random(seed)
            selected, sys_ids = [], []
            for sid, idxs in sorted(by_system.items()):
                rng.shuffle(idxs)
                chosen = idxs[:n_per_class]
                selected.extend(chosen)
                sys_ids.extend([sid] * len(chosen))
            combined = list(zip(selected, sys_ids))
            rng.shuffle(combined)
            self.indices, self.sys_ids = zip(*combined) if combined else ([], [])
            self.indices = list(self.indices); self.sys_ids = list(self.sys_ids)

        def __len__(self): return len(self.indices)
        def __getitem__(self, idx):
            ex = self.ds[self.indices[idx]]
            return {"audio":     _crop(_decode(ex["audio"])),
                    "label":     torch.tensor(_lbl(ex[self.label_key]), dtype=torch.long),
                    "system_id": self.sys_ids[idx]}

    def _collate(batch):
        return {"audio":     torch.stack([b["audio"] for b in batch]),
                "label":     torch.stack([b["label"] for b in batch]),
                "system_id": [b["system_id"] for b in batch]}

    hf_token = HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None
    dataset  = BalancedDataset(HF_DATASET, "validation", CACHE_DIR,
                               hf_token, N_PER_CLASS, SEED)
    loader   = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=_collate)

    attack_systems = sorted(set(r["system_id"] for r in records_cache
                                if r["system_id"] != "-"))

    # ── Run 3 conditions ──────────────────────────────────────────────────────
    conditions: list[tuple[str, list[int]]] = [
        ("baseline",      []),
        ("attn_ablated",  list(range(6))),     # zero all 6 heads at all 3 layers
        ("critical_only", CRITICAL_HEADS),     # zero h0,h4 at all 3 layers
    ]

    all_records: dict[str, list[dict]] = {}
    for cond_name, heads_to_zero in conditions:
        print(f"\n  Condition: {cond_name}  heads_to_zero={heads_to_zero}")
        t0   = time.time()
        recs = run_eval_condition(lit, loader, device, heads_to_zero)
        all_records[cond_name] = recs
        total_eer = compute_eer(
            np.array([r["label"] for r in recs]),
            1.0 / (1.0 + np.exp(-np.array([r["logit"] for r in recs]))))
        print(f"    Done ({time.time()-t0:.1f}s)  overall EER={total_eer:.4f}")

    # ── Per-system EER + bootstrap CIs ───────────────────────────────────────
    print("\n--- Per-system EER with bootstrap CIs ---")
    eer_results  = {}
    ci_results   = {}
    for cond_name, _ in conditions:
        eer_results[cond_name] = per_system_eer(all_records[cond_name], attack_systems)
        ci_results[cond_name]  = bootstrap_eer_ci(all_records[cond_name], attack_systems)

    # Print table
    print(f"\n{'system':8s}  {'baseline EER':>14s}  {'attn_ablated':>14s}  {'critical_only':>14s}  "
          f"{'Δ(attn_abl)':>12s}  {'Δ(crit_only)':>12s}")
    print("-" * 90)
    table_rows = []
    for sid in sorted(attack_systems):
        b  = eer_results["baseline"][sid]
        aa = eer_results["attn_ablated"][sid]
        co = eer_results["critical_only"][sid]
        delta_aa = aa - b
        delta_co = co - b
        ci_b  = ci_results["baseline"][sid]
        ci_aa = ci_results["attn_ablated"][sid]
        ci_co = ci_results["critical_only"][sid]
        print(f"  {sid:6s}  "
              f"{b:.4f} [{ci_b[0]:.3f},{ci_b[1]:.3f}]  "
              f"{aa:.4f} [{ci_aa[0]:.3f},{ci_aa[1]:.3f}]  "
              f"{co:.4f} [{ci_co[0]:.3f},{ci_co[1]:.3f}]  "
              f"{delta_aa:+.4f}  {delta_co:+.4f}")
        table_rows.append({
            "system": sid,
            "eer_baseline": b,
            "eer_attn_ablated": aa,
            "eer_critical_only": co,
            "delta_attn_ablated": delta_aa,
            "delta_critical_only": delta_co,
            "ci_baseline_lo": ci_b[0],  "ci_baseline_hi": ci_b[1],
            "ci_attn_abl_lo": ci_aa[0], "ci_attn_abl_hi": ci_aa[1],
            "ci_crit_only_lo": ci_co[0],"ci_crit_only_hi": ci_co[1],
        })

    # Save CSV
    csv_path = OUT_E4 / "e4_per_system_eer.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        w.writeheader(); w.writerows(table_rows)
    print(f"\n  Saved: {csv_path}")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    _plot_e4(table_rows, attack_systems)

    # ── Markdown report ───────────────────────────────────────────────────────
    _write_e4_report(table_rows, attack_systems)


def _plot_e4(table_rows: list[dict], attack_systems: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sids  = sorted(attack_systems)
    x     = np.arange(len(sids))
    w     = 0.28

    base  = [next(r for r in table_rows if r["system"] == s)["eer_baseline"]      for s in sids]
    aa    = [next(r for r in table_rows if r["system"] == s)["eer_attn_ablated"]   for s in sids]
    co    = [next(r for r in table_rows if r["system"] == s)["eer_critical_only"]  for s in sids]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w,   base, w, label="baseline",       color="#4C72B0", alpha=0.85)
    ax.bar(x,       aa,   w, label="attn_ablated",   color="#C44E52", alpha=0.85)
    ax.bar(x + w,   co,   w, label="critical_only",  color="#55A868", alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(sids, fontsize=10)
    ax.set_ylabel("EER", fontsize=11)
    ax.set_title("E4: Per-system EER under attention ablation (all 3 layers)", fontsize=12)
    ax.legend(fontsize=9)
    ax.axhline(0.5, color="grey", lw=0.7, ls="--", label="chance")
    plt.tight_layout()
    out = OUT_E4 / "e4_per_system_eer.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _write_e4_report(table_rows: list[dict], attack_systems: list[str]) -> None:
    lines = ["# E4: Per-System EER Under Attention Ablation (All 3 Layers)\n",
             "## Conditions\n",
             "- **baseline**: no ablation",
             "- **attn_ablated**: zero ALL 6 heads at ALL 3 layers (skip path only)",
             "- **critical_only**: zero h0,h4 at ALL 3 layers\n",
             "## Results\n",
             "| system | EER baseline | EER attn_ablated | EER critical_only "
             "| Δ(attn_abl) | Δ(crit_only) |",
             "|--------|-------------|-----------------|------------------|"
             "------------|-------------|"]
    for row in sorted(table_rows, key=lambda r: r["system"]):
        lines.append(
            f"| {row['system']} "
            f"| {row['eer_baseline']:.4f} "
            f"| {row['eer_attn_ablated']:.4f} "
            f"| {row['eer_critical_only']:.4f} "
            f"| {row['delta_attn_ablated']:+.4f} "
            f"| {row['delta_critical_only']:+.4f} |"
        )
    lines += ["",
              "## Pre-specified predictions\n",
              "- A01/A03/A04 degrade under attn_ablated (engage GAT routing path)",
              "- A05/A06 show small degradation (skip-route hypothesis: "
              "skip_proj carries signal even without attention)",
              "- critical_only degrades proportionally to attn_ablated for "
              "h0/h4-mediated systems",
              ""]
    (OUT_E4 / "e4_report.md").write_text("\n".join(lines))
    print(f"  Saved: {OUT_E4 / 'e4_report.md'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    run_e4_flag = "--run-e4" in sys.argv

    if not ALL_LAYERS.exists():
        print(f"ERROR: all_layers_artifacts.pt not found at {ALL_LAYERS}")
        print("Run experiments/gat_attn_graphs.py first to build the cache.")
        sys.exit(1)

    print(f"Loading all-layer artifacts: {ALL_LAYERS}")
    records = load_all_layers(ALL_LAYERS)
    print(f"  {len(records)} samples  ({sum(1 for r in records if not r['is_degenerate'])} non-degenerate)")

    id_to_sym, _ = build_vocab()

    # ── E3 ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("E3: Head-Decomposed Hub-Mass Analysis")
    print("=" * 70)
    run_e3(records, id_to_sym)

    if not run_e4_flag:
        print("\n" + "=" * 70)
        print("E3 complete.  To run E4 (requires model + dataset), use:")
        print("  venv/bin/python3 experiments/gat_e3_e4.py --run-e4")
        print("=" * 70)
        return

    # ── E4 ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("E4: Per-System EER Under Attention Ablation")
    print("=" * 70)
    run_e4(records)
    print(f"\nAll E3/E4 outputs in: {OUT_BASE}/")


if __name__ == "__main__":
    main()
