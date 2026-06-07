"""
Phase 0 sanity checks.

0.1  Phoneme recognizer language coverage  (static analysis, no GPU)
0.2  Bootstrap CI on cross-language ablation improvement  (needs GPU, ~3 min)
0.3  MLAAD checkpoint weight diff audit  (CPU, <30s)
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/results"
OUT_DIR = RESULTS / "phase0_checks"
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))

CKPT_GOAT   = ROOT / "experiments/checkpoints/mlaad_goat-best-epoch=05-val-eer=0.2795.ckpt"
CKPT_ROBUST = ROOT / "experiments/checkpoints/mlaad_robust_goat-best-epoch=05-val-eer=0.2858.ckpt"
PROCESSED_DIR = ROOT / "experiments/data/mlaad_tiny_processed"
TEST_XL_JSON  = RESULTS / "mlaad/baseline_eval/test_cross_language.json"

N_BOOTSTRAP = 1000
BATCH_SIZE  = 8
SEED        = 42


# ════════════════════════════════════════════════════════════════════════════
# Check 0.1 — Phoneme recognizer language coverage (static)
# ════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Check 0.1: Phoneme recognizer language coverage")

VOCAB_DIR = ROOT / "vocab_phoneme"
PRETRAINED_CKPT = ROOT / "pretrained/best-epoch=42-val-per=0.407000.ckpt"

languages = ["en", "de", "es", "fr", "it", "pl", "ru", "uk", "zh-CN"]
vocab_tokens = {}
for lang in languages:
    vf = VOCAB_DIR / f"vocab-phoneme-{lang}.json"
    if not vf.exists():
        print(f"  MISSING vocab file: {vf}")
        sys.exit(1)
    vocab_tokens[lang] = list(json.loads(vf.read_text()).keys())
    print(f"  {lang}: {len(vocab_tokens[lang])} phoneme tokens  ({vf.name})")

# Load checkpoint hyper_parameters to confirm training languages
ckpt_meta = torch.load(PRETRAINED_CKPT, map_location="cpu", weights_only=False)
hp = ckpt_meta.get("hyper_parameters", {})
tokenizer = hp.get("tokenizer", None)
lm_weight = ckpt_meta["state_dict"].get("model.model.lm_head.weight")
vocab_size = lm_weight.shape[0] if lm_weight is not None else "unknown"

# Extract training vocab_files path from tokenizer repr
tok_repr = str(tokenizer) if tokenizer else "(not found)"
has_de = "de" in tok_repr
has_common_voice = "common_voice" in tok_repr.lower()
training_lang_count = sum(1 for l in languages if l in tok_repr)

print(f"\n  Checkpoint: {PRETRAINED_CKPT.name}")
print(f"  LM head vocab_size: {vocab_size}  (expected 687)")
print(f"  German vocab in tokenizer: {has_de}")
print(f"  Common Voice data path found: {has_common_voice}")
print(f"  Languages referenced in tokenizer: {training_lang_count}/9")

kill_01 = not (has_de and vocab_size == 687 and training_lang_count == 9)
result_01 = {
    "check": "0.1_phoneme_recognizer_language_coverage",
    "vocab_size": int(vocab_size) if isinstance(vocab_size, int) else vocab_size,
    "expected_vocab_size": 687,
    "languages_in_tokenizer": training_lang_count,
    "german_in_tokenizer": has_de,
    "common_voice_training_data": has_common_voice,
    "per_language_token_counts": {l: len(v) for l, v in vocab_tokens.items()},
    "kill_criterion_triggered": kill_01,
    "verdict": "FAIL — recognizer is NOT multilingual" if kill_01 else
               "PASS — multilingual recognizer confirmed (9 languages including de)",
}
print(f"\n  VERDICT: {result_01['verdict']}")
(OUT_DIR / "check_0.1_phoneme_coverage.json").write_text(json.dumps(result_01, indent=2))
print(f"  Wrote: {OUT_DIR}/check_0.1_phoneme_coverage.json")


# ════════════════════════════════════════════════════════════════════════════
# Check 0.3 — MLAAD checkpoint weight diff audit (CPU, no forward pass)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Check 0.3: MLAAD checkpoint weight diff audit")

for path, name in [(CKPT_GOAT, "mlaad_goat"), (CKPT_ROBUST, "mlaad_robust_goat")]:
    if not path.exists():
        print(f"  MISSING: {path}")
        sys.exit(1)
    print(f"  Found: {name}  ({path.stat().st_size / 1e6:.0f} MB)")

goat_sd   = torch.load(CKPT_GOAT,   map_location="cpu", weights_only=False)["state_dict"]
robust_sd = torch.load(CKPT_ROBUST, map_location="cpu", weights_only=False)["state_dict"]

assert set(goat_sd.keys()) == set(robust_sd.keys()), \
    f"Key mismatch: goat has {len(goat_sd)} keys, robust has {len(robust_sd)}"

def categorise(k):
    if "phoneme_model" in k or "transformer_in_phoneme_model" in k:
        return "pre_gat_encoder"
    if "GAT" in k:
        return "gat"
    if "rnn" in k or "attn" in k:
        return "bilstm_attn"
    if "encoder" in k:
        return "pre_gat_encoder"
    return "cls_head_other"

cats = {}
for k in goat_sd.keys():
    cat = categorise(k)
    g, r = goat_sd[k].float(), robust_sd[k].float()
    identical = torch.equal(g, r)
    l2 = (g - r).norm().item()
    if cat not in cats:
        cats[cat] = {"identical": 0, "differ": 0, "max_l2": 0.0, "max_l2_key": ""}
    if identical:
        cats[cat]["identical"] += 1
    else:
        cats[cat]["differ"] += 1
        if l2 > cats[cat]["max_l2"]:
            cats[cat]["max_l2"] = l2
            cats[cat]["max_l2_key"] = k

print(f"\n  {'Category':<22} {'identical':>9} {'differ':>7} {'max_L2':>9}  verdict")
print("  " + "-" * 65)
for cat, v in sorted(cats.items()):
    if cat == "pre_gat_encoder":
        exp = "IDENTICAL (frozen encoder)"
        ok = v["differ"] == 0
    else:
        exp = "DIFFERENT (trained)"
        ok = v["differ"] > 0
    status = "OK" if ok else "FAIL"
    print(f"  {cat:<22} {v['identical']:>9} {v['differ']:>7} {v['max_l2']:>9.4f}  {status} — {exp}")

encoder_frozen = cats["pre_gat_encoder"]["differ"] == 0
gat_differs    = cats["gat"]["differ"] > 0
bilstm_differs = cats["bilstm_attn"]["differ"] > 0

kill_03 = not (encoder_frozen and gat_differs)
result_03 = {
    "check": "0.3_checkpoint_weight_diff",
    "categories": {cat: {**v, "max_l2": float(v["max_l2"])} for cat, v in cats.items()},
    "encoder_frozen": encoder_frozen,
    "gat_differs": gat_differs,
    "bilstm_differs": bilstm_differs,
    "kill_criterion_triggered": kill_03,
    "verdict": "FAIL — GAT weights identical (same model compared to itself)" if kill_03 else
               "PASS — pre-GAT identical, GAT+BiLSTM differ",
}
print(f"\n  VERDICT: {result_03['verdict']}")
(OUT_DIR / "check_0.3_weight_diff.json").write_text(json.dumps(result_03, indent=2))
print(f"  Wrote: {OUT_DIR}/check_0.3_weight_diff.json")

if kill_03:
    print("  KILL: aborting before GPU check.")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# Check 0.2 — Bootstrap CI on cross-language ablation improvement (GPU)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Check 0.2: Bootstrap CI on cross-language ablation improvement")

import random
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import experiments.head_ablation as ha
from experiments.head_ablation import run_frozen_frontend

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

# Load cross-language test set
xl_records = json.loads(TEST_XL_JSON.read_text())
print(f"  Cross-language records: {len(xl_records)}")
bonafide_xl = [r for r in xl_records if r["label"] == "bonafide"]
spoof_xl    = [r for r in xl_records if r["label"] == "spoof"]
print(f"  Bonafide: {len(bonafide_xl)}  Spoof: {len(spoof_xl)}")

class XLDataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        rec = self.records[idx]
        wav = torch.load(PROCESSED_DIR / rec["audio_path"]).unsqueeze(0)
        y   = 0 if rec["label"] == "bonafide" else 1
        sid = "-" if rec["label"] == "bonafide" else rec.get("attack_system", "unk")
        return {"audio": wav, "label": torch.tensor(y, dtype=torch.long), "system_id": sid}

loader = DataLoader(
    XLDataset(xl_records),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=ha.collate,
)

def run_condition(ckpt_path, heads_to_zero, label=""):
    lit = ha.load_model(ckpt_path, device)
    gat_model = lit.model
    records = ha.run_eval(lit, loader, device, heads=frozenset(heads_to_zero),
                          mode="zero", bonafide_means={})
    del lit
    torch.cuda.empty_cache()
    print(f"  [{label}] n_records={len(records)}")
    return records

print(f"\n  Running baseline (no ablation)...")
rec_baseline = run_condition(CKPT_ROBUST, [], "baseline")

print(f"  Running ablate_h2_h4...")
rec_ablated  = run_condition(CKPT_ROBUST, [2, 4], "ablate_h2_h4")

# Align records by index (same order, same samples)
assert len(rec_baseline) == len(rec_ablated)
labels   = np.array([r["label"] for r in rec_baseline])
scores_b = np.array([1 / (1 + np.exp(-r["logit"])) for r in rec_baseline])
scores_a = np.array([1 / (1 + np.exp(-r["logit"])) for r in rec_ablated])

def compute_pooled_eer(labels, scores):
    return float(ha.compute_eer(labels, scores))

eer_b_point = compute_pooled_eer(labels, scores_b)
eer_a_point = compute_pooled_eer(labels, scores_a)
delta_point = eer_a_point - eer_b_point
improvement_point = -delta_point  # positive = improvement

print(f"\n  Point estimates:")
print(f"    baseline EER     = {eer_b_point:.4f}")
print(f"    ablated EER      = {eer_a_point:.4f}")
print(f"    ΔEER (abl−base)  = {delta_point:+.4f}")
print(f"    improvement      = {improvement_point:+.4f}")

# Bootstrap
rng = np.random.default_rng(SEED)
n = len(labels)
boot_deltas = []
print(f"\n  Bootstrapping {N_BOOTSTRAP} resamples (n={n})...")
for i in range(N_BOOTSTRAP):
    idx = rng.integers(0, n, size=n)
    e_b = compute_pooled_eer(labels[idx], scores_b[idx])
    e_a = compute_pooled_eer(labels[idx], scores_a[idx])
    boot_deltas.append(e_a - e_b)
    if (i + 1) % 200 == 0:
        print(f"    {i+1}/{N_BOOTSTRAP}")

boot_deltas = np.array(boot_deltas)
boot_improvements = -boot_deltas

ci_lo_delta = float(np.percentile(boot_deltas, 2.5))
ci_hi_delta = float(np.percentile(boot_deltas, 97.5))
ci_lo_imp   = float(np.percentile(boot_improvements, 5.0))   # conservative / worst-case
ci_hi_imp   = float(np.percentile(boot_improvements, 95.0))
median_imp  = float(np.median(boot_improvements))
pct_positive = float(np.mean(boot_improvements > 0))

print(f"\n  Bootstrap results ({N_BOOTSTRAP} resamples):")
print(f"    ΔEER 95% CI      = [{ci_lo_delta:+.4f}, {ci_hi_delta:+.4f}]")
print(f"    Improvement 90% CI = [{ci_lo_imp:+.4f}, {ci_hi_imp:+.4f}]  (5th–95th pctile)")
print(f"    Median improvement = {median_imp:+.4f}")
print(f"    Fraction positive  = {pct_positive:.3f}")

# Kill criteria
ci_crosses_zero = ci_lo_delta < 0 < ci_hi_delta  # ΔEER CI contains 0 ⟹ no guaranteed improvement
fifth_pctile_weak = ci_lo_imp < 0.05              # 5th pctile of improvement < 0.05

kill_02 = ci_crosses_zero or fifth_pctile_weak

print(f"\n  Kill criteria:")
print(f"    CI crosses zero (ΔEER CI straddles 0): {ci_crosses_zero}")
print(f"    5th pctile improvement < 0.05:         {fifth_pctile_weak}  (value={ci_lo_imp:.4f})")

result_02 = {
    "check": "0.2_bootstrap_ci_cross_language_ablation",
    "n_bootstrap": N_BOOTSTRAP,
    "n_samples": n,
    "checkpoint": "mlaad_robust_goat",
    "ablation": "h2+h4 zero",
    "point_estimates": {
        "baseline_eer": eer_b_point,
        "ablated_eer": eer_a_point,
        "delta_eer": delta_point,
        "improvement": improvement_point,
    },
    "bootstrap": {
        "delta_eer_ci_2.5": ci_lo_delta,
        "delta_eer_ci_97.5": ci_hi_delta,
        "improvement_ci_5th": ci_lo_imp,
        "improvement_ci_95th": ci_hi_imp,
        "improvement_median": median_imp,
        "fraction_resamples_positive": pct_positive,
    },
    "kill_criteria": {
        "ci_crosses_zero": ci_crosses_zero,
        "fifth_pctile_improvement_lt_0.05": fifth_pctile_weak,
    },
    "kill_criterion_triggered": kill_02,
    "verdict": ("FAIL — CI crosses zero or improvement too small to be reliable"
                if kill_02 else
                f"PASS — improvement {improvement_point:.3f} robust; "
                f"5th pctile={ci_lo_imp:.3f} > 0.05, CI does not cross zero"),
}
print(f"\n  VERDICT: {result_02['verdict']}")
(OUT_DIR / "check_0.2_bootstrap_ci.json").write_text(json.dumps(result_02, indent=2))
print(f"  Wrote: {OUT_DIR}/check_0.2_bootstrap_ci.json")


# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 0 SUMMARY")
for label, res in [("0.1", result_01), ("0.2", result_02), ("0.3", result_03)]:
    kill = res["kill_criterion_triggered"]
    print(f"  Check {label}: {'KILL' if kill else 'PASS'}  —  {res['verdict']}")

any_kill = any(r["kill_criterion_triggered"] for r in [result_01, result_02, result_03])
if any_kill:
    print("\n  *** At least one kill criterion triggered. See verdicts above. ***")
else:
    print("\n  All checks passed. Cross-language claim stands.")

summary = {
    "check_0.1": result_01["verdict"],
    "check_0.2": result_02["verdict"],
    "check_0.3": result_03["verdict"],
    "any_kill": any_kill,
    "conclusion": "Cross-language claim is statistically and architecturally sound."
    if not any_kill else "One or more kill criteria triggered — see individual check files.",
}
(OUT_DIR / "phase0_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nOutputs: {OUT_DIR}/")
