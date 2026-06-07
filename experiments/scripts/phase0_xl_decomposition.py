"""
Phase 0 addendum: Cross-language EER decomposition by language.

The cross-language test set contains:
  bonafide_en: 908, bonafide_de: 199, spoof_de: 220

The pooled EER (baseline=0.677, ablated=0.518) mixes:
  (a) EN bonafide vs DE spoof  — purely spoof-language mismatch
  (b) DE bonafide vs DE spoof  — same-language, but out-of-distribution bonafide

This script re-runs the forward pass and decomposes the ablation improvement
into its language-specific sources.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RESULTS   = ROOT / "experiments/results"
OUT_DIR   = RESULTS / "phase0_checks"
PROCESSED = ROOT / "experiments/data/mlaad_tiny_processed"
TEST_XL   = RESULTS / "mlaad/baseline_eval/test_cross_language.json"
CKPT      = ROOT / "experiments/checkpoints/mlaad_robust_goat-best-epoch=05-val-eer=0.2858.ckpt"

import experiments.head_ablation as ha

BATCH_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load test set, annotate language
xl_records = json.loads(TEST_XL.read_text())
print(f"Cross-language records: {len(xl_records)}")
for r in xl_records:
    # language is stored in the record
    pass

bonafide_en = [r for r in xl_records if r["label"] == "bonafide" and r.get("language") == "en"]
bonafide_de = [r for r in xl_records if r["label"] == "bonafide" and r.get("language") == "de"]
spoof_de    = [r for r in xl_records if r["label"] == "spoof"]
print(f"  bonafide_en={len(bonafide_en)}  bonafide_de={len(bonafide_de)}  spoof_de={len(spoof_de)}")

class XLDataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        rec = self.records[idx]
        wav = torch.load(PROCESSED / rec["audio_path"]).unsqueeze(0)
        y   = 0 if rec["label"] == "bonafide" else 1
        sid = "-" if rec["label"] == "bonafide" else rec.get("attack_system", "unk")
        return {"audio": wav, "label": torch.tensor(y, dtype=torch.long),
                "system_id": sid, "_language": rec.get("language", "?"),
                "_idx": idx}

def run_cond(heads, label=""):
    loader = DataLoader(XLDataset(xl_records), batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=ha.collate)
    lit = ha.load_model(CKPT, device)
    records = ha.run_eval(lit, loader, device, heads=frozenset(heads),
                          mode="zero", bonafide_means={})
    del lit; torch.cuda.empty_cache()
    print(f"  [{label}] {len(records)} records")
    return records

print("\nRunning baseline...")
rec_base = run_cond([], "baseline")
print("Running ablate_h2_h4...")
rec_abl  = run_cond([2, 4], "ablate_h2_h4")

# Attach language back by index (records come out in loader order = xl_records order)
for i, (rb, ra) in enumerate(zip(rec_base, rec_abl)):
    lang = xl_records[i].get("language", "?")
    rb["language"] = lang
    ra["language"] = lang

def compute_eer(labels, scores):
    return float(ha.compute_eer(labels, scores))

def subset_eer(records, bon_langs, spoof_langs=("de",)):
    """EER computed on a specific language subset of bonafide vs spoof."""
    bon   = [r for r in records if r["label"] == 0 and r.get("language") in bon_langs]
    spoof = [r for r in records if r["label"] == 1 and r.get("language") in spoof_langs]
    if not bon or not spoof:
        return None
    labels = np.array([0]*len(bon) + [1]*len(spoof))
    scores = np.array([1/(1+np.exp(-r["logit"])) for r in bon+spoof])
    return compute_eer(labels, scores)

slices = {
    "pooled (all bonafide vs DE spoof)":     (["en","de"], ["de"]),
    "EN bonafide vs DE spoof (lang mismatch)": (["en"],     ["de"]),
    "DE bonafide vs DE spoof (same lang)":    (["de"],      ["de"]),
}

print("\n{'Slice':<45} {'Baseline':>10} {'Ablated':>10} {'Δ':>8} {'Improvement':>12}")
print("-" * 90)
results = {}
for name, (bon_langs, spoof_langs) in slices.items():
    e_b = subset_eer(rec_base, bon_langs, spoof_langs)
    e_a = subset_eer(rec_abl,  bon_langs, spoof_langs)
    if e_b is None or e_a is None:
        print(f"  {name:<45}  N/A")
        continue
    delta = e_a - e_b
    print(f"  {name:<45} {e_b:>10.4f} {e_a:>10.4f} {delta:>+8.4f} {-delta:>+12.4f}")
    results[name] = {"baseline": e_b, "ablated": e_a, "delta": delta, "improvement": -delta}

# Fraction of improvement attributable to each slice
pooled_imp = results.get("pooled (all bonafide vs DE spoof)", {}).get("improvement", 1)
print(f"\nImprovement decomposition:")
for name, v in results.items():
    if name == "pooled (all bonafide vs DE spoof)": continue
    frac = v["improvement"] / pooled_imp if pooled_imp != 0 else 0
    print(f"  {name}: improvement={v['improvement']:+.4f}  ({frac*100:.0f}% of pooled)")

# Check: is German bonafide being systematically misclassified as spoof?
bon_de_scores  = [1/(1+np.exp(-r["logit"])) for r in rec_base if r["label"]==0 and r.get("language")=="de"]
bon_en_scores  = [1/(1+np.exp(-r["logit"])) for r in rec_base if r["label"]==0 and r.get("language")=="en"]
spoof_de_scores = [1/(1+np.exp(-r["logit"])) for r in rec_base if r["label"]==1]
print(f"\nBaseline score distributions (higher score = more spoof-like):")
print(f"  bonafide EN: mean={np.mean(bon_en_scores):.4f}  std={np.std(bon_en_scores):.4f}")
print(f"  bonafide DE: mean={np.mean(bon_de_scores):.4f}  std={np.std(bon_de_scores):.4f}")
print(f"  spoof    DE: mean={np.mean(spoof_de_scores):.4f}  std={np.std(spoof_de_scores):.4f}")

bon_de_abl  = [1/(1+np.exp(-r["logit"])) for r in rec_abl if r["label"]==0 and r.get("language")=="de"]
bon_en_abl  = [1/(1+np.exp(-r["logit"])) for r in rec_abl if r["label"]==0 and r.get("language")=="en"]
spoof_de_abl= [1/(1+np.exp(-r["logit"])) for r in rec_abl if r["label"]==1]
print(f"Ablated score distributions:")
print(f"  bonafide EN: mean={np.mean(bon_en_abl):.4f}  std={np.std(bon_en_abl):.4f}  Δ={np.mean(bon_en_abl)-np.mean(bon_en_scores):+.4f}")
print(f"  bonafide DE: mean={np.mean(bon_de_abl):.4f}  std={np.std(bon_de_abl):.4f}  Δ={np.mean(bon_de_abl)-np.mean(bon_de_scores):+.4f}")
print(f"  spoof    DE: mean={np.mean(spoof_de_abl):.4f}  std={np.std(spoof_de_abl):.4f}  Δ={np.mean(spoof_de_abl)-np.mean(spoof_de_scores):+.4f}")

output = {
    "test_set_composition": {
        "bonafide_en": len(bonafide_en),
        "bonafide_de": len(bonafide_de),
        "spoof_de": len(spoof_de),
    },
    "eer_by_slice": results,
    "score_distributions": {
        "baseline": {
            "bonafide_en": {"mean": float(np.mean(bon_en_scores)), "std": float(np.std(bon_en_scores))},
            "bonafide_de": {"mean": float(np.mean(bon_de_scores)), "std": float(np.std(bon_de_scores))},
            "spoof_de":    {"mean": float(np.mean(spoof_de_scores)), "std": float(np.std(spoof_de_scores))},
        },
        "ablated": {
            "bonafide_en": {"mean": float(np.mean(bon_en_abl)), "std": float(np.std(bon_en_abl))},
            "bonafide_de": {"mean": float(np.mean(bon_de_abl)), "std": float(np.std(bon_de_abl))},
            "spoof_de":    {"mean": float(np.mean(spoof_de_abl)), "std": float(np.std(spoof_de_abl))},
        },
    },
    "interpretation": (
        "If EN-bonafide-vs-DE-spoof EER < DE-bonafide-vs-DE-spoof EER: "
        "German bonafide confusion drives cross-language degradation. "
        "If improvement is concentrated in the EN-vs-DE slice: ablation helps "
        "spoof-language mismatch but not bonafide-language recognition."
    ),
}
out_path = OUT_DIR / "check_0.4_xl_decomposition.json"
out_path.write_text(json.dumps(output, indent=2))
print(f"\nWrote: {out_path}")
