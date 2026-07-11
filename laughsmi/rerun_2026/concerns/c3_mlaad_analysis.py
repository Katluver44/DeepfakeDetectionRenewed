"""C3(a) — MLAAD WavLM-GAT insertion evasion analysis, reusing scripts/
d1_analysis.py::analyze() verbatim, then the d1_defense worst-window scorer."""
import sys
from pathlib import Path
LAUGHSMI = Path("/home/sagemaker-user/DeepfakeDetectionRenewed/laughsmi")
sys.path.insert(0, str(LAUGHSMI / "scripts"))
from d1_analysis import analyze
import csv

C = LAUGHSMI / "rerun_2026" / "concerns"
res = analyze("MLAAD (mlaad_wavlm-gat)",
              C / "base_mlaad.csv", C / "aug_mlaad.csv",
              LAUGHSMI / "data" / "eval_mlaad_aug" / "manifest.csv")
out = C / "tables" / "c3_mlaad_insertion.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(res.keys()))
    w.writeheader()
    w.writerow(res)
print("=== MLAAD WavLM-GAT insertion result ===")
for k, v in res.items():
    print(f"  {k}: {v}")
print("wrote", out)
