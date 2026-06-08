#!/usr/bin/env bash
# Parallel driver for the C/T ablation roadmap. Runs all (experiment, seed)
# training jobs through a concurrency pool on the single A100, then aggregates
# each experiment's multi-metric summary.md and runs the P8 diagnostic.
#
# Speed levers already baked into the scripts: bf16-mixed + TF32 + cudnn autotune.
# This adds job-level concurrency to fill the ~15-20% GPU stall gaps.
set -u
cd /lambda/nfs/algovirginia/workspace/DeepfakeDetectionRenewed
PY=venv/bin/python
BASE=experiments/results/mlaad/p1_ct_calibration/per_system_eer.csv
LOGD=experiments/results/mlaad/ct_feature_injection/run_logs
mkdir -p "$LOGD"
STAMP="$LOGD/parallel_status.log"
K="${1:-6}"     # pool size
echo "parallel chain start $(date -u +%FT%TZ)  pool=$K" > "$STAMP"

# Stop any prior sequential orchestrator / in-flight fp32 runs so they don't
# collide with this parallel run.
pkill -f run_ablation_chain.sh 2>/dev/null
pkill -f p2_ct_feature_injection_train.py 2>/dev/null
pkill -f p4_smoothing_aug_finetune.py 2>/dev/null
sleep 6
echo "cleared prior runs; gpu:" >> "$STAMP"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> "$STAMP" 2>/dev/null

# 18 jobs: 6 experiments x 3 seeds, fed as (exp seed) token pairs to xargs -n2.
JOBS=""
for exp in p2 p3 p4 p5 p6 p7; do
  for s in 42 123 1024; do JOBS="$JOBS $exp $s"; done
done
echo "launching pool over: $JOBS" >> "$STAMP"
echo "$JOBS" | xargs -P "$K" -n2 bash experiments/scripts/_run_one.sh >> "$STAMP" 2>&1
echo "all training jobs finished $(date -u +%FT%TZ)" >> "$STAMP"

# Aggregate per-experiment multi-metric summaries from the per-seed CSVs.
agg() { $PY experiments/scripts/aggregate_summary.py --dir "$1" --name "$2" --baseline-csv "$BASE" >> "$STAMP" 2>&1; }
agg experiments/results/mlaad/ct_feature_injection      "P2 CT feature injection (n_ct=2)"
agg experiments/results/mlaad/p3_ct_window11            "P3 CT injection + T_window11 (n_ct=3)"
agg experiments/results/mlaad/p4_smoothing_aug          "P4 smoothing augmentation"
agg experiments/results/mlaad/p5_hardness_reweight      "P5 system hardness reweight"
agg experiments/results/mlaad/p6_diversity_lambda0.01   "P6 representation diversity regularization"
agg experiments/results/mlaad/p7_crank_lambda0.1        "P7 CLIP C-ranking"

# P8 inference diagnostic (fast; reuses P1 features + wav2vec2).
echo "running P8 $(date -u +%FT%TZ)" >> "$STAMP"
$PY experiments/scripts/p8_cross_encoder_c_ensemble.py > "$LOGD/p8.log" 2>&1
echo ">>> P8 rc=$? $(date -u +%FT%TZ)" >> "$STAMP"

echo "parallel chain DONE $(date -u +%FT%TZ)" >> "$STAMP"
