#!/usr/bin/env bash
# Sequential driver for the C/T ablation roadmap (P3–P8). Waits for any
# in-flight P2 run, then runs each experiment one at a time (single GPU),
# writing a summary.md per experiment. Continues past a failing stage so one
# crash doesn't block the rest; records status to the orchestrator log.
set -u
cd /lambda/nfs/algovirginia/workspace/DeepfakeDetectionRenewed
PY=venv/bin/python
BASE=experiments/results/mlaad/p1_ct_calibration/per_system_eer.csv
LOGD=experiments/results/mlaad/ct_feature_injection/run_logs
mkdir -p "$LOGD"
STAMP="$LOGD/chain_status.log"
SEEDS="42 123 1024"
echo "chain start $(date -u +%FT%TZ)" > "$STAMP"

# Wait for an existing P2 process (passed as $1) to exit, if any.
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  echo "waiting for P2 pid $1 ..." | tee -a "$STAMP"
  until ! kill -0 "$1" 2>/dev/null; do sleep 30; done
  echo "P2 pid $1 finished $(date -u +%FT%TZ)" | tee -a "$STAMP"
fi

run() {  # run <name> <logfile> <cmd...>
  local name="$1"; local log="$2"; shift 2
  echo ">>> $name START $(date -u +%FT%TZ)" | tee -a "$STAMP"
  "$@" > "$log" 2>&1
  local rc=$?
  echo ">>> $name END rc=$rc $(date -u +%FT%TZ)" | tee -a "$STAMP"
}

run "P3_ct_window11" "$LOGD/p3.log" \
  $PY experiments/scripts/p2_ct_feature_injection_train.py \
  --seeds $SEEDS --n-ct 3 --epochs 5 --baseline-csv "$BASE"

run "P4_smoothing_aug" "$LOGD/p4.log" \
  $PY experiments/scripts/p4_smoothing_aug_finetune.py \
  --seeds $SEEDS --epochs 5 --baseline-csv "$BASE"

run "P5_hardness_reweight" "$LOGD/p5.log" \
  $PY experiments/scripts/p5_system_hardness_reweight.py \
  --seeds $SEEDS --epochs 5

run "P6_diversity_reg" "$LOGD/p6.log" \
  $PY experiments/scripts/p6_diversity_regularization.py \
  --seeds $SEEDS --epochs 5 --lambda-div 0.01

run "P7_clip_c_ranking" "$LOGD/p7.log" \
  $PY experiments/scripts/p7_clip_c_ranking.py \
  --seeds $SEEDS --epochs 5 --lambda-rank 0.1

run "P8_cross_encoder_ensemble" "$LOGD/p8.log" \
  $PY experiments/scripts/p8_cross_encoder_c_ensemble.py

echo "chain done $(date -u +%FT%TZ)" | tee -a "$STAMP"
