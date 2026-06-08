#!/usr/bin/env bash
# Run a single (experiment, seed) training job. Used by the parallel pool.
# Usage: _run_one.sh <exp> <seed>
set -u
cd /lambda/nfs/algovirginia/workspace/DeepfakeDetectionRenewed
PY=venv/bin/python
BASE=experiments/results/mlaad/p1_ct_calibration/per_system_eer.csv
LOGD=experiments/results/mlaad/ct_feature_injection/run_logs
mkdir -p "$LOGD"
exp="$1"; seed="$2"
log="$LOGD/${exp}_seed${seed}.log"

case "$exp" in
  p2) cmd=("$PY" experiments/scripts/p2_ct_feature_injection_train.py --seeds "$seed" --n-ct 2 --epochs 5 --baseline-csv "$BASE");;
  p3) cmd=("$PY" experiments/scripts/p2_ct_feature_injection_train.py --seeds "$seed" --n-ct 3 --epochs 5 --baseline-csv "$BASE");;
  p4) cmd=("$PY" experiments/scripts/p4_smoothing_aug_finetune.py --seeds "$seed" --epochs 5 --baseline-csv "$BASE");;
  p5) cmd=("$PY" experiments/scripts/p5_system_hardness_reweight.py --seeds "$seed" --epochs 5);;
  p6) cmd=("$PY" experiments/scripts/p6_diversity_regularization.py --seeds "$seed" --epochs 5 --lambda-div 0.01);;
  p7) cmd=("$PY" experiments/scripts/p7_clip_c_ranking.py --seeds "$seed" --epochs 5 --lambda-rank 0.1);;
  *) echo "unknown exp $exp" >&2; exit 1;;
esac

echo ">>> $exp seed$seed START $(date -u +%FT%TZ)"
"${cmd[@]}" > "$log" 2>&1
rc=$?
echo ">>> $exp seed$seed END rc=$rc $(date -u +%FT%TZ)"
exit $rc
