#!/usr/bin/env bash
# Unattended driver for the MLAAD-side reproduction chain.
# Waits for i2 (already running) to finish, then runs the runnable downstream
# scripts in dependency order. ASVspoof-only steps (i1/i4/j4) are skipped here
# because they need the truncated seed3/seed7 checkpoints.
set -u
cd /home/sagemaker-user/DeepfakeDetectionRenewed
export HF_TOKEN=$(cat secret.txt 2>/dev/null)
LOG=logs
mkdir -p "$LOG"

echo "[driver] waiting for i2 to finish..."
while pgrep -f i2_geometry_battery.py >/dev/null; do sleep 20; done
echo "[driver] i2 finished. wave cache: $(ls -la outputs/px_wave_cache/i2_full_test_waves.npz 2>/dev/null | awk '{print $5}')"

run () {   # name  script
  local name="$1" script="$2"
  echo "[driver] === $name ($script) ==="
  local t0=$(date +%s)
  if python "$script" > "$LOG/${name}.log" 2>&1; then
    echo "[driver] $name OK ($(( $(date +%s)-t0 ))s)"
  else
    echo "[driver] $name FAILED ($(( $(date +%s)-t0 ))s) -- tail:"
    tail -6 "$LOG/${name}.log"
  fi
}

# MLAAD geometry/position/fusion chain
run i3 experiments/scripts/i3_position_geometry.py
run i7 experiments/scripts/i7_axis_fusion.py
run i6 experiments/scripts/i6_testtime_geometry_boost.py
run i5 experiments/scripts/i5_itw_transfer.py
# AASIST MLAAD experiments (assets present)
run j5 experiments/scripts/j5_aasist_crossfamily.py
run j6 experiments/scripts/j6_train_aasist_mlaad.py
# MLAAD audits that depend on the wave cache / i7 fusion
run audit2 experiments/axis_audits/audit2_sdalong_claim.py
run audit4 experiments/axis_audits/audit4_axis_rotation.py
run audit5 experiments/axis_audits/audit5_fusion_claims.py
run audit9 experiments/axis_audits/audit9_hardness_reliability.py

echo "[driver] DONE"
