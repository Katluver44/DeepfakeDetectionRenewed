#!/usr/bin/env bash
# Wait for the user's robust_goat_seed3.ckpt upload to finish, validate it, then run I1.
cd /home/sagemaker-user/DeepfakeDetectionRenewed
export HF_TOKEN=$(cat secret.txt 2>/dev/null)
CK=models/good_models/robust_goat_seed3.ckpt

echo "[i1drv] waiting for $CK upload to stabilize..."
last=-1
while true; do
  cur=$(stat -c%s "$CK" 2>/dev/null || echo 0)
  if [ "$cur" = "$last" ] && [ "$cur" -gt 400000000 ]; then break; fi
  last=$cur
  sleep 15
done
echo "[i1drv] upload stable at $last bytes. validating..."

python - <<PY
import torch, sys
try:
    o=torch.load("$CK", map_location="cpu", weights_only=False)
    sd=o.get("state_dict",o) if isinstance(o,dict) else o
    n=sum(1 for v in sd.values() if hasattr(v,"shape"))
    print(f"[i1drv] checkpoint OK: {n} tensors")
    sys.exit(0 if n>0 else 2)
except Exception as e:
    print(f"[i1drv] checkpoint INVALID: {repr(e)[:120]}")
    sys.exit(2)
PY
if [ $? -ne 0 ]; then echo "[i1drv] ABORT: seed3 checkpoint not loadable"; exit 1; fi

# keep models/robust_goat_seed3.ckpt consistent with the valid good_models copy
ln -sf good_models/robust_goat_seed3.ckpt models/robust_goat_seed3.ckpt

echo "[i1drv] running I1 ..."
python -u experiments/scripts/i1_geometry_causal_decomp.py > logs/i1.log 2>&1
echo "[i1drv] I1 exit=$? ; tail:"
tail -15 logs/i1.log
