#!/usr/bin/env bash
# The /opt/conda env resets on session boundaries. Re-run this to restore deps.
set -e
python -m pip install -q soundfile einops
python -m pip install -q --force-reinstall "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu129
python - <<'PY'
import torch, torchaudio, soundfile, einops
print("torch", torch.__version__, "| torchaudio", torchaudio.__version__, "| soundfile", soundfile.__version__, "| einops OK")
PY
