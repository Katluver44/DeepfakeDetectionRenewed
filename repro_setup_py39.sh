#!/usr/bin/env bash
# ==============================================================================
# DeepfakeDetectionRenewed — Reproducible Environment (Python 3.9 + cu121)
#
# Dependencies (exact pins tested on py3.9):
#   - torch==2.5.1+cu121, torchvision==0.20.1+cu121, torchaudio==2.5.1+cu121
#   - numpy==1.24.4, scipy==1.11.4, pandas==2.2.3, librosa==0.10.1
#   - pytorch_lightning==2.3.3, transformers==4.36.2, einops==0.8.0
#   - yacs==0.1.8, phonemizer==3.2.1, soundfile (latest)
#   - datasets==3.0.1, python-dotenv, gdown
#   - (optional) audiomentations==0.38.0
#
# System packages:
#   espeak-ng espeak-ng-data libespeak-ng1 ffmpeg sox libsndfile1 build-essential
#
# Project env vars / paths:
#   export VOCAB_PHONEME_DIR="$PROJECT_ROOT/vocab_phoneme"
#   pretrained ckpt -> "$PROJECT_ROOT/pretrained/best-epoch=42-val-per=0.407000.ckpt"
#   Jupyter kernel name -> "PLFD-ADD (py3.9)"
#
# Run:
#   bash repro_setup_py39.sh
# ==============================================================================

set -euo pipefail

# --- 0) Project path (defaults to current dir) ---
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
VENV="${VENV:-$PROJECT_ROOT/venv}"
CUDA_CHANNEL="cu121"

echo "PROJECT_ROOT = $PROJECT_ROOT"
echo "VENV         = $VENV"

# --- 1) OS prereqs ---
if command -v apt >/dev/null 2>&1; then
  sudo apt update
  sudo apt install -y python3.9 python3.9-venv espeak-ng espeak-ng-data libespeak-ng1 ffmpeg sox libsndfile1 build-essential
fi

# --- 2) Python 3.9 venv ---
deactivate 2>/dev/null || true
rm -rf "$VENV"
python3.9 -m venv "$VENV"
source "$VENV/bin/activate"
python --version
pip install -U pip setuptools wheel

# --- 3) PyTorch (cu121 wheels; driver 12.x OK incl. 12.7) ---
pip uninstall -y torch torchvision torchaudio || true
pip cache purge || true
pip install --index-url "https://download.pytorch.org/whl/${CUDA_CHANNEL}" \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Make PyTorch index default so future installs keep cu121
mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf <<EOF
[global]
index-url = https://download.pytorch.org/whl/${CUDA_CHANNEL}
extra-index-url = https://pypi.org/simple
EOF

# --- 4) Core Python deps (tested pins) ---
pip install \
  numpy==1.24.4 \
  scipy==1.11.4 \
  pandas==2.2.3 \
  librosa==0.10.1 \
  pytorch_lightning==2.3.3 \
  transformers==4.36.2 \
  einops==0.8.0 \
  yacs==0.1.8 \
  phonemizer==3.2.1 \
  soundfile \
  datasets==3.0.1 \
  python-dotenv \
  gdown

# Optional augments
pip install audiomentations==0.38.0 || true

# --- 5) Record a clean requirements (minus torch trio which is cu121-specific) ---
cat > "$PROJECT_ROOT/requirements.clean.txt" <<'EOF'
audiomentations==0.38.0
datasets==3.0.1
einops==0.8.0
librosa==0.10.1
numpy==1.24.4
pandas==2.2.3
phonemizer==3.2.1
python-dotenv
pytorch_lightning==2.3.3
scipy==1.11.4
soundfile
transformers==4.36.2
yacs==0.1.8
# Torch/torchaudio/torchvision installed via cu121 index
EOF

# --- 6) Ensure pretrained checkpoint exists (download if missing) ---
mkdir -p "$PROJECT_ROOT/pretrained"
CKPT="$PROJECT_ROOT/pretrained/best-epoch=42-val-per=0.407000.ckpt"
if [ ! -f "$CKPT" ]; then
  echo "Downloading checkpoint via gdown…"
  # Google Drive file id for the ckpt provided
  gdown --id 1SbqynkUQxxlhazklZz9OgcVK7Fl2aT-z -O "$CKPT"
fi

# --- 7) Helpful env var for vocab path (append to .bashrc for persistence) ---
if ! grep -q "VOCAB_PHONEME_DIR" ~/.bashrc 2>/dev/null; then
  echo "export VOCAB_PHONEME_DIR=\"$PROJECT_ROOT/vocab_phoneme\"" >> ~/.bashrc
fi
export VOCAB_PHONEME_DIR="$PROJECT_ROOT/vocab_phoneme"

# --- 8) Register Jupyter kernel ---
python -m ipykernel install --user --name=plfd39 --display-name "PLFD-ADD (py3.9)"

# --- 9) Sanity printout ---
python - <<'PY'
import os, platform, torch, torchaudio, numpy as np, scipy, pandas as pd, librosa
print("Python:", platform.python_version())
print("Torch :", torch.__version__, "| CUDA:", torch.version.cuda, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available(): print("GPU   :", torch.cuda.get_device_name(0))
print("torchaudio:", torchaudio.__version__)
print("numpy:", np.__version__, "| scipy:", scipy.__version__, "| pandas:", pd.__version__, "| librosa:", librosa.__version__)
print("VOCAB_PHONEME_DIR:", os.environ.get("VOCAB_PHONEME_DIR"))
PY

echo "✅ Done. In Jupyter/VS Code, choose kernel: 'PLFD-ADD (py3.9)'."
