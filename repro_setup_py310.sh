set -euo pipefail

# --- config ---
PYBIN="${PYBIN:-python3.10}"          # change if you want python3.10 etc 
VENV="${VENV:-venv}"
CUDA_WHL="cu124"                     # torch 2.6 uses cu124 wheels

# --- wipe + recreate venv ---
deactivate 2>/dev/null || true
rm -rf "$VENV"
$PYBIN -m venv "$VENV"
source "$VENV/bin/activate"

python -V
pip install -U pip setuptools wheel

# --- torch 2.6 + cuda ---
pip uninstall -y torch torchvision torchaudio triton nvidia-* || true
pip cache purge || true

pip install --index-url "https://download.pytorch.org/whl/${CUDA_WHL}" \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# --- core stack (PINNED to avoid dependency hell) ---
pip install \
  pytorch-lightning==2.3.3 \
  lightning==2.3.3 \
  torchmetrics==1.3.2 \
  transformers==4.36.2 \
  datasets==2.18.0 \
  huggingface-hub==0.20.3 \
  wandb==0.16.6 \
  numpy==1.26.4 \
  pandas==2.2.3 \
  scipy==1.11.4 \
  scikit-learn==1.3.2 \
  librosa==0.10.1 \
  soundfile==0.12.1 \
  audiomentations==0.38.0 \
  einops==0.8.0 \
  phonemizer==3.2.1 \
  yacs==0.1.8 \
  pillow==10.4.0 \
  matplotlib==3.8.4 \
  seaborn==0.13.2 \
  tqdm==4.66.4 \
  h5py==3.10.0 \
  typing-extensions==4.12.2 \
  ipykernel==6.29.5

# --- sanity checks ---
python - <<'PY'
import torch, transformers, pandas, numpy, pytorch_lightning as pl
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
print("pl:", pl.__version__)
print("transformers:", transformers.__version__)
print("pandas:", pandas.__version__, "numpy:", numpy.__version__)
PY

echo "✅ venv ready: source $VENV/bin/activate"
