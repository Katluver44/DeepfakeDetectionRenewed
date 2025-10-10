#!/bin/bash
# RunPod Setup Script for PLFD Deepfake Detection Training
# This script installs all required dependencies for the training pipeline

set -e  # Exit on error

echo "================================================"
echo "PLFD Training Pipeline - RunPod Setup"
echo "================================================"

# Step 1: Install system dependencies (FFmpeg)
echo ""
echo "[1/4] Installing FFmpeg..."
apt-get update -qq
apt-get install -y ffmpeg espeak-ng libespeak-ng1 > /dev/null 2>&1
echo "✓ FFmpeg and espeak installed"

# Step 2: Install Python packages
echo ""
echo "[2/4] Installing Python packages..."

# Core ML packages (use existing PyTorch if available)
pip install -q --upgrade pip

# Install compatible versions (avoid torchcodec issues)
pip install -q datasets==2.14.0
pip install -q pyarrow==14.0.1
pip install -q huggingface_hub

# Configuration and utilities
pip install -q yacs

# Audio processing
pip install -q librosa==0.10.1
pip install -q phonemizer
pip install -q audio_augmentations==0.1.3
pip install -q einops==0.8.0

# Training framework
pip install -q pytorch_lightning==2.3.3
pip install -q transformers==4.36.2

# Metrics and evaluation
pip install -q scikit-learn
pip install -q torchmetrics
pip install -q scipy==1.15.0

# Utilities
pip install -q pandas==2.2.3
pip install -q rich
pip install -q torch-yin

echo "✓ Python packages installed"

# Step 3: Verify installations
echo ""
echo "[3/4] Verifying installations..."

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchaudio; print(f'TorchAudio: {torchaudio.__version__}')"
python -c "import pytorch_lightning as pl; print(f'PyTorch Lightning: {pl.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets: {datasets.__version__}')"
python -c "import pyarrow; print(f'PyArrow: {pyarrow.__version__}')"

# Check FFmpeg
ffmpeg -version | head -n 1

echo "✓ All verifications passed"

# Step 4: Set up HuggingFace token (if provided)
echo ""
echo "[4/4] Setting up HuggingFace..."

if [ ! -z "$HF_TOKEN" ]; then
    huggingface-cli login --token $HF_TOKEN
    echo "✓ HuggingFace authenticated"
else
    echo "⚠ HF_TOKEN not set - you may need to login manually for private datasets"
    echo "  Use: huggingface-cli login --token YOUR_TOKEN"
fi

# Final summary
echo ""
echo "================================================"
echo "✓ Setup Complete!"
echo "================================================"
echo ""
echo "Installation Summary:"
echo "  - FFmpeg: Installed"
echo "  - Python packages: Installed"
echo "  - datasets: 2.14.0 (compatible version)"
echo "  - pyarrow: 14.0.1 (compatible version)"
echo ""
echo "Next Steps:"
echo "  1. Download phoneme model checkpoint if not present"
echo "  2. Run: python test_pipeline.py"
echo "  3. Run: python train.py --cfg GMM --gpu 0"
echo ""
echo "For full training, edit config.py to increase:"
echo "  - DATASET.train_samples (100 -> -1 for full dataset)"
echo "  - DATASET.val_samples (50 -> -1 for full dataset)"
echo "  - MODEL.epochs (2 -> 20+)"
echo "================================================"

