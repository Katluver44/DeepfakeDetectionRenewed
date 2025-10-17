# RunPod Setup Guide - PLFD Training Pipeline

## Quick Start on RunPod

### Step 1: Launch RunPod Instance

1. Go to [RunPod.io](https://runpod.io)
2. Select a GPU pod (RTX 3090, RTX 4090, or A100 recommended)
3. Use PyTorch template (pre-installed PyTorch)
4. Start the pod

### Step 2: Upload Code & Checkpoint

**Option A: Using Git (Recommended)**
```bash
cd /workspace
git clone YOUR_REPO_URL PLFD-ADD
cd PLFD-ADD
```

**Option B: Using RunPod Upload**
1. Zip your local folder: `PLFD-ADD.zip`
2. Upload via RunPod file browser
3. Extract: `unzip PLFD-ADD.zip`

**Important Files to Upload:**
- ✅ All code files
- ✅ `Best Epoch 42 Validation 0.407.ckpt` (phoneme model)
- ✅ `vocab_phoneme/` directory (all 9 JSON files)

### Step 3: Install Dependencies

**Method 1: Using Setup Script (Recommended)**
```bash
cd /workspace/PLFD-ADD
chmod +x setup_runpod.sh
HF_TOKEN=your_token_here bash setup_runpod.sh
```

**Method 2: Manual Installation**
```bash
cd /workspace/PLFD-ADD

# Install system dependencies
apt-get update
apt-get install -y ffmpeg espeak-ng libespeak-ng1

# Install Python packages
pip install -r requirements_runpod.txt

# Login to HuggingFace (for dataset access)
huggingface-cli login --token $HF_TOKEN
```

### Step 4: Test the Setup

```bash
# Quick test
python test_pipeline.py

# Expected output: All tests pass ✓
```

### Step 5: Run Training

**Quick Test (2 epochs, minimal steps)**
```bash
python train.py --cfg GMM --gpu 0 --batch_size 4
```

**Full Training**
```bash
# First, edit config.py to use full dataset:
# Line 15: train_samples = -1 (was 100)
# Line 16: val_samples = -1 (was 50)  
# Line 24: epochs = 20 (was 2)

# Then run:
python train.py --cfg GMM --gpu 0 --batch_size 16
```

## Environment Details

### Pre-installed on RunPod (PyTorch Template)
- ✅ PyTorch 2.x
- ✅ TorchAudio
- ✅ TorchVision
- ✅ CUDA 11.8 or 12.1
- ✅ Python 3.9-3.11

### Installed by Setup Script
- FFmpeg (audio processing)
- espeak-ng (phonemizer backend)
- datasets 2.14.0 (HuggingFace)
- pyarrow 14.0.1 (data loading)
- pytorch_lightning 2.3.3
- All other dependencies

## Training Configuration

### Default Settings (Quick Test)
```python
batch_size = 4
epochs = 2
train_samples = 100
val_samples = 50
limit_train_batches = 10
limit_val_batches = 5
# Total: 20 training steps (~2 minutes on RTX 3090)
```

### Recommended Settings (Full Training)
```python
batch_size = 16-32  # Adjust based on GPU memory
epochs = 20-50
train_samples = -1  # Full dataset (25,380 samples)
val_samples = -1    # Full dataset (24,844 samples)
# Estimated time: 4-8 hours on RTX 3090
```

## GPU Memory Requirements

| GPU | Batch Size | Notes |
|-----|------------|-------|
| RTX 3090 (24GB) | 16-24 | Recommended |
| RTX 4090 (24GB) | 16-32 | Best performance |
| A100 (40GB) | 32-48 | Production use |
| RTX 3080 (10GB) | 8-12 | Minimal |

If you get OOM errors, reduce batch size.

## Monitoring Training

### View Metrics in Real-time
```bash
# In a separate terminal
tail -f model_checkpoints/GMM/version_X/metrics.csv
```

### Check GPU Usage
```bash
nvidia-smi -l 1
```

### TensorBoard (Optional)
```bash
tensorboard --logdir=model_checkpoints/GMM --port=6006
# Then access via RunPod's port forwarding
```

## Common Issues & Solutions

### Issue 1: "CUDA out of memory"
**Solution:** Reduce batch size
```bash
python train.py --cfg GMM --gpu 0 --batch_size 8
```

### Issue 2: "Dataset not found"
**Solution:** Login to HuggingFace
```bash
huggingface-cli login --token $HF_TOKEN
```

### Issue 3: "FFmpeg not found"
**Solution:** Install FFmpeg
```bash
apt-get update && apt-get install -y ffmpeg
```

### Issue 4: "Checkpoint not found"
**Solution:** Ensure phoneme model is in project root
```bash
ls -lh "Best Epoch 42 Validation 0.407.ckpt"
# Should show ~360MB file
```

## Save Results

### Download Checkpoints
```bash
# Best checkpoint location:
model_checkpoints/GMM/version_X/checkpoints/best-epoch=X-val-auc=X.XXXX.ckpt

# Download via RunPod UI or:
zip -r results.zip model_checkpoints/GMM/
```

### Download Metrics
```bash
# CSV with all metrics:
model_checkpoints/GMM/version_X/metrics.csv
```

## Cost Estimation

| GPU | Price/hr | Quick Test | Full Training |
|-----|----------|------------|---------------|
| RTX 3090 | $0.40 | $0.02 | $2-4 |
| RTX 4090 | $0.80 | $0.03 | $4-6 |
| A100 40GB | $1.89 | $0.06 | $8-12 |

*Prices are approximate and may vary*

## Quick Command Reference

```bash
# Setup
chmod +x setup_runpod.sh && bash setup_runpod.sh

# Test
python test_pipeline.py

# Train (quick)
python train.py --cfg GMM --gpu 0

# Train (full)
python train.py --cfg GMM --gpu 0 --batch_size 16

# Monitor
tail -f model_checkpoints/GMM/version_*/metrics.csv

# GPU stats
watch -n 1 nvidia-smi
```

## Support Files

- **setup_runpod.sh** - Automated setup script
- **requirements_runpod.txt** - Python dependencies
- **test_pipeline.py** - Validation script
- **FINAL_STATUS.md** - Complete documentation

---

**Ready to Train!** 🚀

Once setup is complete, you can train the model on powerful GPUs with:
```bash
python train.py --cfg GMM --gpu 0 --batch_size 16
```

