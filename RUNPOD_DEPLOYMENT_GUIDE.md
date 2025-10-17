# RunPod Deployment - Step-by-Step Guide

## Your RunPod Instance

**SSH Connection:**
```bash
ssh root@149.36.1.57 -p 14016 -i ~/.ssh/id_ed25519
```

---

## Step-by-Step Setup

### Step 1: Connect to RunPod

```bash
# From your local terminal
ssh root@149.36.1.57 -p 14016 -i ~/.ssh/id_ed25519
```

### Step 2: Clone Your Repository

```bash
# Once connected to RunPod
cd /workspace
git clone https://github.com/Katluver44/DeepfakeDetectionRenewed.git
cd DeepfakeDetectionRenewed
```

### Step 3: Upload Phoneme Checkpoint

**Option A: Using SCP (from local terminal)**
```bash
# From your Mac (new terminal, not SSH session)
scp -P 14016 -i ~/.ssh/id_ed25519 \
  "/Users/arjunjindal/Desktop/PLFD-ADD/Best Epoch 42 Validation 0.407.ckpt" \
  root@149.36.1.57:/workspace/DeepfakeDetectionRenewed/
```

**Option B: Using RunPod File Browser**
1. Open RunPod web interface
2. Go to File Browser
3. Upload `Best Epoch 42 Validation 0.407.ckpt` to `/workspace/DeepfakeDetectionRenewed/`

**Option C: Download from Google Drive (on RunPod)**
```bash
# On RunPod terminal
cd /workspace/DeepfakeDetectionRenewed
pip install gdown
gdown "https://drive.google.com/uc?id=1SbqynkUQxxlhazklZz9OgcVK7Fl2aT-z"
mv *.ckpt "Best Epoch 42 Validation 0.407.ckpt"
```

### Step 4: Run Setup Script

```bash
# On RunPod terminal
cd /workspace/DeepfakeDetectionRenewed
bash setup_runpod.sh
```

This installs:
- FFmpeg & espeak-ng
- All Python dependencies
- Authenticates with HuggingFace

### Step 5: Verify Checkpoint

```bash
# On RunPod terminal
ls -lh "Best Epoch 42 Validation 0.407.ckpt"
# Should show ~360MB file

ls -lh vocab_phoneme/
# Should show 9 JSON files
```

### Step 6: Test the Setup

```bash
# On RunPod terminal
python test_pipeline.py
```

**Expected output:**
```
[1/5] Testing imports... ✓
[2/5] Testing configuration... ✓
[3/5] Testing data loading... ✓
[4/5] Testing model instantiation... ✓
[5/5] Testing forward pass... ✓
✓ ALL TESTS PASSED
```

### Step 7: Run Training

**Quick Test (2 minutes):**
```bash
python train.py --cfg GMM --gpu 0 --batch_size 4
```

**Full Training:**
```bash
# First, edit config for full dataset
nano config.py
# Change:
#   train_samples = -1  (line 15, was 100)
#   val_samples = -1    (line 16, was 50)
#   epochs = 20         (line 24, was 2)

# Then train
python train.py --cfg GMM --gpu 0 --batch_size 16
```

---

## Alternative: Use Jupyter Notebook

### Step 1: Start Jupyter on RunPod

```bash
# On RunPod terminal
cd /workspace/DeepfakeDetectionRenewed
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### Step 2: Access Jupyter

RunPod will show a URL like:
```
http://127.0.0.1:8888/?token=XXXX
```

Use RunPod's port forwarding or access via RunPod web interface.

### Step 3: Open demo.ipynb

1. Navigate to `demo.ipynb`
2. In Cell 3, set:
   ```python
   USE_REAL_DATA = True  # Use real ASVspoof dataset
   NUM_EPOCHS = 4        # Or more for full training
   BATCH_SIZE = 16       # Larger batch for GPU
   NUM_TRAIN_SAMPLES = 100  # Or -1 for full dataset
   ```
3. Run All Cells

---

## Monitoring Training

### Check GPU Usage

```bash
# On RunPod, in another terminal
watch -n 1 nvidia-smi
```

### View Training Progress

```bash
# Watch metrics in real-time
tail -f model_checkpoints/GMM/version_*/metrics.csv
```

### Check Logs

```bash
# Training logs
cat logs/lightning_logs/version_*/metrics.csv
```

---

## File Transfer Commands

### Upload Checkpoint from Local to RunPod

```bash
# From your Mac terminal
scp -P 14016 -i ~/.ssh/id_ed25519 \
  "/Users/arjunjindal/Desktop/PLFD-ADD/Best Epoch 42 Validation 0.407.ckpt" \
  root@149.36.1.57:/workspace/DeepfakeDetectionRenewed/
```

### Download Results from RunPod to Local

```bash
# Download best checkpoint
scp -P 14016 -i ~/.ssh/id_ed25519 \
  root@149.36.1.57:/workspace/DeepfakeDetectionRenewed/model_checkpoints/GMM/version_*/checkpoints/best-*.ckpt \
  ~/Downloads/

# Download metrics
scp -P 14016 -i ~/.ssh/id_ed25519 \
  root@149.36.1.57:/workspace/DeepfakeDetectionRenewed/model_checkpoints/GMM/version_*/metrics.csv \
  ~/Downloads/
```

---

## Quick Command Cheat Sheet

```bash
# 1. Connect
ssh root@149.36.1.57 -p 14016 -i ~/.ssh/id_ed25519

# 2. Setup
cd /workspace
git clone https://github.com/Katluver44/DeepfakeDetectionRenewed.git
cd DeepfakeDetectionRenewed

# 3. Upload checkpoint (from local terminal)
scp -P 14016 -i ~/.ssh/id_ed25519 \
  "/Users/arjunjindal/Desktop/PLFD-ADD/Best Epoch 42 Validation 0.407.ckpt" \
  root@149.36.1.57:/workspace/DeepfakeDetectionRenewed/

# 4. Install dependencies (on RunPod)
bash setup_runpod.sh

# 5. Test
python test_pipeline.py

# 6. Train
python train.py --cfg GMM --gpu 0 --batch_size 16
```

---

## Expected GPU Performance

### RTX 3090 / 4090
- Batch size: 16-24
- Training time: 4-6 hours (20 epochs, full dataset)
- Cost: ~$2-3

### A100
- Batch size: 32-48
- Training time: 2-3 hours (20 epochs, full dataset)
- Cost: ~$4-6

---

## Troubleshooting

### "Checkpoint not found"
```bash
# Check if uploaded correctly
ls -lh "Best Epoch 42 Validation 0.407.ckpt"

# If missing, use gdown:
pip install gdown
gdown "https://drive.google.com/uc?id=1SbqynkUQxxlhazklZz9OgcVK7Fl2aT-z"
```

### "CUDA out of memory"
```bash
# Reduce batch size
python train.py --cfg GMM --gpu 0 --batch_size 8
```

### "Dataset download too slow"
```bash
# Edit data/make_dataset.py line 121
# Set USE_DUMMY_DATA = True for testing
# Then switch to False for full training
```

---

## Ready to Deploy!

Your repository is ready. Just follow the steps above to:
1. Connect to RunPod
2. Clone repo
3. Upload checkpoint
4. Run setup
5. Start training! 🚀

