# Installation Summary - PLFD Training Pipeline

## ✅ For RunPod (Cloud GPU Training)

### One-Command Setup

Upload your code to RunPod, then run:

```bash
cd /workspace/PLFD-ADD
HF_TOKEN=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 bash setup_runpod.sh
```

This installs everything automatically!

### Manual Installation (if needed)

```bash
# 1. System dependencies
apt-get update
apt-get install -y ffmpeg espeak-ng libespeak-ng1

# 2. Python packages (exact versions that work)
pip install datasets==2.14.0 pyarrow==14.0.1
pip install -r requirements_runpod.txt

# 3. HuggingFace login
huggingface-cli login --token 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
```

### Test Installation

```bash
python test_pipeline.py
# Should show: ✓ ALL TESTS PASSED
```

### Start Training

```bash
# Quick test (2 epochs, ~2 minutes)
python train.py --cfg GMM --gpu 0

# Full training (edit config.py first)
python train.py --cfg GMM --gpu 0 --batch_size 16
```

## 📄 RunPod Files Included

- **setup_runpod.sh** - Automated setup script
- **requirements_runpod.txt** - Python dependencies
- **RUNPOD_SETUP.md** - Complete RunPod guide

## 🔑 Critical Dependencies

### Exact Versions (Avoid Compatibility Issues)

```
datasets==2.14.0        # NOT latest (torchcodec issues)
pyarrow==14.0.1         # NOT latest (compatibility)
pytorch_lightning==2.3.3
transformers==4.36.2
```

### System Packages

```
ffmpeg              # Audio processing
espeak-ng           # Phonemizer backend
libespeak-ng1       # Phonemizer library
```

## 📦 Required Files

Upload these to RunPod:

1. **All code files** (train.py, config.py, etc.)
2. **Phoneme model**: `Best Epoch 42 Validation 0.407.ckpt` (~360MB)
3. **Vocabularies**: `vocab_phoneme/` folder (9 JSON files)

## 🚀 Quick Start Checklist

```
[ ] Upload code to RunPod
[ ] Upload phoneme checkpoint
[ ] Run setup_runpod.sh
[ ] Test with test_pipeline.py
[ ] Start training!
```

## 💡 Pro Tips

1. **Use PyTorch template** on RunPod (has PyTorch pre-installed)
2. **Start with RTX 3090** (good price/performance)
3. **Test first** with quick training (2 epochs)
4. **Monitor GPU** with `watch -n 1 nvidia-smi`
5. **Save checkpoints** regularly to avoid data loss

## 🐛 Troubleshooting

**Problem:** "torchcodec error"  
**Solution:** Use `datasets==2.14.0` (already in setup)

**Problem:** "CUDA out of memory"  
**Solution:** Reduce batch size: `--batch_size 8`

**Problem:** "Checkpoint not found"  
**Solution:** Upload `Best Epoch 42 Validation 0.407.ckpt` to project root

**Problem:** "FFmpeg not found"  
**Solution:** Run `apt-get install -y ffmpeg`

## 📊 Expected Results

### Quick Test (2 epochs, dummy data)
- Time: ~2 minutes on RTX 3090
- Cost: ~$0.02
- Purpose: Verify pipeline works

### Full Training (20 epochs, real data)
- Time: 4-8 hours on RTX 3090
- Cost: ~$2-4
- Purpose: Train production model

## 📖 Documentation

- **RUNPOD_SETUP.md** - Complete RunPod guide
- **FINAL_STATUS.md** - Local setup status
- **QUICK_START.md** - Quick reference
- **DEPENDENCIES_STATUS.md** - Dependency details

---

## Ready for RunPod! 🎯

Everything is configured for cloud GPU training. Just upload and run!

```bash
# On RunPod terminal:
cd /workspace/PLFD-ADD
bash setup_runpod.sh
python train.py --cfg GMM --gpu 0
```

