#!/usr/bin/env python3
"""
Test script to verify data loading and model instantiation.
Quick validation before running full training.
"""
import torch
import pytorch_lightning as pl
from argparse import Namespace

# Set seeds for reproducibility
pl.seed_everything(42)
torch.set_float32_matmul_precision("medium")

print("=" * 80)
print("TESTING PIPELINE SETUP")
print("=" * 80)

# Test 1: Import modules
print("\n[1/5] Testing imports...")
try:
    from config import get_cfg_defaults
    from data.make_dataset import make_data
    from models import make_model
    from callbacks import BinaryACC_Callback, BinaryAUC_Callback, EER_Callback
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test 2: Load config
print("\n[2/5] Testing configuration...")
try:
    cfg = get_cfg_defaults()
    print(f"✓ Config loaded")
    print(f"  - Batch size: {cfg.DATASET.batch_size}")
    print(f"  - Epochs: {cfg.MODEL.epochs}")
    print(f"  - Train samples: {cfg.DATASET.train_samples}")
    print(f"  - Val samples: {cfg.DATASET.val_samples}")
except Exception as e:
    print(f"✗ Config failed: {e}")
    exit(1)

# Test 3: Load data
print("\n[3/5] Testing data loading...")
try:
    args = Namespace(
        test=0,
        test_as_val=999,
        test_noise=0,
    )
    ds, dl = make_data(cfg.DATASET, args=args)
    print(f"✓ Data loaded")
    print(f"  - Train dataset size: {len(ds.train)}")
    print(f"  - Val dataset size: {len(ds.val)}")
    print(f"  - Train batches: {len(dl.train)}")
    print(f"  - Val batches: {len(dl.val)}")
    
    # Get a sample batch
    sample_batch = next(iter(dl.train))
    print(f"  - Sample batch audio shape: {sample_batch['audio'].shape}")
    print(f"  - Sample batch labels shape: {sample_batch['label'].shape}")
    print(f"  - Sample rate: {sample_batch['sample_rate']}")
    
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Create model
print("\n[4/5] Testing model instantiation...")
try:
    model_args = Namespace(profiler=None)
    model = make_model("GMM", cfg, args=model_args)
    print(f"✓ Model created: {model.__class__.__name__}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Forward pass
print("\n[5/5] Testing forward pass...")
try:
    model.eval()
    with torch.no_grad():
        batch_res = model._shared_pred(sample_batch, batch_idx=0, stage="test")
    
    print(f"✓ Forward pass successful")
    print(f"  - Output keys: {list(batch_res.keys())}")
    print(f"  - Logit shape: {batch_res['logit'].shape}")
    print(f"  - Sample predictions: {torch.sigmoid(batch_res['logit'][:3])}")
    
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED - Pipeline is ready!")
print("=" * 80)
print("\nYou can now run training with:")
print("  python train.py --cfg GMM")
print("\nFor GPU training:")
print("  python train.py --cfg GMM --gpu 0")
print("\nFor CPU training:")
print("  python train.py --cfg GMM --gpu -1")
print("=" * 80)


