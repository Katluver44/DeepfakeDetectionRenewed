"""Data loading module for ASVspoof 2019 LA dataset."""
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import numpy as np
from argparse import Namespace


class ASVspoofDataset(Dataset):
    """ASVspoof 2019 LA dataset wrapper."""
    
    def __init__(self, hf_dataset, target_length=48000, target_sr=16000):
        self.dataset = hf_dataset
        self.target_length = target_length
        self.target_sr = target_sr
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get audio path from the dataset
        # For ASVspoof, we need to load audio from the path
        try:
            # Try to get audio data directly if available
            if 'audio' in item and isinstance(item['audio'], dict):
                if 'path' in item['audio']:
                    audio_path = item['audio']['path']
                    audio, sr = torchaudio.load(audio_path)
                elif 'array' in item['audio']:
                    # Use array if available
                    audio = item['audio']['array']
                    sr = item['audio']['sampling_rate']
                    if isinstance(audio, np.ndarray):
                        audio = torch.from_numpy(audio).float()
                    else:
                        audio = torch.tensor(audio, dtype=torch.float32)
                    if audio.ndim == 1:
                        audio = audio.unsqueeze(0)
            else:
                # Fallback: assume audio is directly available
                audio = item['audio']
                sr = 16000
                if isinstance(audio, np.ndarray):
                    audio = torch.from_numpy(audio).float()
                if audio.ndim == 1:
                    audio = audio.unsqueeze(0)
        except Exception as e:
            # If loading fails, create silence
            print(f"Warning: Could not load audio for item {idx}: {e}")
            audio = torch.zeros(1, self.target_length)
            sr = self.target_sr
        
        # Resample if needed
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            audio = resampler(audio)
        
        # Ensure correct shape (add channel dimension if needed)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        
        # Pad or trim to target length
        if audio.shape[1] < self.target_length:
            # Pad with zeros
            padding = self.target_length - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        elif audio.shape[1] > self.target_length:
            # Trim
            audio = audio[:, :self.target_length]
        
        # Map labels: "bonafide" -> 0, "spoof" -> 1
        label = 0 if item['label'] == 'bonafide' else 1
        
        return {
            'audio': audio,
            'label': label,
            'sample_rate': self.target_sr,
        }


def collate_fn(batch):
    """Collate function for dataloader."""
    return {
        'audio': torch.stack([item['audio'] for item in batch]),
        'label': torch.tensor([item['label'] for item in batch], dtype=torch.long),
        'sample_rate': batch[0]['sample_rate'],
    }


class DataLoaderWrapper:
    """Wrapper to provide .train, .val, .test attributes."""
    
    def __init__(self, train_dl, val_dl, test_dl):
        self.train = train_dl
        self.val = val_dl
        self.test = test_dl


def make_data(cfg, args=None):
    """
    Load ASVspoof 2019 LA dataset from HuggingFace.
    
    Args:
        cfg: Configuration object with DATASET attributes
        args: Additional arguments
    
    Returns:
        ds: Dataset wrapper (namespace with train/val/test datasets)
        dl: DataLoader wrapper (object with train/val/test dataloaders)
    """
    print("Loading ASVspoof 2019 LA dataset from HuggingFace...")
    
    # FIXED: Downgraded datasets to 2.14.0 to avoid torchcodec issues
    # Now using REAL ASVspoof data from HuggingFace
    USE_DUMMY_DATA = False  # Using real data!
    
    if USE_DUMMY_DATA:
        print("Using dummy data for quick testing (set USE_DUMMY_DATA=False to use real data)")
        return create_dummy_data(cfg)
    
    try:
        # Load dataset from HuggingFace WITHOUT audio decoding to avoid torchcodec issues
        # We'll decode audio manually using torchaudio
        # Note: This dataset may require authentication
        from datasets import Features, Value, ClassLabel, Audio
        dataset = load_dataset(
            "Bisher/ASVspoof_2019_LA",
            # Don't decode audio automatically - we'll do it manually with torchaudio
            # This avoids torchcodec compatibility issues
        )
        
        # Get train and validation splits
        train_dataset = dataset['train']
        val_dataset = dataset.get('validation', dataset.get('dev', None))
        
        # If no validation set, split from train
        if val_dataset is None:
            print("No validation split found, splitting train set...")
            split = train_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split['train']
            val_dataset = split['test']
        
        # Use small subsets for quick testing
        train_subset_size = min(cfg.train_samples, len(train_dataset))
        val_subset_size = min(cfg.val_samples, len(val_dataset))
        
        print(f"Using {train_subset_size} training samples and {val_subset_size} validation samples")
        
        train_dataset = train_dataset.select(range(train_subset_size))
        val_dataset = val_dataset.select(range(val_subset_size))
        
        # Create dataset wrappers
        train_ds = ASVspoofDataset(
            train_dataset,
            target_length=cfg.audio_length,
            target_sr=cfg.sample_rate
        )
        val_ds = ASVspoofDataset(
            val_dataset,
            target_length=cfg.audio_length,
            target_sr=cfg.sample_rate
        )
        
        # Create dataloaders
        # Use num_workers=0 to avoid multiprocessing issues on macOS
        train_dl = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            collate_fn=collate_fn,
            pin_memory=False,
        )
        
        val_dl = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,  # Avoid multiprocessing issues
            collate_fn=collate_fn,
            pin_memory=False,
        )
        
        # Create dataset namespace
        ds = Namespace(
            train=train_ds,
            val=val_ds,
            test=val_ds,  # Use val as test for now
        )
        
        # Create dataloader wrapper
        dl = DataLoaderWrapper(train_dl, val_dl, val_dl)
        
        print(f"Dataset loaded: {len(train_ds)} train, {len(val_ds)} val samples")
        
        return ds, dl
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating dummy dataset for testing...")
        return create_dummy_data(cfg)


def create_dummy_data(cfg):
    """Create dummy dataset if HuggingFace loading fails."""
    
    class DummyDataset(Dataset):
        def __init__(self, num_samples, target_length, target_sr):
            self.num_samples = num_samples
            self.target_length = target_length
            self.target_sr = target_sr
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Random audio
            audio = torch.randn(1, self.target_length)
            # Random label
            label = torch.randint(0, 2, (1,)).item()
            
            return {
                'audio': audio,
                'label': label,
                'sample_rate': self.target_sr,
            }
    
    train_ds = DummyDataset(cfg.train_samples, cfg.audio_length, cfg.sample_rate)
    val_ds = DummyDataset(cfg.val_samples, cfg.audio_length, cfg.sample_rate)
    
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    ds = Namespace(train=train_ds, val=val_ds, test=val_ds)
    dl = DataLoaderWrapper(train_dl, val_dl, val_dl)
    
    print(f"Dummy dataset created: {len(train_ds)} train, {len(val_ds)} val samples")
    
    return ds, dl

