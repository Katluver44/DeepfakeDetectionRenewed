# loader.py
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio
import torchaudio

# -------------------------------------------------------------------
# Safe audio backend (prevents TorchCodec conflicts)
# -------------------------------------------------------------------
torchaudio.set_audio_backend("sox_io")


class HuggingFaceAudioDataset(Dataset):
    """
    Generic Hugging Face dataset wrapper.
    Loads audio and labels, ensures uniform (1,48000) tensors @16 kHz.
    """

    def __init__(self, name: str, split: str, cache_dir: str = "./data", limit: int = None, token: str = None):
        self.dataset = load_dataset(name, split=split, cache_dir=cache_dir, token=token)
        # Force decoding and resampling to 16 kHz
        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=16000))
        if limit:
            self.dataset = self.dataset.select(range(limit))

        # auto-detect label column
        example = self.dataset[0]
        for k in example.keys():
            if "label" in k.lower() or "key" in k.lower():
                self.label_key = k
                break
        else:
            self.label_key = "label"

        print(f"✓ Loaded {len(self.dataset)} samples from {name}:{split}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        audio_data = sample["audio"]
        waveform = torch.tensor(audio_data["array"], dtype=torch.float32).unsqueeze(0)
        sr = audio_data["sampling_rate"]

        # pad/trim to 3 s → 48000 samples
        target_len = 48000
        if waveform.shape[1] < target_len:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[1]))
        elif waveform.shape[1] > target_len:
            waveform = waveform[:, :target_len]

        label_raw = sample[self.label_key]
        if isinstance(label_raw, str):
            label = 0 if "bona" in label_raw.lower() else 1
        else:
            label = int(label_raw)

        return {
            "audio": waveform,
            "label": torch.tensor(label, dtype=torch.long),
            "sample_rate": 16000,
        }


def get_dataloader(name: str, split: str, batch_size: int = 4, cache_dir: str = "./data", limit: int = None, token: str = None):
    ds = HuggingFaceAudioDataset(name, split, cache_dir=cache_dir, limit=limit, token=token)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)