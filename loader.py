# loader.py
import os, glob, math, random
from typing import Optional, Callable, Sequence, List
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

# Optional HF datasets
try:
    from datasets import load_dataset, Audio as HFAudio
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# -------------------------------------------------------------------
# Stable audio backend
# -------------------------------------------------------------------
torchaudio.set_audio_backend("sox_io")

TARGET_SR = 16000
TARGET_SAMPLES = 3 * TARGET_SR  # 48000 samples = 3 seconds
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus")


# ======================= helpers =======================
def _tile_pad_to_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Tile (repeat) along time dim until >= target_len, then trim. x: (1, T)"""
    T = x.shape[1]
    if T <= 0:
        return torch.zeros(1, target_len, dtype=torch.float32)
    if T >= target_len:
        return x[:, :target_len]
    reps = math.ceil(target_len / T)
    return x.repeat(1, reps)[:, :target_len]


def _crop_policy(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Paper policy: train=random 3s; eval=center 3s. Self-pad if short. x: (1, T)"""
    T = x.shape[1]
    if T < TARGET_SAMPLES:
        return _tile_pad_to_length(x, TARGET_SAMPLES)
    if mode == "train":
        start = torch.randint(0, T - TARGET_SAMPLES + 1, (1,)).item()
    else:
        start = (T - TARGET_SAMPLES) // 2
    return x[:, start : start + TARGET_SAMPLES]


def _ensure_sr(wave: torch.Tensor, sr: int) -> torch.Tensor:
    """Resample to 16 kHz if needed. wave: (C, T)"""
    if sr == TARGET_SR:
        return wave
    resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
    return resampler(wave)


def _list_audio_files(root: str) -> List[str]:
    return [
        p for ext in AUDIO_EXTS
        for p in glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
    ]


# ======================= datasets =======================
class HFAudioDataset(Dataset):
    """
    Hugging Face dataset wrapper with PLFD-ADD policies.
    - 16 kHz via datasets.Audio
    - train: random 3s crop (self-pad); eval: center 3s crop (self-pad)
    - one-time shuffle before limit to avoid first-N bias
    """
    def __init__(
        self,
        name: str,
        split: str,
        cache_dir: str = "./data",
        limit: Optional[int] = None,
        token: Optional[str] = None,
        mode: str = "train",
        seed: Optional[int] = 42,
        label_key_fallbacks: Sequence[str] = ("label", "key"),
    ):
        assert mode in ("train", "eval")
        if not HF_AVAILABLE:
            raise RuntimeError("`datasets` not installed. `pip install datasets soundfile`")

        ds = load_dataset(name, split=split, cache_dir=cache_dir, token=token)
        ds = ds.cast_column("audio", HFAudio(sampling_rate=TARGET_SR))

        if seed is not None:
            ds = ds.shuffle(seed=seed)  # avoid first-N bias
        if limit is not None:
            ds = ds.select(range(limit))

        self.ds = ds
        self.mode = mode

        # Auto-detect label column
        example = self.ds[0]
        lk = None
        for k in example.keys():
            kl = k.lower()
            if ("label" in kl) or (kl == "key") or ("key" in kl):
                lk = k
                break
        self.label_key = lk or "label"

        print(f"✓ HF {name}:{split} [{mode}] → {len(self.ds)} samples")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        a = ex["audio"]
        sr = a["sampling_rate"]
        assert sr == TARGET_SR, f"Expected {TARGET_SR} Hz, got {sr}"
        wave = torch.tensor(a["array"], dtype=torch.float32).unsqueeze(0)  # (1, T)

        wave = _crop_policy(wave, self.mode)

        raw = ex[self.label_key]
        if isinstance(raw, str):
            y = 0 if "bona" in raw.lower() else 1
        else:
            y = int(raw)

        return {"audio": wave, "label": torch.tensor(y).long(), "sample_rate": TARGET_SR}


class CSVAudioDataset(Dataset):
    """
    CSV with at least columns: path,label
    Optional: speaker_id, system_id (propagated for analysis)
    """
    def __init__(
        self,
        csv_path: str,
        mode: str = "train",
        seed: Optional[int] = 42,
        limit: Optional[int] = None,
        path_col: str = "path",
        label_col: str = "label",
        extra_cols: Sequence[str] = ("speaker_id", "system_id"),
    ):
        import csv
        assert mode in ("train", "eval")
        rows = []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
        if seed is not None:
            random.Random(seed).shuffle(rows)
        if limit is not None:
            rows = rows[:limit]
        self.rows = rows
        self.mode = mode
        self.path_col = path_col
        self.label_col = label_col
        self.extra_cols = extra_cols
        print(f"✓ CSV {os.path.basename(csv_path)} [{mode}] → {len(self.rows)} samples")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        path = r[self.path_col]
        wave, sr = torchaudio.load(path)  # (C, T)
        wave = _ensure_sr(wave, sr)
        if wave.shape[0] > 1:
            wave = wave.mean(dim=0, keepdim=True)  # to mono
        wave = _crop_policy(wave, self.mode)

        raw = r[self.label_col]
        if isinstance(raw, str):
            y = 0 if raw.lower() in ("0", "real", "bonafide", "bona", "genuine") else 1
        else:
            y = int(raw)

        out = {"audio": wave, "label": torch.tensor(y).long(), "sample_rate": TARGET_SR}
        for k in self.extra_cols:
            if k in r:
                out[k] = r[k]
        return out


class FolderAudioDataset(Dataset):
    """
    Recursively loads audio from a root folder.
    Default labeling by parent folder:
      'bonafide'/'real' -> 0, 'spoof'/'fake' -> 1, else 1
    Or pass label_fn(path)->int.
    """
    def __init__(
        self,
        root: str,
        mode: str = "train",
        seed: Optional[int] = 42,
        limit: Optional[int] = None,
        label_fn: Optional[Callable[[str], int]] = None,
    ):
        assert mode in ("train", "eval")
        files = _list_audio_files(root)
        if len(files) == 0:
            raise RuntimeError(f"No audio files found under {root}")
        if seed is not None:
            random.Random(seed).shuffle(files)
        if limit is not None:
            files = files[:limit]
        self.files = files
        self.mode = mode
        self.label_fn = label_fn
        print(f"✓ Folder {root} [{mode}] → {len(self.files)} samples")

    @staticmethod
    def _default_label_from_parent(path: str) -> int:
        parent = os.path.basename(os.path.dirname(path)).lower()
        if "bona" in parent or parent == "real":
            return 0
        if "spoof" in parent or parent == "fake":
            return 1
        return 1  # conservative default

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        wave, sr = torchaudio.load(path)
        wave = _ensure_sr(wave, sr)
        if wave.shape[0] > 1:
            wave = wave.mean(dim=0, keepdim=True)
        wave = _crop_policy(wave, self.mode)

        y = int(self.label_fn(path)) if self.label_fn else self._default_label_from_parent(path)
        return {"audio": wave, "label": torch.tensor(y).long(), "sample_rate": TARGET_SR, "path": path}


# ==================== dataloader factories ====================
def get_train_dataloader(
    source: str,
    *,
    # HF
    hf_name: Optional[str] = None,
    hf_cache_dir: str = "./data",
    hf_token: Optional[str] = None,
    # CSV
    csv_path: Optional[str] = None,
    # Folder
    folder_root: Optional[str] = None,
    label_fn: Optional[Callable[[str], int]] = None,
    # common
    batch_size: int = 4,
    limit: Optional[int] = None,
    seed: Optional[int] = 42,
    num_workers: int = 2,
    drop_last: bool = False,
):
    """
    source ∈ {'hf', 'csv', 'folder'}
    """
    if source == "hf":
        ds = HFAudioDataset(
            name=hf_name, split="train", cache_dir=hf_cache_dir,
            limit=limit, token=hf_token, mode="train", seed=seed
        )
    elif source == "csv":
        if not csv_path:
            raise ValueError("csv_path required for source='csv'")
        ds = CSVAudioDataset(csv_path, mode="train", seed=seed, limit=limit)
    elif source == "folder":
        if not folder_root:
            raise ValueError("folder_root required for source='folder'")
        ds = FolderAudioDataset(folder_root, mode="train", seed=seed, limit=limit, label_fn=label_fn)
    else:
        raise ValueError("source must be one of {'hf','csv','folder'}")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def get_eval_dataloader(
    source: str,
    *,
    split: str = "val",  # accepts: 'val'|'validation'|'dev'|'test'
    # HF
    hf_name: Optional[str] = None,
    hf_cache_dir: str = "./data",
    hf_token: Optional[str] = None,
    # CSV
    csv_path: Optional[str] = None,
    # Folder
    folder_root: Optional[str] = None,
    label_fn: Optional[Callable[[str], int]] = None,
    # common
    batch_size: int = 4,
    limit: Optional[int] = None,
    seed: Optional[int] = 42,
    num_workers: int = 2,
):
    """
    source ∈ {'hf','csv','folder'}
    For CSV/Folder eval, call twice if you want separate val/test roots/CSVs.
    """
    if source == "hf":
        split_alias = (split or "").lower()
        if split_alias in ("val", "validation", "dev"):
            hf_split = "validation"
        elif split_alias in ("test", "testing"):
            hf_split = "test"
        else:
            raise ValueError(f"Unknown split '{split}'. Use one of: val, validation, dev, test.")
        ds = HFAudioDataset(
            name=hf_name, split=hf_split, cache_dir=hf_cache_dir,
            limit=limit, token=hf_token, mode="eval", seed=seed
        )
    elif source == "csv":
        if not csv_path:
            raise ValueError("csv_path required for source='csv'")
        ds = CSVAudioDataset(csv_path, mode="eval", seed=seed, limit=limit)
    elif source == "folder":
        if not folder_root:
            raise ValueError("folder_root required for source='folder'")
        ds = FolderAudioDataset(folder_root, mode="eval", seed=seed, limit=limit, label_fn=label_fn)
    else:
        raise ValueError("source must be one of {'hf','csv','folder'}")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


# ============ Backward compatibility (old API) ============
def get_dataloader(
    name: str,
    split: str,
    batch_size: int = 4,
    cache_dir: str = "./data",
    limit: Optional[int] = None,
    token: Optional[str] = None,
):
    """
    Legacy helper to keep old notebooks working.
    - If split == 'train'  -> uses train policy (random crop, shuffle=True)
    - If split in {'val','validation','dev','test'} -> eval policy (center crop)
    """
    sp = (split or "").lower()
    if sp == "train":
        return get_train_dataloader(
            source="hf",
            hf_name=name,
            hf_cache_dir=cache_dir,
            hf_token=token,
            batch_size=batch_size,
            limit=limit,
        )
    elif sp in ("val", "validation", "dev", "test", "testing"):
        return get_eval_dataloader(
            source="hf",
            split=split,
            hf_name=name,
            hf_cache_dir=cache_dir,
            hf_token=token,
            batch_size=batch_size,
            limit=limit,
        )
    else:
        raise ValueError(f"Unknown split '{split}'. Expected 'train', 'validation'/'val'/'dev', or 'test'.")
