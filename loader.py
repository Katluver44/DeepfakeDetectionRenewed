# loader.py
import os, glob, math, random
from typing import Optional, Callable, Sequence, List, Dict, Any
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
# torchaudio >= 2.9 dropped the legacy backend-selection API (dispatcher is
# automatic now); guard so this module still imports on newer torchaudio.
if hasattr(torchaudio, "set_audio_backend"):
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


def _label_to_int(raw: Any) -> int:
    """Best-effort conversion to {0=real/bonafide, 1=spoof/fake}."""
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("0", "real", "bonafide", "bonafid", "bona", "genuine", "human"):
            return 0
        if "bona" in s or "real" in s or "genuine" in s:
            return 0
        return 1
    return int(raw)


def _make_balanced_indices(labels: List[int], seed: int = 42, limit: Optional[int] = None) -> List[int]:
    """
    Returns a shuffled index list with exact 50/50 class balance (0 vs 1).
    The resulting length is 2*k where k = min(count0, count1), optionally capped by limit.
    If limit is provided, it is interpreted as TOTAL samples desired; exact balance requires even limit.
    """
    idx0 = [i for i, y in enumerate(labels) if int(y) == 0]
    idx1 = [i for i, y in enumerate(labels) if int(y) == 1]

    rng = random.Random(seed)
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    k = min(len(idx0), len(idx1))
    if k == 0:
        raise RuntimeError(f"Cannot balance: class counts are (0={len(idx0)}, 1={len(idx1)})")

    if limit is not None:
        k = min(k, limit // 2)

    balanced = idx0[:k] + idx1[:k]
    rng.shuffle(balanced)
    return balanced


# ======================= datasets =======================
class HFAudioDataset(Dataset):
    """
    Hugging Face dataset wrapper with PLFD-ADD policies.
    - 16 kHz via datasets.Audio
    - train: random 3s crop (self-pad); eval: center 3s crop (self-pad)
    - optional: enforce exact 50/50 real/spoof for train OR eval via indices (balance=True)
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
        balance: bool = True,
    ):
        assert mode in ("train", "eval")
        if not HF_AVAILABLE:
            raise RuntimeError("`datasets` not installed. `pip install datasets soundfile`")

        ds = load_dataset(name, split=split, cache_dir=cache_dir, token=token)
        ds = ds.cast_column("audio", HFAudio(sampling_rate=TARGET_SR))

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

        # Build label list WITHOUT touching audio
        labels: List[int] = []
        for i in range(len(self.ds)):
            raw = self.ds[i][self.label_key]
            labels.append(_label_to_int(raw))

        self.indices = list(range(len(self.ds)))
        s = 42 if seed is None else seed

        # Balance applies to BOTH train and eval when requested.
        if balance:
            self.indices = _make_balanced_indices(labels, seed=s, limit=limit)
        else:
            # preserve original "shuffle before limit" behavior
            if seed is not None:
                rng = random.Random(s)
                rng.shuffle(self.indices)
            if limit is not None:
                self.indices = self.indices[:limit]

        print(f"✓ HF {name}:{split} [{mode}] → {len(self.indices)} samples (balance={balance})")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        ex = self.ds[real_idx]
        a = ex["audio"]
        sr = a["sampling_rate"]
        assert sr == TARGET_SR, f"Expected {TARGET_SR} Hz, got {sr}"
        wave = torch.tensor(a["array"], dtype=torch.float32).unsqueeze(0)  # (1, T)

        wave = _crop_policy(wave, self.mode)

        y = _label_to_int(ex[self.label_key])
        return {"audio": wave, "label": torch.tensor(y).long(), "sample_rate": TARGET_SR}


class CSVAudioDataset(Dataset):
    """
    CSV with at least columns: path,label
    Optional: speaker_id, system_id (propagated for analysis)
    balance=True enforces exact 50/50 via indexing (works for train OR eval).
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
        balance: bool = True,
    ):
        import csv
        assert mode in ("train", "eval")
        rows: List[Dict[str, str]] = []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)

        self.mode = mode
        self.path_col = path_col
        self.label_col = label_col
        self.extra_cols = extra_cols
        s = 42 if seed is None else seed

        labels = [_label_to_int(r[label_col]) for r in rows]

        if balance:
            idx = _make_balanced_indices(labels, seed=s, limit=limit)
            self.rows = [rows[i] for i in idx]
        else:
            if seed is not None:
                random.Random(s).shuffle(rows)
            if limit is not None:
                rows = rows[:limit]
            self.rows = rows

        print(f"✓ CSV {os.path.basename(csv_path)} [{mode}] → {len(self.rows)} samples (balance={balance})")

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

        y = _label_to_int(r[self.label_col])

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
    balance=True enforces exact 50/50 via indexing (works for train OR eval).
    """
    def __init__(
        self,
        root: str,
        mode: str = "train",
        seed: Optional[int] = 42,
        limit: Optional[int] = None,
        label_fn: Optional[Callable[[str], int]] = None,
        balance: bool = True,
    ):
        assert mode in ("train", "eval")
        files = _list_audio_files(root)
        if len(files) == 0:
            raise RuntimeError(f"No audio files found under {root}")

        self.mode = mode
        self.label_fn = label_fn
        self.files = files
        s = 42 if seed is None else seed

        labels = []
        for p in self.files:
            y = int(self.label_fn(p)) if self.label_fn else self._default_label_from_parent(p)
            labels.append(int(y))

        if balance:
            idx = _make_balanced_indices(labels, seed=s, limit=limit)
            self.files = [self.files[i] for i in idx]
        else:
            if seed is not None:
                random.Random(s).shuffle(self.files)
            if limit is not None:
                self.files = self.files[:limit]

        print(f"✓ Folder {root} [{mode}] → {len(self.files)} samples (balance={balance})")

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
    balance: bool = True,
):
    """
    source ∈ {'hf', 'csv', 'folder'}
    balance=True enforces exact 50/50 real/spoof for the dataset (by indexing).
    """
    if source == "hf":
        ds = HFAudioDataset(
            name=hf_name, split="train", cache_dir=hf_cache_dir,
            limit=limit, token=hf_token, mode="train", seed=seed, balance=balance
        )
    elif source == "csv":
        if not csv_path:
            raise ValueError("csv_path required for source='csv'")
        ds = CSVAudioDataset(csv_path, mode="train", seed=seed, limit=limit, balance=balance)
    elif source == "folder":
        if not folder_root:
            raise ValueError("folder_root required for source='folder'")
        ds = FolderAudioDataset(folder_root, mode="train", seed=seed, limit=limit, label_fn=label_fn, balance=balance)
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
    split: str = "val",
    hf_name: Optional[str] = None,
    hf_cache_dir: str = "./data",
    hf_token: Optional[str] = None,
    csv_path: Optional[str] = None,
    folder_root: Optional[str] = None,
    label_fn: Optional[Callable[[str], int]] = None,
    batch_size: int = 4,
    limit: Optional[int] = None,
    seed: Optional[int] = 42,
    num_workers: int = 2,
    balance: bool = True,
):
    """
    source ∈ {'hf','csv','folder'}
    NOTE: by default, we keep TEST unbalanced even if you forget and pass balance=True.
    """
    split_alias = (split or "").lower()
    if split_alias in ("test", "testing"):
        balance = False

    if source == "hf":
        if split_alias in ("val", "validation", "dev"):
            hf_split = "validation"
        elif split_alias in ("test", "testing"):
            hf_split = "test"
        else:
            raise ValueError(f"Unknown split '{split}'. Use one of: val, validation, dev, test.")
        ds = HFAudioDataset(
            name=hf_name, split=hf_split, cache_dir=hf_cache_dir,
            limit=limit, token=hf_token, mode="eval", seed=seed, balance=balance
        )
    elif source == "csv":
        if not csv_path:
            raise ValueError("csv_path required for source='csv'")
        ds = CSVAudioDataset(csv_path, mode="eval", seed=seed, limit=limit, balance=balance)
    elif source == "folder":
        if not folder_root:
            raise ValueError("folder_root required for source='folder'")
        ds = FolderAudioDataset(folder_root, mode="eval", seed=seed, limit=limit, label_fn=label_fn, balance=balance)
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
    - train -> random crop (shuffle=True) + balanced 50/50
    - val/dev -> center crop + balanced 50/50
    - test -> center crop + unbalanced by default
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
            balance=True,
        )
    elif sp in ("val", "validation", "dev"):
        return get_eval_dataloader(
            source="hf",
            split=split,
            hf_name=name,
            hf_cache_dir=cache_dir,
            hf_token=token,
            batch_size=batch_size,
            limit=limit,
            balance=True,
        )
    elif sp in ("test", "testing"):
        return get_eval_dataloader(
            source="hf",
            split=split,
            hf_name=name,
            hf_cache_dir=cache_dir,
            hf_token=token,
            batch_size=batch_size,
            limit=limit,
            balance=False,
        )
    else:
        raise ValueError(f"Unknown split '{split}'. Expected 'train', 'validation'/'val'/'dev', or 'test'.")
