"""Configuration module using YACS."""
from yacs.config import CfgNode as CN

_C = CN()

# Dataset configuration
_C.DATASET = CN()
_C.DATASET.name = "asvspoof2019"
_C.DATASET.batch_size = 4
_C.DATASET.num_workers = 4
_C.DATASET.train_samples = 100  # Small subset for quick testing
_C.DATASET.val_samples = 50
_C.DATASET.sample_rate = 16000
_C.DATASET.audio_length = 48000  # 3 seconds at 16kHz

# Model configuration
_C.MODEL = CN()
_C.MODEL.epochs = 2
_C.MODEL.learning_rate = 1e-4

# Phoneme GAT specific configuration
_C.PhonemeGAT = CN()
_C.PhonemeGAT.backbone = "wavlm"
_C.PhonemeGAT.use_raw = False
_C.PhonemeGAT.use_GAT = True
_C.PhonemeGAT.n_edges = 10
_C.PhonemeGAT.use_aug = True
_C.PhonemeGAT.use_pool = True
_C.PhonemeGAT.use_clip = True

# Checkpoint path
_C.CHECKPOINT = CN()
_C.CHECKPOINT.phoneme_model = "Best Epoch 42 Validation 0.407.ckpt"


def get_cfg_defaults(config_file=None, ablation=None):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    
    if config_file is not None:
        # Load from YAML file if provided (optional for now)
        try:
            config.merge_from_file(config_file)
        except FileNotFoundError:
            print(f"Config file {config_file} not found, using defaults")
    
    config.freeze()
    return config

