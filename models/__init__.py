"""Models module for instantiating deepfake detection models."""
from argparse import Namespace
from phoneme_GAT.modules import Phoneme_GAT_lit


def make_model(model_name, cfg, args=None):
    """
    Create and return a model instance.
    
    Args:
        model_name: Name of the model (e.g., "GMM")
        cfg: Configuration object
        args: Additional arguments
    
    Returns:
        model: Instantiated PyTorch Lightning module
    """
    print(f"Creating model: {model_name}")
    
    # For now, we only support Phoneme_GAT
    # In the future, other models can be added here
    
    model = Phoneme_GAT_lit(cfg=cfg, args=args)
    
    print(f"Model created: {model.__class__.__name__}")
    
    return model


