"""Wrapper to run scripts/generate_laugh_bank.py under torch>=2.6 where
torch.load defaults to weights_only=True, which breaks Bark's own
(numpy-pickled) checkpoints. Patch the default before Bark loads anything.
"""
import functools
import sys
from pathlib import Path

import torch

_orig_load = torch.load


@functools.wraps(_orig_load)
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)


torch.load = _patched_load

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.argv = ["generate_laugh_bank.py"] + sys.argv[1:]

import runpy
runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "generate_laugh_bank.py"), run_name="__main__")
