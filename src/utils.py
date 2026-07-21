"""Shared training utilities: reproducible seeding, device selection, and
YAML config loading. Used identically by both the Swin and ViT training runs
so that "random seed, when practical" and "device selection" stay consistent
across the two experimental-control-matched runs (Part 2).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Seed every source of randomness Claude's training loop touches.

    cudnn.deterministic=True and cudnn.benchmark=False trade some GPU speed
    for run-to-run reproducibility, which matters more here than raw
    throughput since Part 3 explicitly requires reproducible random seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Pick the best available device: CUDA, then Apple MPS, then CPU.

    Colab GPUs surface as CUDA; this also works unmodified on a local Mac
    (MPS) or a machine with no GPU at all (CPU), so the same training script
    runs everywhere without edits.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_yaml_config(path: str) -> dict[str, Any]:
    """Load a YAML config file (e.g. configs/primary.yaml) into a dict."""
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
