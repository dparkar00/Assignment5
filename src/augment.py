"""MixUp and CutMix batch-level augmentation, plus the soft-label cross
entropy loss they require.

Unlike RandAugment (a per-image transform applied in src/data.py), MixUp and
CutMix mix pairs of images *within a batch* and blend their labels
accordingly, so they're implemented here as a batch-level operation used
inside the training loop, not as a torchvision transform.

This is applied identically to both the primary (Swin) and baseline (ViT)
models via shared config values in both YAML files -- "data augmentations"
is one of the settings the assignment requires to be the same across both
runs (Part 2, Experimental Controls), so the MixupCutmix hyperparameters
below are read from each config's `data:` section and must match between
configs/primary.yaml and configs/vit_baseline.yaml.
"""

from __future__ import annotations

import dataclasses
import random

import numpy as np
import torch
import torch.nn.functional as F


@dataclasses.dataclass
class MixupCutmixConfig:
    """Hyperparameters for MixUp/CutMix. Mirrors the well-known timm/DeiT
    recipe: per batch, with probability `prob` apply an augmentation at all;
    if applying, pick CutMix vs MixUp via `switch_prob`.
    """

    num_classes: int = 100
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    prob: float = 0.5
    switch_prob: float = 0.5
    label_smoothing: float = 0.1
    enabled: bool = True


def one_hot_with_smoothing(
    targets: torch.Tensor, num_classes: int, smoothing: float
) -> torch.Tensor:
    """Convert integer class labels to smoothed one-hot targets.

    Constructed so that, for a batch MixUp/CutMix leaves untouched (the
    `prob` check fails), the resulting soft-label loss is mathematically
    equivalent to nn.CrossEntropyLoss(label_smoothing=smoothing) on hard
    labels -- label smoothing behavior is preserved, not replaced, once
    MixUp/CutMix is enabled.
    """
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    one_hot = torch.full(
        (targets.size(0), num_classes), off_value, device=targets.device, dtype=torch.float32
    )
    one_hot.scatter_(1, targets.unsqueeze(1), on_value)
    return one_hot


def soft_cross_entropy(outputs: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """Cross entropy against soft (mixed) targets instead of hard labels."""
    log_probs = F.log_softmax(outputs, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def _rand_bbox(height: int, width: int, lam: float) -> tuple[int, int, int, int]:
    """Sample a random bounding box for CutMix, sized so the cut-out area's
    fraction of the image matches (1 - lam).
    """
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h, cut_w = int(height * cut_ratio), int(width * cut_ratio)

    center_y = random.randint(0, height - 1)
    center_x = random.randint(0, width - 1)

    y1 = max(center_y - cut_h // 2, 0)
    y2 = min(center_y + cut_h // 2, height)
    x1 = max(center_x - cut_w // 2, 0)
    x2 = min(center_x + cut_w // 2, width)
    return y1, y2, x1, x2


def _apply_mixup(
    images: torch.Tensor,
    soft_targets: torch.Tensor,
    permuted_soft_targets: torch.Tensor,
    permutation: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MixUp branch: blend the whole image (and label) with a permuted sample."""
    lam = float(np.random.beta(alpha, alpha))
    mixed_images = lam * images + (1.0 - lam) * images[permutation]
    mixed_targets = lam * soft_targets + (1.0 - lam) * permuted_soft_targets
    return mixed_images, mixed_targets


def _apply_cutmix(
    images: torch.Tensor,
    soft_targets: torch.Tensor,
    permuted_soft_targets: torch.Tensor,
    permutation: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CutMix branch: paste a patch from a permuted sample; blend labels by cut area."""
    lam = float(np.random.beta(alpha, alpha))
    height, width = images.shape[-2], images.shape[-1]
    y1, y2, x1, x2 = _rand_bbox(height, width, lam)
    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[permutation][:, :, y1:y2, x1:x2]
    # Recompute lambda from the actual cut area (may differ slightly from the
    # sampled value due to integer rounding / edge clamping).
    actual_lam = 1.0 - ((x2 - x1) * (y2 - y1) / (height * width))
    mixed_targets = actual_lam * soft_targets + (1.0 - actual_lam) * permuted_soft_targets
    return mixed_images, mixed_targets


def apply_mixup_cutmix(
    images: torch.Tensor, targets: torch.Tensor, config: MixupCutmixConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MixUp or CutMix to one training batch.

    Returns (possibly-mixed images, soft targets). If `config.enabled` is
    False, or the per-batch `prob` roll fails, returns the original images
    with smoothed one-hot targets (no mixing) -- this is training-only and
    must never be called on validation/test data, which stays deterministic.
    """
    soft_targets = one_hot_with_smoothing(targets, config.num_classes, config.label_smoothing)

    if not config.enabled or random.random() > config.prob:
        return images, soft_targets

    permutation = torch.randperm(images.size(0), device=images.device)
    permuted_soft_targets = soft_targets[permutation]

    if random.random() < config.switch_prob:
        return _apply_cutmix(
            images, soft_targets, permuted_soft_targets, permutation, config.cutmix_alpha
        )
    return _apply_mixup(
        images, soft_targets, permuted_soft_targets, permutation, config.mixup_alpha
    )
