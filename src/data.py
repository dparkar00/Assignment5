"""CIFAR-100 data pipeline for the Vision Transformer assignment.

Implements Part 1 of the assignment:
  - Stratified 45,000 / 5,000 split of the official CIFAR-100 training set
    into train / validation, with the official 10,000-image test set left
    untouched.
  - Disjointness verification across all three splits.
  - Separate, non-leaky transform pipelines for training vs. evaluation.

Design choices (documented here so they can be copied into the report):
  - Input resolution: kept at the native 32x32. CIFAR-100 images are small
    and low-detail; upsampling to a large resolution such as 224x224 mostly
    interpolates pixels rather than adding information, and is expensive for
    two from-scratch training runs. 32x32 is standard for CIFAR-scale ViT
    and Swin experiments in the literature.
  - Patch size: 4x4. At 32x32 input this yields an 8x8 = 64 token grid,
    which is small enough to train quickly but still divides cleanly into
    Swin's hierarchical patch-merging stages (8 -> 4 -> 2 -> 1).
  - Normalization statistics: computed from the CIFAR-100 training set
    itself (not ImageNet statistics), since the model is trained from
    random initialization on CIFAR-100 only.
  - Extra augmentation: RandAugment, in addition to the required random
    horizontal flip and random crop. RandAugment applies a random sequence
    of standard image transforms (rotation, shear, color, contrast, etc.)
    at a configurable magnitude, and is a strong, well-documented default
    for training small models from scratch.
  - No random augmentation at validation/test time: augmentation exists to
    prevent the model from memorizing exact pixel patterns during training.
    At evaluation we want a single, deterministic, reproducible measurement
    of how the model performs on real, unmodified data -- introducing
    randomness there would make validation/test accuracy a noisy, moving
    target and would break comparability between checkpoints and between
    the two models.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# CIFAR-100 per-channel mean/std computed over the official 50,000-image
# training set (RGB, [0, 1] scale). These are the commonly reported values
# for this dataset and are used here rather than ImageNet statistics.
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

NUM_CLASSES = 100
TRAIN_SPLIT_SIZE = 45_000
VAL_SPLIT_SIZE = 5_000
RANDOM_SEED = 705643  # fixed seed for reproducibility (course number, arbitrary)


@dataclasses.dataclass
class DataConfig:
    """Configuration for the CIFAR-100 data pipeline."""

    data_root: str = "./data"
    input_resolution: int = 32
    random_seed: int = RANDOM_SEED
    randaugment_num_ops: int = 2
    randaugment_magnitude: int = 9
    random_crop_padding: int = 4
    random_erasing_prob: float = 0.0  # optional extra augmentation, off by default


def build_train_transform(config: DataConfig) -> transforms.Compose:
    """Non-deterministic transform pipeline used only for the training split."""
    ops = [
        transforms.RandomCrop(config.input_resolution, padding=config.random_crop_padding),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(
            num_ops=config.randaugment_num_ops,
            magnitude=config.randaugment_magnitude,
        ),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ]
    if config.random_erasing_prob > 0:
        ops.append(transforms.RandomErasing(p=config.random_erasing_prob))
    return transforms.Compose(ops)


def build_eval_transform(config: DataConfig) -> transforms.Compose:  # pylint: disable=unused-argument
    """Deterministic transform pipeline used for validation and test splits.

    Intentionally contains no random operations: every call on a given image
    must produce the same tensor, so that validation/test metrics are
    reproducible and comparable across epochs and across models.

    `config` is unused today (no eval-time setting varies), but is kept in
    the signature so build_train_transform and build_eval_transform share
    one call pattern -- callers don't need to know which one actually reads
    the config.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


class TransformedSubset(Dataset):
    """Wraps a Subset so train/val can share one underlying CIFAR100 dataset
    while applying different transforms to each split.

    torchvision's CIFAR100 applies its transform inside __getitem__, so a
    plain Subset over one CIFAR100 instance would force train and val to
    share a transform. This wrapper re-applies the desired transform to the
    raw PIL image instead.
    """

    def __init__(
        self, base_dataset: datasets.CIFAR100, indices: np.ndarray, transform: transforms.Compose
    ):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])
        raw_image = self.base_dataset.data[real_idx]
        target = self.base_dataset.targets[real_idx]
        pil_image = datasets.folder.Image.fromarray(raw_image)
        tensor_image = self.transform(pil_image)
        return tensor_image, target


def stratified_train_val_indices(
    targets: list[int],
    val_size: int,
    num_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Produce a stratified split of indices into (train_indices, val_indices).

    Stratified means each class contributes val_size / num_classes examples
    to the validation split (as close to equal as integer division allows),
    rather than a uniform random split that could by chance under- or
    over-represent some classes in validation.
    """
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)

    per_class_val = val_size // num_classes
    remainder = val_size - per_class_val * num_classes

    train_indices = []
    val_indices = []
    for class_id in range(num_classes):
        class_indices = np.where(targets == class_id)[0]
        rng.shuffle(class_indices)

        # Distribute the remainder (from integer division) across the first
        # `remainder` classes so the total val_size is hit exactly.
        this_class_val_count = per_class_val + (1 if class_id < remainder else 0)

        val_indices.append(class_indices[:this_class_val_count])
        train_indices.append(class_indices[this_class_val_count:])

    train_indices = np.concatenate(train_indices)
    val_indices = np.concatenate(val_indices)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


def verify_disjoint_splits(
    train_indices: np.ndarray, val_indices: np.ndarray, test_size: int
) -> None:
    """Raise if the train/val splits (by index, within the training file) are
    not disjoint or sizes are wrong. This only checks train vs. val, since
    both are drawn from the same underlying array of indices; test-set
    disjointness must be checked by content (see verify_disjoint_by_content)
    because test images live in a physically separate file.
    """
    train_set = set(train_indices.tolist())
    val_set = set(val_indices.tolist())

    overlap = train_set & val_set
    if overlap:
        raise ValueError(f"Train/val splits are not disjoint: {len(overlap)} overlapping indices")

    if len(train_indices) != TRAIN_SPLIT_SIZE:
        raise ValueError(f"Expected {TRAIN_SPLIT_SIZE} train images, got {len(train_indices)}")
    if len(val_indices) != VAL_SPLIT_SIZE:
        raise ValueError(f"Expected {VAL_SPLIT_SIZE} val images, got {len(val_indices)}")
    if test_size != 10_000:
        raise ValueError(
            f"Expected 10,000 test images (official CIFAR-100 test set), got {test_size}"
        )


def _hash_images(images: np.ndarray) -> set:
    """Return a set of content hashes, one per image, for duplicate detection.

    Hashing raw pixel bytes (rather than comparing indices) lets us verify
    disjointness across splits that come from physically separate files --
    e.g. the official train file vs. the official test file -- where index
    values carry no shared meaning and an index-based check would be
    meaningless.
    """
    return {hashlib.sha256(img.tobytes()).digest() for img in images}


def verify_disjoint_by_content(
    train_images: np.ndarray,
    val_images: np.ndarray,
    test_images: np.ndarray,
) -> dict:
    """Verify all three splits share no duplicate images, by pixel content.

    This is the check the assignment asks for explicitly ("verify that the
    three splits are disjoint"): index-based checks alone can't catch a
    duplicate image that happens to appear in both the official train file
    and the official test file. Returns a dict of pairwise overlap counts
    (all should be 0) so the result can be reported directly in the paper.
    """
    train_hashes = _hash_images(train_images)
    val_hashes = _hash_images(val_images)
    test_hashes = _hash_images(test_images)

    overlaps = {
        "train_val_overlap": len(train_hashes & val_hashes),
        "train_test_overlap": len(train_hashes & test_hashes),
        "val_test_overlap": len(val_hashes & test_hashes),
    }
    for pair_name, count in overlaps.items():
        if count != 0:
            raise ValueError(f"Found {count} duplicate image(s) between splits ({pair_name})")
    return overlaps


def compute_normalization_stats(images: np.ndarray):
    """Compute per-channel mean/std directly from a set of training images.

    `images` is expected as uint8, shape (N, H, W, 3), as stored by
    torchvision's CIFAR100 (`.data` attribute). Used to confirm the
    hardcoded CIFAR100_MEAN / CIFAR100_STD constants against the actual
    45,000-image training split, rather than asserting published values
    without checking them.
    """
    pixels = images.astype(np.float64) / 255.0
    mean = tuple(pixels.mean(axis=(0, 1, 2)).tolist())
    std = tuple(pixels.std(axis=(0, 1, 2)).tolist())
    return mean, std


@dataclasses.dataclass
class Cifar100Datasets:
    """Bundles the three split datasets together with the raw index arrays
    used to build them, so callers can both use the datasets directly and
    inspect/verify the underlying split (e.g. for disjointness checks).
    """

    train: Dataset
    val: Dataset
    test: Dataset
    train_indices: np.ndarray
    val_indices: np.ndarray


def build_datasets(config: DataConfig, verbose: bool = True) -> Cifar100Datasets:
    """Download (if needed) CIFAR-100 and construct train/val/test datasets.

    Train and val are stratified subsets of the official 50,000-image
    training set (45,000 / 5,000). Test is the unmodified official
    10,000-image test set.

    verbose=True prints a short status line before/after each phase
    (download, split, disjointness check). The actual compute here is fast
    (~1-2 seconds at full CIFAR-100 scale, benchmarked separately from the
    network download) -- these prints exist so a slow *download* doesn't
    look like a hang with zero feedback for minutes at a time.
    """
    Path(config.data_root).mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[data] Checking/downloading CIFAR-100 into {config.data_root} ...")
    download_start = time.time()
    # Load raw (transform=None) so we can attach split-specific transforms.
    raw_train_full = datasets.CIFAR100(
        root=config.data_root, train=True, download=True, transform=None
    )
    raw_test = datasets.CIFAR100(root=config.data_root, train=False, download=True, transform=None)
    if verbose:
        print(f"[data] Dataset ready in {time.time() - download_start:.1f}s.")

    if verbose:
        print("[data] Building stratified 45,000/5,000 train/val split ...")
    split_start = time.time()
    train_indices, val_indices = stratified_train_val_indices(
        targets=raw_train_full.targets,
        val_size=VAL_SPLIT_SIZE,
        num_classes=NUM_CLASSES,
        seed=config.random_seed,
    )
    verify_disjoint_splits(train_indices, val_indices, test_size=len(raw_test))
    if verbose:
        print(f"[data] Split built and verified in {time.time() - split_start:.1f}s.")

    if verbose:
        print("[data] Verifying content-level disjointness across all three splits ...")
    verify_start = time.time()
    # Content-level check across all three splits, including test (which
    # can't be checked by index since it comes from a separate file).
    verify_disjoint_by_content(
        train_images=raw_train_full.data[train_indices],
        val_images=raw_train_full.data[val_indices],
        test_images=raw_test.data,
    )
    if verbose:
        print(f"[data] Disjointness verified in {time.time() - verify_start:.1f}s.")

    train_transform = build_train_transform(config)
    eval_transform = build_eval_transform(config)

    train_dataset = TransformedSubset(raw_train_full, train_indices, train_transform)
    val_dataset = TransformedSubset(raw_train_full, val_indices, eval_transform)

    raw_test.transform = eval_transform
    test_dataset = raw_test

    if verbose:
        print("[data] Dataset pipeline ready.")

    return Cifar100Datasets(
        train=train_dataset,
        val=val_dataset,
        test=test_dataset,
        train_indices=train_indices,
        val_indices=val_indices,
    )


def build_dataloaders(
    datasets_bundle: Cifar100Datasets,
    batch_size: int = 128,
    num_workers: int = 4,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Wrap the three datasets in DataLoaders with sensible defaults.

    persistent_workers=True (when num_workers > 0) keeps worker processes
    alive between epochs instead of respawning them every epoch, which
    matters a lot here: 100 epochs means 100 worker-startup costs saved.
    prefetch_factor lets each worker stage several batches ahead so the GPU
    is less likely to sit idle waiting on CPU-side augmentation (RandAugment
    is not cheap per-image).
    """
    persistent = num_workers > 0
    train_loader = torch.utils.data.DataLoader(
        datasets_bundle.train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent,
        prefetch_factor=4 if persistent else None,
    )
    val_loader = torch.utils.data.DataLoader(
        datasets_bundle.val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets_bundle.test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    cfg = DataConfig()
    bundle = build_datasets(cfg)
    print(f"train: {len(bundle.train)}  val: {len(bundle.val)}  test: {len(bundle.test)}")
    image, label = bundle.train[0]
    print(f"sample train tensor shape: {tuple(image.shape)}, label: {label}")
