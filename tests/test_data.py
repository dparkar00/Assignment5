"""Unit tests for src/data.py.

These tests exercise the splitting logic directly against synthetic label
arrays so they run without downloading CIFAR-100, and are fast enough to
run on every commit.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (  # noqa: E402
    build_eval_transform,
    build_train_transform,
    compute_normalization_stats,
    stratified_train_val_indices,
    verify_disjoint_by_content,
    verify_disjoint_splits,
    DataConfig,
)


def make_synthetic_targets(num_classes: int, per_class: int) -> list[int]:
    """Build a synthetic label list mimicking CIFAR-100's 500 images/class train set."""
    targets = []
    for class_id in range(num_classes):
        targets.extend([class_id] * per_class)
    return targets


class TestStratifiedSplit:
    def test_split_sizes_match_request(self):
        targets = make_synthetic_targets(num_classes=100, per_class=500)  # 50,000 total
        train_idx, val_idx = stratified_train_val_indices(
            targets, val_size=5000, num_classes=100, seed=705643
        )
        assert len(train_idx) == 45_000
        assert len(val_idx) == 5_000

    def test_splits_are_disjoint(self):
        targets = make_synthetic_targets(num_classes=100, per_class=500)
        train_idx, val_idx = stratified_train_val_indices(
            targets, val_size=5000, num_classes=100, seed=705643
        )
        assert set(train_idx.tolist()).isdisjoint(set(val_idx.tolist()))

    def test_splits_cover_all_indices_exactly_once(self):
        targets = make_synthetic_targets(num_classes=100, per_class=500)
        train_idx, val_idx = stratified_train_val_indices(
            targets, val_size=5000, num_classes=100, seed=705643
        )
        combined = np.concatenate([train_idx, val_idx])
        assert len(combined) == len(set(combined.tolist())) == 50_000

    def test_val_split_is_class_balanced(self):
        targets = np.asarray(make_synthetic_targets(num_classes=100, per_class=500))
        _, val_idx = stratified_train_val_indices(
            targets.tolist(), val_size=5000, num_classes=100, seed=705643
        )
        val_labels = targets[val_idx]
        counts = np.bincount(val_labels, minlength=100)
        # 5000 / 100 classes = exactly 50 per class with no remainder
        assert (counts == 50).all()

    def test_seed_is_reproducible(self):
        targets = make_synthetic_targets(num_classes=100, per_class=500)
        train_idx_a, val_idx_a = stratified_train_val_indices(
            targets, val_size=5000, num_classes=100, seed=705643
        )
        train_idx_b, val_idx_b = stratified_train_val_indices(
            targets, val_size=5000, num_classes=100, seed=705643
        )
        assert np.array_equal(train_idx_a, train_idx_b)
        assert np.array_equal(val_idx_a, val_idx_b)


class TestVerifyDisjointSplits:
    def test_raises_on_overlap(self):
        train_idx = np.arange(0, 45_000)
        val_idx = np.arange(44_999, 49_999)  # deliberately overlaps by one index
        with pytest.raises(ValueError, match="not disjoint"):
            verify_disjoint_splits(train_idx, val_idx, test_size=10_000)

    def test_raises_on_wrong_train_size(self):
        train_idx = np.arange(0, 44_000)
        val_idx = np.arange(44_000, 49_000)
        with pytest.raises(ValueError, match="45000"):
            verify_disjoint_splits(train_idx, val_idx, test_size=10_000)

    def test_passes_on_correct_sizes(self):
        train_idx = np.arange(0, 45_000)
        val_idx = np.arange(45_000, 50_000)
        # Should not raise.
        verify_disjoint_splits(train_idx, val_idx, test_size=10_000)


class TestDisjointByContent:
    def test_passes_on_truly_distinct_images(self):
        rng = np.random.default_rng(0)
        train_images = rng.integers(0, 255, size=(20, 4, 4, 3), dtype=np.uint8)
        val_images = rng.integers(0, 255, size=(10, 4, 4, 3), dtype=np.uint8)
        test_images = rng.integers(0, 255, size=(10, 4, 4, 3), dtype=np.uint8)
        overlaps = verify_disjoint_by_content(train_images, val_images, test_images)
        assert overlaps == {"train_val_overlap": 0, "train_test_overlap": 0, "val_test_overlap": 0}

    def test_raises_on_duplicate_image_across_splits(self):
        rng = np.random.default_rng(1)
        train_images = rng.integers(0, 255, size=(5, 4, 4, 3), dtype=np.uint8)
        val_images = rng.integers(0, 255, size=(5, 4, 4, 3), dtype=np.uint8)
        test_images = rng.integers(0, 255, size=(5, 4, 4, 3), dtype=np.uint8)
        test_images[0] = train_images[0]
        with pytest.raises(ValueError, match="duplicate"):
            verify_disjoint_by_content(train_images, val_images, test_images)


class TestNormalizationStats:
    def test_matches_known_values_for_uniform_gray_images(self):
        images = np.full((10, 4, 4, 3), 128, dtype=np.uint8)
        mean, std = compute_normalization_stats(images)
        assert all(abs(m - 128 / 255) < 1e-6 for m in mean)
        assert all(s < 1e-6 for s in std)

    def test_returns_three_channel_tuples(self):
        rng = np.random.default_rng(2)
        images = rng.integers(0, 255, size=(50, 4, 4, 3), dtype=np.uint8)
        mean, std = compute_normalization_stats(images)
        assert len(mean) == 3
        assert len(std) == 3
        assert all(0.0 <= m <= 1.0 for m in mean)


class TestTransforms:
    def test_eval_transform_has_no_random_ops(self):
        config = DataConfig()
        eval_transform = build_eval_transform(config)
        transform_names = [type(t).__name__ for t in eval_transform.transforms]
        random_op_markers = ("Random", "RandAugment")
        offending = [name for name in transform_names if any(marker in name for marker in random_op_markers)]
        assert not offending, f"Eval transform must be deterministic, found: {offending}"

    def test_train_transform_includes_required_augmentations(self):
        config = DataConfig()
        train_transform = build_train_transform(config)
        transform_names = {type(t).__name__ for t in train_transform.transforms}
        assert "RandomCrop" in transform_names
        assert "RandomHorizontalFlip" in transform_names
        assert "RandAugment" in transform_names  # the "at least one additional augmentation"
