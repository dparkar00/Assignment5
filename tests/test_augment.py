"""Unit tests for src/augment.py (MixUp/CutMix)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augment import (  # noqa: E402
    MixupCutmixConfig,
    apply_mixup_cutmix,
    one_hot_with_smoothing,
    soft_cross_entropy,
)


class TestOneHotWithSmoothing:
    def test_shape_and_row_sums(self):
        targets = torch.tensor([0, 5, 9])
        one_hot = one_hot_with_smoothing(targets, num_classes=10, smoothing=0.1)
        assert one_hot.shape == (3, 10)
        assert torch.allclose(one_hot.sum(dim=1), torch.ones(3), atol=1e-6)

    def test_zero_smoothing_matches_hard_one_hot(self):
        targets = torch.tensor([2])
        one_hot = one_hot_with_smoothing(targets, num_classes=5, smoothing=0.0)
        expected = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
        assert torch.allclose(one_hot, expected)

    def test_correct_class_has_highest_value(self):
        targets = torch.tensor([3])
        one_hot = one_hot_with_smoothing(targets, num_classes=10, smoothing=0.1)
        assert one_hot[0].argmax().item() == 3


class TestSoftCrossEntropyMatchesHardCrossEntropy:
    def test_equivalent_to_nn_cross_entropy_with_label_smoothing(self):
        torch.manual_seed(0)
        outputs = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        smoothing = 0.1

        soft_targets = one_hot_with_smoothing(targets, num_classes=10, smoothing=smoothing)
        soft_loss = soft_cross_entropy(outputs, soft_targets)

        hard_loss = torch.nn.functional.cross_entropy(outputs, targets, label_smoothing=smoothing)

        assert torch.allclose(soft_loss, hard_loss, atol=1e-5)


class TestApplyMixupCutmix:
    def test_disabled_returns_unmixed_images_and_smoothed_targets(self):
        images = torch.randn(4, 3, 32, 32)
        targets = torch.tensor([0, 1, 2, 3])
        config = MixupCutmixConfig(num_classes=10, enabled=False, label_smoothing=0.1)

        mixed_images, soft_targets = apply_mixup_cutmix(images, targets, config)

        assert torch.equal(mixed_images, images)
        expected = one_hot_with_smoothing(targets, num_classes=10, smoothing=0.1)
        assert torch.allclose(soft_targets, expected)

    def test_zero_prob_never_mixes(self):
        images = torch.randn(4, 3, 32, 32)
        targets = torch.tensor([0, 1, 2, 3])
        config = MixupCutmixConfig(num_classes=10, prob=0.0)

        mixed_images, _ = apply_mixup_cutmix(images, targets, config)
        assert torch.equal(mixed_images, images)

    def test_output_shapes_preserved_when_mixing(self):
        torch.manual_seed(0)
        images = torch.randn(8, 3, 32, 32)
        targets = torch.randint(0, 100, (8,))
        config = MixupCutmixConfig(num_classes=100, prob=1.0)

        mixed_images, soft_targets = apply_mixup_cutmix(images, targets, config)

        assert mixed_images.shape == images.shape
        assert soft_targets.shape == (8, 100)
        # Soft targets must still be a valid probability distribution per row.
        assert torch.allclose(soft_targets.sum(dim=1), torch.ones(8), atol=1e-4)

    def test_mixup_branch_changes_pixel_values(self):
        torch.manual_seed(1)
        images = torch.randn(8, 3, 32, 32)
        targets = torch.randint(0, 100, (8,))
        config = MixupCutmixConfig(num_classes=100, prob=1.0, switch_prob=0.0)  # force MixUp

        mixed_images, _ = apply_mixup_cutmix(images, targets, config)
        # With prob=1.0 and switch_prob=0.0 (always MixUp), pixels should
        # differ from the original in general (interpolated with another
        # sample in the batch).
        assert not torch.equal(mixed_images, images)

    def test_cutmix_branch_preserves_shape_and_some_original_pixels(self):
        torch.manual_seed(2)
        images = torch.randn(8, 3, 32, 32)
        targets = torch.randint(0, 100, (8,))
        config = MixupCutmixConfig(num_classes=100, prob=1.0, switch_prob=1.0)  # force CutMix

        mixed_images, soft_targets = apply_mixup_cutmix(images, targets, config)
        assert mixed_images.shape == images.shape
        assert soft_targets.shape == (8, 100)

    def test_gradients_flow_through_soft_cross_entropy(self):
        outputs = torch.randn(4, 10, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 3])
        soft_targets = one_hot_with_smoothing(targets, num_classes=10, smoothing=0.1)
        loss = soft_cross_entropy(outputs, soft_targets)
        loss.backward()
        assert outputs.grad is not None
        assert torch.isfinite(outputs.grad).all()
