"""Unit tests for src/models.py.

Covers, per the assignment's suggested test topics for Part 6: model output
shape, patch-embedding/token shape, parameter-count calculation, invalid
patch-size handling, checkpoint saving/loading, and configuration
validation -- plus the Part 2.2 requirement that the ViT baseline's
parameter count sits within 10% of the primary Swin model's.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import (  # noqa: E402
    PatchEmbed,
    SwinConfig,
    SwinTransformer,
    ViTConfig,
    VisionTransformer,
    count_parameters,
)


BATCH_SIZE = 4
NUM_CLASSES = 100


def make_dummy_batch(resolution: int = 32, batch_size: int = BATCH_SIZE) -> torch.Tensor:
    return torch.randn(batch_size, 3, resolution, resolution)


class TestPatchEmbedding:
    def test_output_shape(self):
        embed = PatchEmbed(input_resolution=32, patch_size=4, in_channels=3, embed_dim=96)
        x = make_dummy_batch()
        out = embed(x)
        # 32 / 4 = 8 -> 8*8 = 64 tokens
        assert out.shape == (BATCH_SIZE, 64, 96)

    def test_rejects_resolution_not_divisible_by_patch_size(self):
        with pytest.raises(ValueError, match="divisible"):
            PatchEmbed(input_resolution=33, patch_size=4, in_channels=3, embed_dim=96)


class TestSwinTransformer:
    def test_output_shape_matches_num_classes(self):
        model = SwinTransformer(SwinConfig())
        out = model(make_dummy_batch())
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_output_shape_with_different_batch_sizes(self):
        model = SwinTransformer(SwinConfig())
        for batch_size in [1, 2, 8]:
            out = model(make_dummy_batch(batch_size=batch_size))
            assert out.shape == (batch_size, NUM_CLASSES)

    def test_config_validation_rejects_mismatched_depths(self):
        with pytest.raises(ValueError, match="depths"):
            SwinConfig(num_stages=3, depths=(2, 2), num_heads=(3, 6, 12))

    def test_config_validation_rejects_mismatched_heads(self):
        with pytest.raises(ValueError, match="num_heads"):
            SwinConfig(num_stages=3, depths=(2, 2, 6), num_heads=(3, 6))

    def test_invalid_patch_size_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            SwinTransformer(SwinConfig(input_resolution=32, patch_size=5))

    def test_all_parameters_receive_gradients(self):
        model = SwinTransformer(SwinConfig())
        x = make_dummy_batch()
        target = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
        loss = torch.nn.functional.cross_entropy(model(x), target)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} received no gradient"

    def test_parameter_count_matches_manual_sum(self):
        model = SwinTransformer(SwinConfig())
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert count_parameters(model) == manual_count

    def test_configurable_num_stages_and_window_size(self):
        # Confirms the architecture actually respects the configurable
        # settings the assignment requires, not just default values.
        config = SwinConfig(
            input_resolution=32,
            patch_size=2,
            window_size=8,
            embed_dim=64,
            num_stages=4,
            depths=(2, 2, 2, 2),
            num_heads=(2, 4, 8, 16),
        )
        model = SwinTransformer(config)
        out = model(make_dummy_batch())
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)


class TestVisionTransformer:
    def test_output_shape_matches_num_classes(self):
        model = VisionTransformer(ViTConfig())
        out = model(make_dummy_batch())
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_output_shape_with_different_batch_sizes(self):
        model = VisionTransformer(ViTConfig())
        for batch_size in [1, 2, 8]:
            out = model(make_dummy_batch(batch_size=batch_size))
            assert out.shape == (batch_size, NUM_CLASSES)

    def test_invalid_patch_size_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            VisionTransformer(ViTConfig(input_resolution=32, patch_size=5))

    def test_head_dim_must_divide_embed_dim(self):
        with pytest.raises(ValueError, match="divisible"):
            VisionTransformer(ViTConfig(embed_dim=100, num_heads=6))

    def test_all_parameters_receive_gradients(self):
        model = VisionTransformer(ViTConfig())
        x = make_dummy_batch()
        target = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
        loss = torch.nn.functional.cross_entropy(model(x), target)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} received no gradient"

    def test_parameter_count_matches_manual_sum(self):
        model = VisionTransformer(ViTConfig())
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert count_parameters(model) == manual_count


class TestParameterMatching:
    """Part 2.2: the ViT baseline must be within 10% of the primary model's
    trainable parameter count.
    """

    def test_default_configs_are_within_ten_percent(self):
        swin = SwinTransformer(SwinConfig())
        vit = VisionTransformer(ViTConfig())
        swin_params = count_parameters(swin)
        vit_params = count_parameters(vit)
        pct_diff = abs(swin_params - vit_params) / swin_params * 100
        assert pct_diff <= 10.0, (
            f"ViT baseline ({vit_params:,}) is {pct_diff:.2f}% away from "
            f"Swin ({swin_params:,}); must be <= 10%"
        )


class TestCheckpointing:
    """Checkpoint saving and loading, per the Part 6 example test topics."""

    def test_swin_checkpoint_round_trip_preserves_output(self, tmp_path):
        model = SwinTransformer(SwinConfig())
        model.eval()
        x = make_dummy_batch(batch_size=2)
        with torch.no_grad():
            original_output = model(x)

        checkpoint_path = tmp_path / "swin_checkpoint.pt"
        torch.save(model.state_dict(), checkpoint_path)

        reloaded = SwinTransformer(SwinConfig())
        reloaded.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        reloaded.eval()
        with torch.no_grad():
            reloaded_output = reloaded(x)

        assert torch.allclose(original_output, reloaded_output, atol=1e-6)

    def test_vit_checkpoint_round_trip_preserves_output(self, tmp_path):
        model = VisionTransformer(ViTConfig())
        model.eval()
        x = make_dummy_batch(batch_size=2)
        with torch.no_grad():
            original_output = model(x)

        checkpoint_path = tmp_path / "vit_checkpoint.pt"
        torch.save(model.state_dict(), checkpoint_path)

        reloaded = VisionTransformer(ViTConfig())
        reloaded.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        reloaded.eval()
        with torch.no_grad():
            reloaded_output = reloaded(x)

        assert torch.allclose(original_output, reloaded_output, atol=1e-6)
