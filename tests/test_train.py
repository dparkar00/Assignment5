"""Unit tests for src/train.py and src/utils.py.

These exercise the training loop against synthetic in-memory data so they
run quickly and without downloading CIFAR-100 or contacting W&B's servers
(tests set WANDB_MODE=disabled).
"""

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("WANDB_MODE", "disabled")

from src.models import SwinTransformer, VisionTransformer  # noqa: E402
from src.train import (  # noqa: E402
    build_model,
    build_synthetic_loaders,
    build_warmup_cosine_schedule,
    run_one_epoch,
    train,
)
from src.utils import get_device, set_seed  # noqa: E402


SWIN_MODEL_CONFIG = {
    "name": "swin",
    "input_resolution": 32,
    "patch_size": 4,
    "in_channels": 3,
    "window_size": 4,
    "embed_dim": 24,
    "num_stages": 2,
    "depths": [1, 1],
    "num_heads": [2, 4],
    "mlp_ratio": 2.0,
    "dropout": 0.0,
    "drop_path_rate": 0.0,
    "num_classes": 10,
}

VIT_MODEL_CONFIG = {
    "name": "vit",
    "input_resolution": 32,
    "patch_size": 4,
    "in_channels": 3,
    "embed_dim": 32,
    "depth": 2,
    "num_heads": 4,
    "mlp_ratio": 2.0,
    "dropout": 0.0,
    "drop_path_rate": 0.0,
    "num_classes": 10,
}


class TestSetSeedReproducibility:
    def test_same_seed_gives_identical_tensors(self):
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.equal(a, b)

    def test_different_seeds_give_different_tensors(self):
        set_seed(1)
        a = torch.randn(10)
        set_seed(2)
        b = torch.randn(10)
        assert not torch.equal(a, b)


class TestGetDevice:
    def test_returns_a_valid_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cuda", "mps", "cpu")


class TestBuildModel:
    def test_builds_swin_from_config(self):
        model = build_model(SWIN_MODEL_CONFIG)
        assert isinstance(model, SwinTransformer)

    def test_builds_vit_from_config(self):
        model = build_model(VIT_MODEL_CONFIG)
        assert isinstance(model, VisionTransformer)

    def test_rejects_unknown_model_name(self):
        with pytest.raises(ValueError, match="Unknown model name"):
            build_model({**VIT_MODEL_CONFIG, "name": "resnet"})


class TestWarmupCosineSchedule:
    def test_lr_increases_during_warmup(self):
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        scheduler = build_warmup_cosine_schedule(optimizer, warmup_epochs=5, total_epochs=20)

        lrs = []
        for _ in range(5):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()
        # Warmup should be monotonically non-decreasing.
        assert all(lrs[i] <= lrs[i + 1] for i in range(len(lrs) - 1))

    def test_lr_decreases_after_warmup(self):
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        scheduler = build_warmup_cosine_schedule(optimizer, warmup_epochs=5, total_epochs=20)

        for _ in range(5):  # finish warmup
            scheduler.step()
        post_warmup_lr = optimizer.param_groups[0]["lr"]
        for _ in range(10):
            scheduler.step()
        later_lr = optimizer.param_groups[0]["lr"]
        assert later_lr < post_warmup_lr


class TestRunOneEpoch:
    def test_training_pass_returns_finite_metrics(self):
        model = build_model(SWIN_MODEL_CONFIG)
        train_loader, _ = build_synthetic_loaders(batch_size=8, num_classes=10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        loss, accuracy = run_one_epoch(
            model, train_loader, torch.device("cpu"), criterion, optimizer, grad_clip_norm=1.0, scaler=None
        )
        assert loss == loss  # not NaN
        assert 0.0 <= accuracy <= 1.0

    def test_validation_pass_does_not_update_weights(self):
        model = build_model(VIT_MODEL_CONFIG)
        _, val_loader = build_synthetic_loaders(batch_size=8, num_classes=10)
        criterion = torch.nn.CrossEntropyLoss()

        params_before = [p.clone() for p in model.parameters()]
        run_one_epoch(
            model, val_loader, torch.device("cpu"), criterion, optimizer=None, grad_clip_norm=0.0, scaler=None
        )
        params_after = list(model.parameters())
        assert all(torch.equal(a, b) for a, b in zip(params_before, params_after))


class TestFullDryRunTrainingLoop:
    def test_dry_run_produces_csv_log_and_checkpoints(self, tmp_path):
        config_path = tmp_path / "test_config.yaml"
        checkpoint_dir = tmp_path / "checkpoints"
        log_path = tmp_path / "logs" / "test_training.csv"

        config_path.write_text(
            f"""
model:
  name: vit
  input_resolution: 32
  patch_size: 4
  in_channels: 3
  embed_dim: 32
  depth: 2
  num_heads: 4
  mlp_ratio: 2.0
  dropout: 0.0
  drop_path_rate: 0.0
  num_classes: 10

data:
  data_root: ./data
  random_seed: 42
  randaugment_num_ops: 2
  randaugment_magnitude: 9
  random_crop_padding: 4

training:
  batch_size: 8
  random_seed: 42
  epochs: 2
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.01
  lr_schedule: cosine
  warmup_epochs: 1
  label_smoothing: 0.0
  grad_clip_norm: 1.0
  mixed_precision: false
  checkpoint_dir: {checkpoint_dir}
  log_path: {log_path}
  wandb_project: test-project
  wandb_run_name: test-run
"""
        )

        train(str(config_path), dry_run=True, epochs_override=2)

        assert log_path.exists()
        log_contents = log_path.read_text()
        assert "train_loss" in log_contents
        assert log_contents.strip().count("\n") == 2  # header + 2 epoch rows

        assert (checkpoint_dir / "last.pt").exists()
        assert (checkpoint_dir / "best.pt").exists()

        checkpoint = torch.load(checkpoint_dir / "best.pt", weights_only=True)
        assert "model_state_dict" in checkpoint
        assert "val_accuracy" in checkpoint
        assert "epoch" in checkpoint
