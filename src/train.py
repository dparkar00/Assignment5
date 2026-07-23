"""Training script for the Vision Transformer assignment (Part 3).

Trains either the primary Swin model or the plain ViT baseline from random
initialization, with:
  - AdamW optimizer with weight decay
  - Linear warmup + cosine-decay learning-rate schedule
  - Gradient clipping
  - Mixed precision (when running on CUDA)
  - Label smoothing
  - Per-epoch training and validation
  - Best-checkpoint selection by validation accuracy (never by test accuracy)
  - Weights & Biases logging, plus a locally exported CSV log (the
    assignment requires both: a tracking platform AND an exported log file)
  - Reproducible seeding and explicit device selection

Usage:
    python -m src.train --config configs/primary.yaml
    python -m src.train --config configs/vit_baseline.yaml

For fast local testing without downloading CIFAR-100 or contacting W&B's
servers, use --dry-run, which trains on a small synthetic dataset for a
couple of epochs with W&B disabled. This validates the training loop itself
(forward/backward pass, checkpointing, CSV logging) without needing network
access or a full dataset.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import time
from pathlib import Path

import torch
import wandb
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from src.augment import MixupCutmixConfig, apply_mixup_cutmix, soft_cross_entropy
from src.data import build_data_config, build_datasets, build_dataloaders
from src.models import (
    SwinConfig,
    SwinTransformer,
    ViTConfig,
    VisionTransformer,
    count_parameters,
)
from src.utils import get_device, load_yaml_config, set_seed


def build_model(model_config: dict) -> nn.Module:
    """Instantiate SwinTransformer or VisionTransformer from a config dict."""
    name = model_config["name"]
    if name == "swin":
        config = SwinConfig(
            input_resolution=model_config["input_resolution"],
            patch_size=model_config["patch_size"],
            in_channels=model_config["in_channels"],
            window_size=model_config["window_size"],
            embed_dim=model_config["embed_dim"],
            num_stages=model_config["num_stages"],
            depths=tuple(model_config["depths"]),
            num_heads=tuple(model_config["num_heads"]),
            mlp_ratio=model_config["mlp_ratio"],
            dropout=model_config["dropout"],
            drop_path_rate=model_config["drop_path_rate"],
            num_classes=model_config["num_classes"],
        )
        return SwinTransformer(config)
    if name == "vit":
        config = ViTConfig(
            input_resolution=model_config["input_resolution"],
            patch_size=model_config["patch_size"],
            in_channels=model_config["in_channels"],
            embed_dim=model_config["embed_dim"],
            depth=model_config["depth"],
            num_heads=model_config["num_heads"],
            mlp_ratio=model_config["mlp_ratio"],
            dropout=model_config["dropout"],
            drop_path_rate=model_config["drop_path_rate"],
            num_classes=model_config["num_classes"],
        )
        return VisionTransformer(config)
    raise ValueError(f"Unknown model name: {name!r} (expected 'swin' or 'vit')")


def build_warmup_cosine_schedule(
    optimizer: torch.optim.Optimizer, warmup_epochs: int, total_epochs: int
) -> LambdaLR:
    """Linear warmup for `warmup_epochs`, then cosine decay to 0 by the end
    of training. Stepped once per epoch (not per batch) for simplicity.
    """

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress))).item()

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


@dataclasses.dataclass
class EpochContext:
    """Bundles the training-specific knobs run_one_epoch needs, so its
    signature doesn't keep growing every time a new training feature (grad
    clipping, mixed precision, MixUp/CutMix, ...) is added. `optimizer=None`
    signals a validation pass -- everything else is ignored in that case.
    """

    optimizer: torch.optim.Optimizer | None = None
    grad_clip_norm: float = 0.0
    scaler: torch.cuda.amp.GradScaler | None = None
    mixup_config: MixupCutmixConfig | None = None


def _forward_and_loss(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    mixup_config: MixupCutmixConfig | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass producing (outputs, loss), applying MixUp/CutMix first
    if configured. Shared by both the plain and mixed-precision code paths
    in _train_step, so that branching logic isn't duplicated.
    """
    if mixup_config is not None:
        mixed_images, soft_targets = apply_mixup_cutmix(images, targets, mixup_config)
        outputs = model(mixed_images)
        loss = soft_cross_entropy(outputs, soft_targets)
    else:
        outputs = model(images)
        loss = criterion(outputs, targets)
    return outputs, loss


def _train_step(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    context: EpochContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward + backward + optimizer step for one training batch. Returns
    (outputs, loss) for the caller to accumulate metrics from; model
    parameters are updated in place.
    """
    context.optimizer.zero_grad(set_to_none=True)
    use_amp = context.scaler is not None

    if use_amp:
        with torch.autocast(device_type=images.device.type, dtype=torch.float16):
            outputs, loss = _forward_and_loss(
                model, images, targets, criterion, context.mixup_config
            )
        context.scaler.scale(loss).backward()
        if context.grad_clip_norm > 0:
            context.scaler.unscale_(context.optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), context.grad_clip_norm)
        context.scaler.step(context.optimizer)
        context.scaler.update()
    else:
        outputs, loss = _forward_and_loss(model, images, targets, criterion, context.mixup_config)
        loss.backward()
        if context.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), context.grad_clip_norm)
        context.optimizer.step()

    return outputs, loss


def _eval_step(
    model: nn.Module, images: torch.Tensor, targets: torch.Tensor, criterion: nn.Module
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass only, no backward, no MixUp/CutMix -- for validation
    batches, which must stay a deterministic measurement on real data.
    """
    outputs = model(images)
    loss = criterion(outputs, targets)
    return outputs, loss


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    context: EpochContext,
) -> tuple[float, float]:
    """Run one full pass over `loader`. Training if `context.optimizer` is
    given, otherwise a no-grad validation pass. Returns (avg_loss, accuracy).

    Training accuracy under MixUp/CutMix is computed against each sample's
    *original* (pre-mix) label, which slightly undercounts correct
    predictions on mixed batches; this is standard practice and only
    affects the informational train_accuracy metric, not the loss actually
    optimized or any validation number.
    """
    is_training = context.optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    correct = 0
    total = 0

    grad_context = torch.enable_grad() if is_training else torch.no_grad()
    with grad_context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if is_training:
                outputs, loss = _train_step(model, images, targets, criterion, context)
            else:
                outputs, loss = _eval_step(model, images, targets, criterion)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += batch_size

    return total_loss / total, correct / total


def build_synthetic_loaders(
    batch_size: int, num_classes: int = 100
) -> tuple[DataLoader, DataLoader]:
    """Small in-memory dataset for --dry-run: validates the training loop
    itself without downloading CIFAR-100 or needing real data.
    """
    train_x = torch.randn(256, 3, 32, 32)
    train_y = torch.randint(0, num_classes, (256,))
    val_x = torch.randn(64, 3, 32, 32)
    val_y = torch.randint(0, num_classes, (64,))
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


@dataclasses.dataclass
class RunConfig:
    """Bundles the three YAML config sections (model/data/training) as one
    object, instead of passing three separate dicts to every function that
    needs some combination of them. Built once in `train()` and threaded
    through everything downstream.
    """

    model: dict
    data: dict
    training: dict


def resolve_run_config(
    config_path: str, data_root_override: str | None, checkpoint_dir_override: str | None
) -> RunConfig:
    """Load a YAML config and apply any path overrides, without ever
    writing back to the file on disk (see `train`'s docstring for why that
    matters for `git pull`).
    """
    raw_config = load_yaml_config(config_path)
    model_cfg, data_cfg, train_cfg = raw_config["model"], raw_config["data"], raw_config["training"]

    if data_root_override is not None:
        data_cfg = {**data_cfg, "data_root": data_root_override}
    if checkpoint_dir_override is not None:
        train_cfg = {**train_cfg, "checkpoint_dir": checkpoint_dir_override}

    return RunConfig(model=model_cfg, data=data_cfg, training=train_cfg)


def print_run_settings(config: RunConfig, device: torch.device, epochs: int) -> None:
    """Print every setting the assignment requires to be documented, so a
    full record of the run's configuration is captured in the run's stdout
    log alongside the CSV/W&B metrics.
    """
    model_cfg, data_cfg, train_cfg = config.model, config.data, config.training
    print(f"Device: {device}")
    print(f"Model: {model_cfg['name']}")
    print(
        f"Epochs: {epochs}  Batch size: {train_cfg['batch_size']}  "
        f"Seed: {train_cfg['random_seed']}"
    )
    print(
        f"Optimizer: {train_cfg['optimizer']}  LR: {train_cfg['learning_rate']}  "
        f"Weight decay: {train_cfg['weight_decay']}"
    )
    print(
        f"LR schedule: {train_cfg['lr_schedule']}  Warmup epochs: {train_cfg['warmup_epochs']}  "
        f"Label smoothing: {train_cfg['label_smoothing']}"
    )
    print(
        f"Grad clip norm: {train_cfg['grad_clip_norm']}  "
        f"Mixed precision: {train_cfg['mixed_precision'] and device.type == 'cuda'}"
    )
    mixup_enabled = data_cfg.get("mixup_cutmix_enabled", False)
    if mixup_enabled:
        print(
            f"MixUp/CutMix: enabled  mixup_alpha={data_cfg.get('mixup_alpha', 0.2)}  "
            f"cutmix_alpha={data_cfg.get('cutmix_alpha', 1.0)}  "
            f"prob={data_cfg.get('mixup_cutmix_prob', 0.5)}  "
            f"switch_prob={data_cfg.get('mixup_switch_prob', 0.5)}"
        )
    else:
        print("MixUp/CutMix: disabled")
    print("Pretrained weights: False (model built with random initialization)")


def build_loaders_from_config(config: RunConfig, dry_run: bool) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders, either synthetic (--dry-run) or real
    CIFAR-100 via src.data, using the resolution/batch size from config.
    """
    model_cfg, data_cfg, train_cfg = config.model, config.data, config.training
    if dry_run:
        return build_synthetic_loaders(
            batch_size=train_cfg["batch_size"], num_classes=model_cfg["num_classes"]
        )

    data_config = build_data_config(model_cfg, data_cfg)
    print(
        "Loading CIFAR-100 (downloads ~170MB on first run only; cached after "
        "that) ..."
    )
    datasets_bundle = build_datasets(data_config)
    train_loader, val_loader, _test_loader = build_dataloaders(
        datasets_bundle,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg.get("num_workers", 4),
    )
    return train_loader, val_loader


def build_mixup_config(config: RunConfig) -> MixupCutmixConfig | None:
    """Build the MixUp/CutMix config from the shared `data:` YAML section.

    Lives under `data:`, not `training:`, because MixUp/CutMix is a data
    augmentation, and "data augmentations" is one of the settings the
    assignment requires to be identical between configs/primary.yaml and
    configs/vit_baseline.yaml (Part 2, Experimental Controls). Returns None
    if the config doesn't enable it (e.g. an older config without these
    keys), in which case run_one_epoch falls back to standard hard-label
    cross entropy.
    """
    model_cfg, data_cfg, train_cfg = config.model, config.data, config.training
    if not data_cfg.get("mixup_cutmix_enabled", False):
        return None
    return MixupCutmixConfig(
        num_classes=model_cfg["num_classes"],
        mixup_alpha=data_cfg.get("mixup_alpha", 0.2),
        cutmix_alpha=data_cfg.get("cutmix_alpha", 1.0),
        prob=data_cfg.get("mixup_cutmix_prob", 0.5),
        switch_prob=data_cfg.get("mixup_switch_prob", 0.5),
        label_smoothing=train_cfg["label_smoothing"],
        enabled=True,
    )


def save_checkpoint(model: nn.Module, epoch: int, val_accuracy: float, path: Path) -> None:
    """Save a checkpoint containing everything needed to resume evaluation:
    epoch number, model weights, and the validation accuracy that earned it.
    """
    checkpoint = {
        "epoch": epoch, "model_state_dict": model.state_dict(), "val_accuracy": val_accuracy
    }
    torch.save(checkpoint, path)


def log_epoch_to_csv(log_path: Path, row: list) -> None:
    """Append one epoch's metrics as a row to the exported CSV log."""
    with open(log_path, "a", newline="", encoding="utf-8") as csv_file:
        csv.writer(csv_file).writerow(row)


@dataclasses.dataclass
class TrainingSetup:
    """Everything built from config that the epoch loop needs to actually
    train: optimizer, LR schedule, loss function, AMP scaler, and
    MixUp/CutMix config. Collapses five separate train() locals into one.
    """

    optimizer: torch.optim.Optimizer
    scheduler: LambdaLR
    criterion: nn.Module
    scaler: torch.cuda.amp.GradScaler | None
    mixup_config: MixupCutmixConfig | None


@dataclasses.dataclass
class ModelBundle:
    """Model + data loaders + device for a run, built together since the
    model must be moved to the same device the loaders will stream to.
    """

    model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    device: torch.device


def build_model_bundle(config: RunConfig, dry_run: bool) -> ModelBundle:
    """Pick a device, build the data loaders, and build+place the model."""
    device = get_device()
    train_loader, val_loader = build_loaders_from_config(config, dry_run)
    model = build_model(config.model).to(device)
    return ModelBundle(model=model, train_loader=train_loader, val_loader=val_loader, device=device)


def build_training_setup(bundle: ModelBundle, config: RunConfig, epochs: int) -> TrainingSetup:
    """Construct the optimizer, schedule, loss, scaler, and MixUp/CutMix
    config for a run. Also where the "only AdamW is supported" check lives.
    """
    train_cfg = config.training
    if train_cfg["optimizer"].lower() != "adamw":
        raise ValueError(
            f"Only 'adamw' is currently wired up (got {train_cfg['optimizer']!r}); "
            "AdamW is used because it decouples weight decay from the gradient "
            "update, which empirically works better than plain Adam or SGD for "
            "training Transformers from scratch."
        )
    optimizer = AdamW(
        bundle.model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = build_warmup_cosine_schedule(optimizer, train_cfg["warmup_epochs"], epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"])
    mixup_config = build_mixup_config(config)

    use_amp = train_cfg["mixed_precision"] and bundle.device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    return TrainingSetup(
        optimizer=optimizer, scheduler=scheduler, criterion=criterion, scaler=scaler,
        mixup_config=mixup_config,
    )


@dataclasses.dataclass
class OutputPaths:
    """Where this run's checkpoints and CSV log go."""

    checkpoint_dir: Path
    log_path: Path


@dataclasses.dataclass
class RunState:
    """Bundles the pieces that stay fixed for the whole run, so the
    per-epoch helper below doesn't need a long parameter list -- none of
    these change epoch to epoch.
    """

    bundle: ModelBundle
    paths: OutputPaths
    train_context: EpochContext
    val_context: EpochContext


def prepare_run_state(
    bundle: ModelBundle, config: RunConfig, setup: TrainingSetup, num_params: int, dry_run: bool
) -> RunState:
    """Create output directories, start the W&B run, write the CSV header,
    and bundle everything the epoch loop needs into one RunState.
    """
    train_cfg, model_cfg, data_cfg = config.training, config.model, config.data

    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(train_cfg["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=train_cfg["wandb_project"],
        name=train_cfg["wandb_run_name"],
        mode="disabled" if dry_run else "online",
        config={
            **model_cfg, **data_cfg, **train_cfg,
            "num_parameters": num_params, "device": str(bundle.device), "pretrained": False,
        },
    )

    csv_fields = [
        "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "learning_rate",
        "epoch_duration_seconds",
    ]
    with open(log_path, "w", newline="", encoding="utf-8") as csv_file:
        csv.writer(csv_file).writerow(csv_fields)

    return RunState(
        bundle=bundle,
        paths=OutputPaths(checkpoint_dir=checkpoint_dir, log_path=log_path),
        train_context=EpochContext(
            optimizer=setup.optimizer, grad_clip_norm=train_cfg["grad_clip_norm"],
            scaler=setup.scaler, mixup_config=setup.mixup_config,
        ),
        val_context=EpochContext(),
    )


def run_epoch_and_log(state: RunState, setup: TrainingSetup, epoch: int, epochs: int) -> float:
    """Run one epoch of train+val, log to stdout/W&B/CSV, checkpoint the
    latest weights, and return this epoch's validation accuracy (the
    caller is responsible for tracking the best one across epochs).
    """
    epoch_start = time.time()
    train_loss, train_accuracy = run_one_epoch(
        state.bundle.model, state.bundle.train_loader, state.bundle.device,
        setup.criterion, state.train_context,
    )
    val_loss, val_accuracy = run_one_epoch(
        state.bundle.model, state.bundle.val_loader, state.bundle.device,
        setup.criterion, state.val_context,
    )

    current_lr = setup.optimizer.param_groups[0]["lr"]
    setup.scheduler.step()
    epoch_duration = time.time() - epoch_start

    print(
        f"Epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  train_acc={train_accuracy:.4f}  "
        f"val_loss={val_loss:.4f}  val_acc={val_accuracy:.4f}  lr={current_lr:.6f}  "
        f"time={epoch_duration:.1f}s"
    )
    wandb.log(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": current_lr,
            "epoch_duration_seconds": epoch_duration,
        }
    )
    log_epoch_to_csv(
        state.paths.log_path,
        [epoch + 1, train_loss, train_accuracy, val_loss, val_accuracy, current_lr, epoch_duration],
    )
    save_checkpoint(
        state.bundle.model, epoch + 1, val_accuracy, state.paths.checkpoint_dir / "last.pt"
    )

    return val_accuracy


def train(
    config_path: str,
    dry_run: bool = False,
    epochs_override: int | None = None,
    data_root_override: str | None = None,
    checkpoint_dir_override: str | None = None,
) -> None:
    """Full training entry point, driven entirely by a YAML config file.

    data_root_override / checkpoint_dir_override let a caller (e.g. the
    Colab notebook, redirecting to Google Drive) point at different paths
    without editing the YAML file itself -- editing the tracked config file
    in place is what causes every subsequent `git pull` to conflict with
    "local changes." These overrides only affect this run's in-memory
    config, never the file on disk.
    """
    config = resolve_run_config(config_path, data_root_override, checkpoint_dir_override)

    set_seed(config.training["random_seed"])
    epochs = epochs_override if epochs_override is not None else config.training["epochs"]

    bundle = build_model_bundle(config, dry_run)
    print_run_settings(config, bundle.device, epochs)
    num_params = count_parameters(bundle.model)
    print(f"Trainable parameters: {num_params:,}")

    setup = build_training_setup(bundle, config, epochs)
    state = prepare_run_state(bundle, config, setup, num_params, dry_run)

    best_val_accuracy = 0.0
    for epoch in range(epochs):
        val_accuracy = run_epoch_and_log(state, setup, epoch, epochs)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(
                state.bundle.model, epoch + 1, val_accuracy, state.paths.checkpoint_dir / "best.pt"
            )

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    wandb.finish()

    # Explicit cleanup: this function is called twice in a row in the same
    # Python process when training both models from one Colab cell, so
    # anything left referencing GPU/CPU memory (model, optimizer, dataloader
    # workers) should be released rather than accumulating across both runs.
    del bundle, setup, state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train the Swin or ViT model on CIFAR-100.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train on synthetic data for a couple epochs with W&B off; sanity-checks the loop.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override the config's epoch count (useful with --dry-run)."
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Override the config's data.data_root (e.g. a Google Drive path on Colab), "
        "without editing the YAML file itself.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Override the config's training.checkpoint_dir, without editing the YAML file itself.",
    )
    args = parser.parse_args()
    train(
        args.config,
        dry_run=args.dry_run,
        epochs_override=args.epochs,
        data_root_override=args.data_root,
        checkpoint_dir_override=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
