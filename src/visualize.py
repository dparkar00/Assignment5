"""Figure generation for Part 5: training_curves.png, confusion_matrices.png,
error_examples.png (the three figures named in the assignment's submission
structure).

Uses matplotlib's non-interactive 'Agg' backend throughout, since this runs
in headless environments (Colab, CI, this sandbox) with no display.
"""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")

# matplotlib.pyplot must be imported after matplotlib.use("Agg") above --
# the backend has to be set before pyplot picks one on its own, which is
# why these two imports can't be moved to the top with the rest.
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position
import numpy as np  # noqa: E402  pylint: disable=wrong-import-position


def _display_figure(fig) -> None:
    """Display a figure inline in a notebook (Colab/Jupyter).

    plt.show() silently does nothing under the 'Agg' backend forced above
    (Agg has no display capability at all -- it's rasterize-to-file-only),
    which is why a naive show=True previously produced no image, just the
    function's returned file path auto-printed by the notebook. IPython's
    display() renders the figure object directly through the notebook's
    rich-display protocol, bypassing the backend's own show() mechanism
    entirely, so it works correctly regardless of which backend is active.
    Falls back to a no-op outside a notebook (e.g. a plain script/CI run),
    where there's no rich display to render into anyway.
    """
    try:
        from IPython.display import display  # pylint: disable=import-outside-toplevel

        display(fig)
    except ImportError:
        pass


def _read_training_log(csv_path: str) -> dict:
    """Read a training CSV log (as written by src/train.py) into a dict of
    parallel lists, one entry per column.
    """
    columns = {
        "epoch": [], "train_loss": [], "train_accuracy": [], "val_loss": [],
        "val_accuracy": [], "learning_rate": [], "epoch_duration_seconds": [],
    }
    with open(csv_path, "r", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            for key, values in columns.items():
                values.append(float(row[key]))
    return columns


def plot_training_curves(log_paths: dict, output_path: str, show: bool = False) -> str:
    """Plot train/val loss, train/val accuracy, and LR schedule for one or
    more models on shared axes, so the curves are directly comparable.

    log_paths: {model_display_name: csv_log_path}, e.g.
        {"Swin (primary)": "logs/primary_training.csv",
         "ViT (baseline)": "logs/vit_training.csv"}
    show: if True, display inline (e.g. in a Colab/Jupyter cell) in
        addition to saving.
    """
    logs = {name: _read_training_log(path) for name, path in log_paths.items()}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for name, log in logs.items():
        axes[0, 0].plot(log["epoch"], log["train_loss"], label=name)
        axes[0, 1].plot(log["epoch"], log["val_loss"], label=name)
        axes[1, 0].plot(log["epoch"], log["train_accuracy"], label=name)
        axes[1, 0].plot(log["epoch"], log["val_accuracy"], label=f"{name} (val)", linestyle="--")
        axes[1, 1].plot(log["epoch"], log["learning_rate"], label=name)

    axes[0, 0].set_title("Training Loss")
    axes[0, 1].set_title("Validation Loss")
    axes[1, 0].set_title("Training / Validation Accuracy")
    axes[1, 1].set_title("Learning Rate Schedule")
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show:
        _display_figure(fig)
    plt.close(fig)
    return output_path


def plot_confusion_matrices(matrices: dict, output_path: str, show: bool = False) -> str:
    """Plot one confusion-matrix heatmap per model, side by side.

    matrices: {model_display_name: confusion_matrix (2D list or ndarray)}
    show: if True, display inline (e.g. in a Colab/Jupyter cell) in
        addition to saving -- the figure isn't closed until after showing.

    With 100 CIFAR-100 classes, per-cell tick labels aren't legible -- this
    plots the full matrix as a log-scaled heatmap (standard practice for
    many-class confusion matrices), which still makes the dominant diagonal
    and any off-diagonal confusion clusters visually apparent.
    """
    fig, axes = plt.subplots(1, len(matrices), figsize=(7 * len(matrices), 6))
    if len(matrices) == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, matrices.items()):
        cm_array = np.array(cm)
        # log1p so zero-count cells don't break a log color scale.
        image = ax.imshow(np.log1p(cm_array), cmap="viridis")
        ax.set_title(f"{name} -- Confusion Matrix (log scale)")
        ax.set_xlabel("Predicted class index")
        ax.set_ylabel("True class index")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show:
        _display_figure(fig)
    plt.close(fig)
    return output_path


def plot_misclassified_examples(
    examples: list, output_path: str, columns: int = 4, show: bool = False
) -> str:
    """Plot a grid of misclassified test images with true/predicted labels
    and predicted probability, per the Part 5 "at least 12 misclassified
    test images" requirement.

    examples: list of dicts with keys "image" (HxWx3 uint8 array),
        "true_label", "predicted_label", "predicted_probability".
    """
    num_examples = len(examples)
    rows = (num_examples + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3.2 * rows))
    axes_flat = np.array(axes).reshape(-1)

    for ax, example in zip(axes_flat, examples):
        ax.imshow(example["image"])
        ax.set_title(
            f"true: {example['true_label']}\n"
            f"pred: {example['predicted_label']} ({example['predicted_probability']:.2f})",
            fontsize=8,
        )
        ax.axis("off")

    for ax in axes_flat[num_examples:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show:
        _display_figure(fig)
    plt.close(fig)
    return output_path