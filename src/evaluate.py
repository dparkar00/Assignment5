"""Evaluation script for Part 5 (Evaluation and Experimental Analysis).

Loads the checkpoint with the best validation accuracy and evaluates it
ONCE on the complete CIFAR-100 test set, computing every metric the
assignment requires (top-1 accuracy, macro/weighted precision-recall-F1,
per-class breakdown, confusion matrix, one-vs-rest macro ROC-AUC).

Per the assignment: "Do not repeatedly evaluate on the test set while
tuning your models. Select all model settings and hyperparameters using
validation results." This script is meant to be run once per model, after
training and hyperparameter selection are already finished -- both of
which should have been driven entirely by validation accuracy (see
src/train.py, which only ever checkpoints "best" by val_accuracy and never
touches the test set).

Usage:
    python -m src.evaluate --config configs/primary.yaml \
        --checkpoint checkpoints/primary/best.pt
    python -m src.evaluate --config configs/vit_baseline.yaml \
        --checkpoint checkpoints/vit_baseline/best.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data import build_data_config, build_dataloaders, build_datasets
from src.metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
    save_per_class_csv,
    top_and_bottom_classes,
)
from src.train import build_model
from src.utils import get_device, load_yaml_config
from src.visualize import plot_confusion_matrices, plot_misclassified_examples


def load_model_from_checkpoint(
    config_path: str, checkpoint_path: str, device: torch.device
) -> tuple[torch.nn.Module, dict, dict]:
    """Build a model from its YAML config and load trained weights into it."""
    config = load_yaml_config(config_path)
    model = build_model(config["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, checkpoint


@torch.no_grad()
def run_inference(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over a dataloader once. Returns (y_true, y_pred, y_proba).

    y_proba (softmax outputs) is what ROC-AUC needs; y_pred (argmax of
    y_proba) is what every other metric needs. Both are derived from a
    single forward pass per batch, not computed separately.
    """
    model.eval()
    true_batches, pred_batches, proba_batches = [], [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        proba = F.softmax(outputs, dim=1)
        preds = proba.argmax(dim=1)
        true_batches.append(targets.numpy())
        pred_batches.append(preds.cpu().numpy())
        proba_batches.append(proba.cpu().numpy())
    return np.concatenate(true_batches), np.concatenate(pred_batches), np.concatenate(proba_batches)


def build_test_loader(config: dict, data_root_override: str | None) -> tuple:
    """Build the test dataloader and its class names from a model's config."""
    data_cfg, model_cfg, train_cfg = config["data"], config["model"], config["training"]
    data_config = build_data_config(model_cfg, data_cfg, data_root_override)
    datasets_bundle = build_datasets(data_config)
    _, _, test_loader = build_dataloaders(datasets_bundle, batch_size=train_cfg["batch_size"])
    class_names = getattr(datasets_bundle.test, "classes", None)
    raw_images = getattr(datasets_bundle.test, "data", None)
    return test_loader, class_names, raw_images


def collect_misclassified_examples(
    raw_images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: list,
    n: int = 12,
    rule: str = "most_confident_incorrect",
    seed: int = 42,
) -> list:
    """Select misclassified test images for the Part 5 error-analysis
    figure (at least 12 required).

    rule="most_confident_incorrect": the model's most confident WRONG
        predictions -- usually the most informative failures to inspect,
        since these are cases the model was sure about and still got wrong.
    rule="random": a fixed random sample of misclassified images, seeded
        for reproducibility.

    Both are documented selection rules, as the assignment requires ("use
    a documented selection rule"). raw_images must be index-aligned with
    y_true/y_pred/y_proba (true when the test DataLoader used shuffle=False,
    which src/data.py's build_dataloaders always does for the test split).
    """
    incorrect_indices = np.where(y_pred != y_true)[0]
    if len(incorrect_indices) == 0:
        return []

    if rule == "most_confident_incorrect":
        confidences = y_proba[incorrect_indices, y_pred[incorrect_indices]]
        order = np.argsort(-confidences)
        selected = incorrect_indices[order][:n]
    elif rule == "random":
        rng = np.random.default_rng(seed)
        count = min(n, len(incorrect_indices))
        selected = rng.choice(incorrect_indices, size=count, replace=False)
    else:
        raise ValueError(
            f"Unknown selection rule: {rule!r} (expected 'most_confident_incorrect' or 'random')"
        )

    return [
        {
            "image": raw_images[idx],
            "true_label": class_names[y_true[idx]],
            "predicted_label": class_names[y_pred[idx]],
            "predicted_probability": float(y_proba[idx, y_pred[idx]]),
        }
        for idx in selected
    ]


def print_summary(model_name: str, checkpoint: dict, metrics: ClassificationMetrics) -> None:
    """Print the headline numbers -- the full per-class breakdown and
    confusion matrix go to the saved JSON instead, since they're too large
    for a readable console summary (100 classes for CIFAR-100).
    """
    print(f"\n=== {model_name} -- test set evaluation ===")
    print(f"Checkpoint: epoch {checkpoint['epoch']}, val_accuracy={checkpoint['val_accuracy']:.4f}")
    print(f"Top-1 accuracy:        {metrics.top1_accuracy:.4f}")
    print(f"Macro precision:       {metrics.macro_precision:.4f}")
    print(f"Macro recall:          {metrics.macro_recall:.4f}")
    print(f"Macro F1:              {metrics.macro_f1:.4f}")
    print(f"Weighted F1:           {metrics.weighted_f1:.4f}")
    print(f"ROC-AUC (macro, OvR):  {metrics.roc_auc_macro_ovr:.4f}")

    extremes = top_and_bottom_classes(metrics, n=5)
    print("\nTop 5 classes by F1:")
    for name, f1_score in extremes["highest_f1"]:
        print(f"  {name}: {f1_score:.4f}")
    print("Bottom 5 classes by F1:")
    for name, f1_score in extremes["lowest_f1"]:
        print(f"  {name}: {f1_score:.4f}")


@dataclasses.dataclass
class InferenceResults:
    """Bundles one test-set inference pass, so evaluate() can both compute
    metrics from it and build the error-examples figure from it, without
    threading five separate arrays through the function signatures.
    """

    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    class_names: list
    raw_images: np.ndarray | None


def _run_test_inference(
    model: torch.nn.Module, config: dict, device: torch.device, data_root: str | None
) -> InferenceResults:
    """Build the test loader and run inference once -- the only part of
    evaluation that touches real data.
    """
    test_loader, class_names, raw_images = build_test_loader(config, data_root)
    y_true, y_pred, y_proba = run_inference(model, test_loader, device)
    return InferenceResults(y_true, y_pred, y_proba, class_names, raw_images)


def _save_metrics_json(metrics: ClassificationMetrics, model_name: str, output_dir: str) -> Path:
    """Save the full metrics (including per-class breakdown and confusion
    matrix, too large for a console summary) to a JSON file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / f"{model_name}_test_metrics.json"
    with open(result_file, "w", encoding="utf-8") as json_file:
        json.dump(metrics.to_dict(), json_file, indent=2)
    return result_file


def _save_error_examples_figure(
    results: InferenceResults, model_name: str, output_dir: str, n: int = 12, show: bool = False
) -> Path | None:
    """Select and plot misclassified test images, per the Part 5
    "at least 12 misclassified test images" requirement. Returns None if
    raw images weren't available (e.g. an unusual test dataset without a
    `.data` attribute) rather than failing the whole evaluation.
    """
    if results.raw_images is None:
        print(
            f"[evaluate] No raw image data available -- "
            f"skipping error-examples figure for {model_name}."
        )
        return None

    examples = collect_misclassified_examples(
        results.raw_images, results.y_true, results.y_pred, results.y_proba,
        results.class_names, n=n
    )
    if not examples:
        print(f"[evaluate] No misclassified examples found for {model_name} (100% test accuracy?).")
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figure_path = output_path / f"{model_name}_error_examples.png"
    plot_misclassified_examples(examples, str(figure_path), show=show)
    return figure_path


def _save_confusion_matrix_figure(
    metrics: ClassificationMetrics, model_name: str, figures_dir: str, show: bool = False
) -> Path:
    """Plot and save this model's confusion matrix, per Part 5's required
    figures. Saved standalone (one model), unlike the combined two-model
    figure a report would eventually want -- see src/visualize.py to build
    that separately once both models have been evaluated.
    """
    output_path = Path(figures_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figure_path = output_path / f"{model_name}_confusion_matrix.png"
    plot_confusion_matrices({model_name: metrics.confusion_matrix}, str(figure_path), show=show)
    return figure_path


def _save_and_display_per_class_table(
    metrics: ClassificationMetrics, model_name: str, output_dir: str, show: bool = False
) -> Path:
    """Save the per-class breakdown as CSV (still useful for the report's
    appendix/raw data), and additionally *display* it as a readable table
    inline when show=True -- so a notebook caller doesn't have to reload
    the CSV themselves just to see it.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / f"{model_name}_per_class.csv"
    save_per_class_csv(metrics, str(csv_path))

    if show:
        try:
            import pandas as pd  # pylint: disable=import-outside-toplevel
            from IPython.display import display  # pylint: disable=import-outside-toplevel

            display(pd.read_csv(csv_path))
        except ImportError:
            # Outside a notebook / pandas unavailable -- print a plain table instead.
            print(f"\n{model_name} per-class breakdown:")
            with open(csv_path, "r", encoding="utf-8") as csv_file:
                print(csv_file.read())

    return csv_path


def _save_and_report_all_outputs(
    metrics: ClassificationMetrics,
    results: InferenceResults,
    model_name: str,
    output_dir: str,
    figures_dir: str,
    show: bool,
) -> None:
    """Save (and optionally display) every Part 5 deliverable this
    evaluation produces: the full metrics JSON, the per-class CSV/table,
    the confusion matrix figure, and the error-examples figure. Bundled
    into one call so evaluate() itself doesn't need a separate local
    variable and print statement per output.
    """
    result_file = _save_metrics_json(metrics, model_name, output_dir)
    print(f"\nFull metrics saved to {result_file}")

    csv_path = _save_and_display_per_class_table(metrics, model_name, output_dir, show=show)
    print(f"Per-class breakdown saved to {csv_path}")

    cm_path = _save_confusion_matrix_figure(metrics, model_name, figures_dir, show=show)
    print(f"Confusion matrix figure saved to {cm_path}")

    figure_path = _save_error_examples_figure(results, model_name, figures_dir, show=show)
    if figure_path is not None:
        print(f"Error-examples figure saved to {figure_path}")


def evaluate(
    config_path: str,
    checkpoint_path: str,
    data_root: str | None = None,
    output_dir: str = "./eval_results",
    figures_dir: str = "./figures",
    show: bool = False,
) -> ClassificationMetrics:
    """Full evaluation entry point: load checkpoint, run test inference
    once, compute all required metrics, save + display the per-class
    table, confusion matrix, and error-examples figure, print a summary.

    show: if True, displays the per-class table and figures inline (e.g.
        in a Colab/Jupyter cell), in addition to always saving them to
        disk. Call this function directly from a notebook cell (not via
        `!python -m src.evaluate`, which runs as a subprocess and can't
        render inline) to get inline display.
    """
    device = get_device()
    model, config, checkpoint = load_model_from_checkpoint(config_path, checkpoint_path, device)
    model_name = config["model"]["name"]

    print(
        f"Evaluating {model_name} on the complete test set "
        "(this should only be run once per model -- see module docstring)..."
    )
    results = _run_test_inference(model, config, device, data_root)
    metrics = compute_classification_metrics(
        results.y_true, results.y_pred, results.y_proba,
        num_classes=config["model"]["num_classes"], class_names=results.class_names,
    )
    print_summary(model_name, checkpoint, metrics)
    _save_and_report_all_outputs(metrics, results, model_name, output_dir, figures_dir, show)

    return metrics


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on the CIFAR-100 test set (Part 5)."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the model's YAML config."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the checkpoint (best.pt) to evaluate.",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Override the config's data.data_root, without editing the YAML file itself.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./eval_results",
        help="Directory to save the full metrics JSON to.",
    )
    parser.add_argument(
        "--figures-dir", type=str, default="./figures",
        help="Directory to save the error-examples figure to.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help=(
            "Display the per-class table and figures inline in addition to saving them. "
            "Only takes effect when evaluate() is called directly from a notebook cell -- "
            "running this CLI via `!python -m src.evaluate` executes as a subprocess and "
            "cannot render inline, so this flag has no visible effect there."
        ),
    )
    args = parser.parse_args()
    evaluate(
        args.config, args.checkpoint,
        data_root=args.data_root, output_dir=args.output_dir, figures_dir=args.figures_dir,
        show=args.show,
    )


if __name__ == "__main__":
    main()
