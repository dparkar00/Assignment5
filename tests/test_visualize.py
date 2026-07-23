"""Unit tests for src/visualize.py.

These confirm the plotting functions run without error on valid input and
actually produce non-empty image files -- not that the plots look a
particular way, which isn't practical to assert automatically.
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visualize import (  # noqa: E402
    plot_confusion_matrices,
    plot_misclassified_examples,
    plot_training_curves,
)


def write_fake_log_csv(path: Path, num_epochs: int = 10) -> None:
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "epoch", "train_loss", "train_accuracy", "val_loss",
                "val_accuracy", "learning_rate", "epoch_duration_seconds",
            ]
        )
        for epoch in range(1, num_epochs + 1):
            writer.writerow([epoch, 4.0 - epoch * 0.3, epoch * 0.05, 4.2 - epoch * 0.28,
                              epoch * 0.045, 0.001 * (1 - epoch / num_epochs), 19.0])


class TestPlotTrainingCurves:
    def test_produces_nonempty_file(self, tmp_path):
        primary_log = tmp_path / "primary.csv"
        baseline_log = tmp_path / "baseline.csv"
        write_fake_log_csv(primary_log)
        write_fake_log_csv(baseline_log)

        output_path = tmp_path / "curves.png"
        result = plot_training_curves(
            {"Swin": str(primary_log), "ViT": str(baseline_log)}, str(output_path)
        )

        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_works_with_single_model(self, tmp_path):
        log_path = tmp_path / "only.csv"
        write_fake_log_csv(log_path)
        output_path = tmp_path / "curves_single.png"

        result = plot_training_curves({"Swin": str(log_path)}, str(output_path))
        assert Path(result).exists()


class TestPlotConfusionMatrices:
    def test_produces_nonempty_file(self, tmp_path):
        cm = np.random.randint(0, 20, size=(10, 10))
        np.fill_diagonal(cm, 200)
        output_path = tmp_path / "cm.png"

        result = plot_confusion_matrices({"Swin": cm.tolist(), "ViT": cm.tolist()}, str(output_path))

        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_works_with_single_model(self, tmp_path):
        cm = np.eye(5, dtype=int) * 10
        output_path = tmp_path / "cm_single.png"

        result = plot_confusion_matrices({"Swin": cm.tolist()}, str(output_path))
        assert Path(result).exists()


class TestPlotMisclassifiedExamples:
    def test_produces_nonempty_file_for_twelve_examples(self, tmp_path):
        examples = [
            {
                "image": np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8),
                "true_label": f"class{i}",
                "predicted_label": f"class{i + 1}",
                "predicted_probability": 0.7,
            }
            for i in range(12)
        ]
        output_path = tmp_path / "errors.png"

        result = plot_misclassified_examples(examples, str(output_path))

        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_handles_non_grid_multiple_count(self, tmp_path):
        # 5 examples with 4 columns -> incomplete last row; must not crash.
        examples = [
            {
                "image": np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8),
                "true_label": "a",
                "predicted_label": "b",
                "predicted_probability": 0.5,
            }
            for _ in range(5)
        ]
        output_path = tmp_path / "errors_partial.png"

        result = plot_misclassified_examples(examples, str(output_path), columns=4)
        assert Path(result).exists()
