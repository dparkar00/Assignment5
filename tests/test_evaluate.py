"""Unit tests for src/evaluate.py.

build_test_loader() (and therefore the full evaluate() entry point) needs
the real CIFAR-100 download and is not covered here -- that's an
integration concern that has to run somewhere with network access. What's
tested here is everything that doesn't require real data: checkpoint
loading, the inference loop, and the full metrics computation round-trip
using a tiny model and synthetic data.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import load_model_from_checkpoint, print_summary, run_inference  # noqa: E402
from src.metrics import compute_classification_metrics  # noqa: E402
from src.train import build_model, save_checkpoint  # noqa: E402


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


def make_synthetic_test_loader(num_samples: int = 40, num_classes: int = 10, batch_size: int = 8):
    images = torch.randn(num_samples, 3, 32, 32)
    targets = torch.randint(0, num_classes, (num_samples,))
    return DataLoader(TensorDataset(images, targets), batch_size=batch_size, shuffle=False)


class TestRunInference:
    def test_output_shapes_and_types(self):
        model = build_model(VIT_MODEL_CONFIG)
        loader = make_synthetic_test_loader(num_samples=40, num_classes=10)

        y_true, y_pred, y_proba = run_inference(model, loader, torch.device("cpu"))

        assert y_true.shape == (40,)
        assert y_pred.shape == (40,)
        assert y_proba.shape == (40, 10)
        assert isinstance(y_true, np.ndarray)

    def test_probabilities_sum_to_one_per_row(self):
        model = build_model(VIT_MODEL_CONFIG)
        loader = make_synthetic_test_loader(num_samples=16, num_classes=10)

        _, _, y_proba = run_inference(model, loader, torch.device("cpu"))

        assert np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predictions_match_argmax_of_probabilities(self):
        model = build_model(VIT_MODEL_CONFIG)
        loader = make_synthetic_test_loader(num_samples=16, num_classes=10)

        _, y_pred, y_proba = run_inference(model, loader, torch.device("cpu"))

        assert np.array_equal(y_pred, y_proba.argmax(axis=1))

    def test_does_not_update_model_parameters(self):
        model = build_model(VIT_MODEL_CONFIG)
        loader = make_synthetic_test_loader(num_samples=16, num_classes=10)
        params_before = [p.clone() for p in model.parameters()]

        run_inference(model, loader, torch.device("cpu"))

        params_after = list(model.parameters())
        assert all(torch.equal(a, b) for a, b in zip(params_before, params_after))


class TestLoadModelFromCheckpoint(object):
    def test_round_trip_loads_matching_weights(self, tmp_path):
        model = build_model(VIT_MODEL_CONFIG)
        checkpoint_path = tmp_path / "best.pt"
        save_checkpoint(model, epoch=5, val_accuracy=0.42, path=checkpoint_path)

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
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
"""
        )

        loaded_model, config, checkpoint = load_model_from_checkpoint(
            str(config_path), str(checkpoint_path), torch.device("cpu")
        )

        assert config["model"]["name"] == "vit"
        assert checkpoint["epoch"] == 5
        assert checkpoint["val_accuracy"] == 0.42

        # Confirm the loaded weights actually match the original model's,
        # not just that loading didn't crash.
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            original_output = model(x)
            loaded_output = loaded_model(x)
        assert torch.allclose(original_output, loaded_output, atol=1e-6)


class TestCollectMisclassifiedExamples:
    def test_selects_only_incorrect_predictions(self):
        from src.evaluate import collect_misclassified_examples

        num_classes = 4
        raw_images = np.random.randint(0, 255, size=(6, 32, 32, 3), dtype=np.uint8)
        y_true = np.array([0, 1, 2, 3, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 0, 2])  # indices 3, 5 are wrong
        y_proba = np.eye(num_classes)[y_pred]
        class_names = [f"class{i}" for i in range(num_classes)]

        examples = collect_misclassified_examples(
            raw_images, y_true, y_pred, y_proba, class_names, n=12
        )

        assert len(examples) == 2
        true_labels = {ex["true_label"] for ex in examples}
        assert true_labels == {"class3", "class1"}

    def test_most_confident_incorrect_rule_sorts_by_confidence(self):
        from src.evaluate import collect_misclassified_examples

        num_classes = 3
        raw_images = np.random.randint(0, 255, size=(3, 32, 32, 3), dtype=np.uint8)
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])  # all wrong
        # Confidences: 0.5, 0.9, 0.7 -- expect order index 1, 2, 0.
        y_proba = np.array([[0.5, 0.5, 0.0], [0.1, 0.9, 0.0], [0.3, 0.7, 0.0]])
        class_names = [f"class{i}" for i in range(num_classes)]

        examples = collect_misclassified_examples(
            raw_images, y_true, y_pred, y_proba, class_names, n=3, rule="most_confident_incorrect"
        )

        confidences = [ex["predicted_probability"] for ex in examples]
        assert confidences == sorted(confidences, reverse=True)
        assert confidences[0] == 0.9

    def test_returns_empty_list_when_nothing_misclassified(self):
        from src.evaluate import collect_misclassified_examples

        num_classes = 2
        raw_images = np.random.randint(0, 255, size=(3, 32, 32, 3), dtype=np.uint8)
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1, 0])  # perfect
        y_proba = np.eye(num_classes)[y_pred]
        class_names = ["a", "b"]

        examples = collect_misclassified_examples(raw_images, y_true, y_pred, y_proba, class_names)
        assert examples == []

    def test_invalid_rule_raises(self):
        from src.evaluate import collect_misclassified_examples

        num_classes = 2
        raw_images = np.random.randint(0, 255, size=(2, 32, 32, 3), dtype=np.uint8)
        y_true = np.array([0, 1])
        y_pred = np.array([1, 0])
        y_proba = np.eye(num_classes)[y_pred]
        class_names = ["a", "b"]

        import pytest

        with pytest.raises(ValueError, match="Unknown selection rule"):
            collect_misclassified_examples(
                raw_images, y_true, y_pred, y_proba, class_names, rule="not_a_real_rule"
            )
class TestFullEvaluationRoundTrip:
    def test_synthetic_end_to_end_produces_sane_metrics(self, tmp_path, capsys):
        """Exercises the real pipeline (checkpoint -> inference -> metrics
        -> print_summary) on synthetic data, standing in for the parts of
        evaluate() that don't require the real CIFAR-100 download.
        """
        num_classes = 10
        model = build_model({**VIT_MODEL_CONFIG, "num_classes": num_classes})
        checkpoint_path = tmp_path / "best.pt"
        save_checkpoint(model, epoch=10, val_accuracy=0.55, path=checkpoint_path)
        checkpoint = torch.load(checkpoint_path, weights_only=True)

        loader = make_synthetic_test_loader(num_samples=50, num_classes=num_classes)
        y_true, y_pred, y_proba = run_inference(model, loader, torch.device("cpu"))

        metrics = compute_classification_metrics(y_true, y_pred, y_proba, num_classes=num_classes)

        # Sanity bounds -- not exact values, since data/model are random.
        assert 0.0 <= metrics.top1_accuracy <= 1.0
        assert 0.0 <= metrics.macro_f1 <= 1.0
        assert len(metrics.per_class_precision) == num_classes
        assert len(metrics.confusion_matrix) == num_classes
        assert len(metrics.confusion_matrix[0]) == num_classes

        print_summary("vit", checkpoint, metrics)
        captured = capsys.readouterr()
        assert "Top-1 accuracy" in captured.out
        assert "Top 5 classes by F1" in captured.out
