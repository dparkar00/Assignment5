"""Unit tests for src/metrics.py."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import compute_classification_metrics, top_and_bottom_classes  # noqa: E402


class TestPerfectPredictions:
    def test_perfect_predictions_give_perfect_scores(self):
        num_classes = 4
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = y_true.copy()
        # Confident, correct probabilities: put ~1.0 on the true class.
        y_proba = np.full((8, num_classes), 0.01)
        y_proba[np.arange(8), y_true] = 0.97

        metrics = compute_classification_metrics(y_true, y_pred, y_proba, num_classes)

        assert metrics.top1_accuracy == 1.0
        assert metrics.macro_precision == 1.0
        assert metrics.macro_recall == 1.0
        assert metrics.macro_f1 == 1.0
        assert metrics.weighted_f1 == 1.0
        assert metrics.roc_auc_macro_ovr == 1.0


class TestKnownConfusionPattern:
    @staticmethod
    def _fake_probabilities(y_pred, num_classes):
        """Build normalized fake probabilities that peak at y_pred (a valid
        stand-in for real softmax output, which always sums to 1 per row).
        """
        proba = np.eye(num_classes)[y_pred] * 0.9 + 0.05
        return proba / proba.sum(axis=1, keepdims=True)

    def test_matches_hand_computed_accuracy(self):
        # 3 classes, 6 samples, 4 correct -> accuracy = 4/6
        num_classes = 3
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])  # 4 correct: idx 0,2,3,4
        y_proba = self._fake_probabilities(y_pred, num_classes)

        metrics = compute_classification_metrics(y_true, y_pred, y_proba, num_classes)

        assert abs(metrics.top1_accuracy - 4 / 6) < 1e-9

    def test_confusion_matrix_shape_and_diagonal(self):
        num_classes = 3
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        y_proba = self._fake_probabilities(y_pred, num_classes)

        metrics = compute_classification_metrics(y_true, y_pred, y_proba, num_classes)
        cm = np.array(metrics.confusion_matrix)

        assert cm.shape == (num_classes, num_classes)
        # Class 0: one correct (pred=0), one wrong (pred=1) -> row [1,1,0]
        assert list(cm[0]) == [1, 1, 0]
        # Class 2: one correct (pred=2), one wrong (pred=0) -> row [1,0,1]
        assert list(cm[2]) == [1, 0, 1]


class TestZeroSupportClassHandling:
    def test_class_absent_from_predictions_does_not_crash(self):
        # Class 2 never appears in y_true or y_pred, but num_classes=3 --
        # must not crash, and should report zero for that class rather
        # than silently dropping it from the per-class arrays.
        num_classes = 3
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        y_proba = np.column_stack(
            [
                np.where(y_pred == 0, 0.9, 0.05),
                np.where(y_pred == 1, 0.9, 0.05),
                np.full(6, 0.05),
            ]
        )
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

        metrics = compute_classification_metrics(y_true, y_pred, y_proba, num_classes)

        assert len(metrics.per_class_precision) == num_classes
        assert len(metrics.per_class_recall) == num_classes
        assert len(metrics.per_class_f1) == num_classes
        assert metrics.per_class_support[2] == 0


class TestClassNames:
    def test_default_class_names_are_string_indices(self):
        num_classes = 3
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        y_proba = np.eye(num_classes)

        metrics = compute_classification_metrics(y_true, y_pred, y_proba, num_classes)
        assert metrics.class_names == ["0", "1", "2"]

    def test_custom_class_names_preserved(self):
        num_classes = 3
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        y_proba = np.eye(num_classes)
        names = ["apple", "bicycle", "cloud"]

        metrics = compute_classification_metrics(y_true, y_pred, y_proba, num_classes, class_names=names)
        assert metrics.class_names == names


class TestFindConfusionPatterns:
    def test_identifies_most_common_confusion(self):
        num_classes = 4
        class_names = ["cat", "dog", "bird", "fish"]
        y_true = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3])
        y_pred = np.array([1, 1, 1, 0, 1, 1, 3, 3, 2, 3, 0])
        y_proba = np.eye(num_classes)[y_pred]

        from src.metrics import find_confusion_patterns

        metrics = compute_classification_metrics(
            y_true, y_pred, y_proba, num_classes, class_names=class_names
        )
        patterns = find_confusion_patterns(metrics, n=3)

        assert len(patterns) == 3
        assert patterns[0]["true_class"] == "cat"
        assert patterns[0]["predicted_class"] == "dog"
        assert patterns[0]["count"] == 3

    def test_excludes_correct_predictions(self):
        from src.metrics import find_confusion_patterns

        num_classes = 3
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])  # all correct, zero confusions
        y_proba = np.eye(num_classes)[y_pred]

        metrics = compute_classification_metrics(y_true, y_pred, y_proba, num_classes)
        patterns = find_confusion_patterns(metrics, n=3)

        assert patterns == []

    def test_returns_fewer_than_n_when_not_enough_confusions_exist(self):
        from src.metrics import find_confusion_patterns

        num_classes = 3
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 0, 1, 1])  # exactly one confusion: 0->1
        y_proba = np.eye(num_classes)[y_pred]

        metrics = compute_classification_metrics(y_true, y_pred, y_proba, num_classes)
        patterns = find_confusion_patterns(metrics, n=3)

        assert len(patterns) == 1


class TestSavePerClassCsv:
    def test_writes_readable_csv_sorted_by_f1(self, tmp_path):
        from src.metrics import save_per_class_csv

        num_classes = 3
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 0, 2, 2])  # class 0 perfect, class 2 perfect, class 1 worse
        y_proba = np.eye(num_classes)[y_pred]
        class_names = ["cat", "dog", "bird"]

        metrics = compute_classification_metrics(y_true, y_pred, y_proba, num_classes, class_names=class_names)
        output_path = save_per_class_csv(metrics, str(tmp_path / "per_class.csv"))

        import csv as csv_module

        with open(output_path, newline="", encoding="utf-8") as f:
            rows = list(csv_module.DictReader(f))

        assert len(rows) == num_classes
        assert set(rows[0].keys()) == {"class_name", "precision", "recall", "f1_score", "support"}
        f1_values = [float(row["f1_score"]) for row in rows]
        assert f1_values == sorted(f1_values, reverse=True)


class TestTopAndBottomClasses:
    def test_identifies_highest_and_lowest_f1(self):
        num_classes = 5
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        # Class 0 perfect, class 4 always wrong, others mixed.
        y_pred = np.array([0, 0, 1, 0, 2, 3, 3, 2, 0, 1])
        y_proba = np.eye(num_classes)[y_pred]

        metrics = compute_classification_metrics(y_true, y_pred, y_proba, num_classes)
        extremes = top_and_bottom_classes(metrics, n=2)

        assert len(extremes["highest_f1"]) == 2
        assert len(extremes["lowest_f1"]) == 2
        # Class "0" (name defaults to str(index)) should be among the best,
        # since every class-0 sample was predicted correctly.
        highest_names = [name for name, _ in extremes["highest_f1"]]
        assert "0" in highest_names

    def test_to_dict_is_json_serializable(self):
        import json

        num_classes = 3
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        y_proba = np.eye(num_classes)

        metrics = compute_classification_metrics(y_true, y_pred, y_proba, num_classes)
        serialized = json.dumps(metrics.to_dict())
        assert isinstance(serialized, str)