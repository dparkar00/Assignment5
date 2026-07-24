"""Classification metrics for Part 5 (Evaluation and Experimental Analysis).

Computes exactly the metrics the assignment requires: top-1 accuracy,
macro-averaged precision/recall/F1, weighted F1, per-class precision/
recall/F1, a confusion matrix, and one-vs-rest macro ROC-AUC computed from
predicted class *probabilities* (not hard predictions, per the assignment's
explicit instruction).

Uses scikit-learn's implementations rather than reimplementing this math by
hand -- precision/recall/F1/ROC-AUC have enough edge cases (zero-support
classes, multi-class averaging conventions) that a well-tested library
implementation is the right call here, not a from-scratch one.
"""

from __future__ import annotations

import csv
import dataclasses

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclasses.dataclass
class ClassificationMetrics:
    """All Part-5-required metrics for one model's test-set evaluation."""

    top1_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    roc_auc_macro_ovr: float
    per_class_precision: list
    per_class_recall: list
    per_class_f1: list
    per_class_support: list
    confusion_matrix: list
    class_names: list

    def to_dict(self) -> dict:
        """Convert to a plain dict, ready for json.dump."""
        return dataclasses.asdict(self)


def _compute_averaged_scores(y_true: np.ndarray, y_pred: np.ndarray, labels: list) -> tuple:
    """Macro precision/recall/F1 and weighted F1 -- the four averaged
    (non-per-class) score types the assignment requires.
    """
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    return float(macro_precision), float(macro_recall), float(macro_f1), float(weighted_f1)


def _compute_per_class_scores(y_true: np.ndarray, y_pred: np.ndarray, labels: list) -> tuple:
    """Per-class precision/recall/F1/support, as plain lists ready for JSON."""
    precision, recall, f1_score, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return precision.tolist(), recall.tolist(), f1_score.tolist(), support.tolist()


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    num_classes: int,
    class_names: list | None = None,
) -> ClassificationMetrics:
    """Compute every metric Part 5 requires from one evaluation pass.

    y_true: (N,) int array of ground-truth labels.
    y_pred: (N,) int array of hard predictions (argmax of y_proba).
    y_proba: (N, num_classes) float array of predicted class probabilities
        (softmax outputs) -- required for ROC-AUC, which must be computed
        from probabilities, not from y_pred.
    num_classes: total number of classes (100 for CIFAR-100). Passed
        explicitly (rather than inferred from the data) so classes with
        zero test-set predictions still appear in the per-class breakdown
        and confusion matrix, instead of silently being dropped.
    """
    labels = list(range(num_classes))
    if class_names is None:
        class_names = [str(i) for i in labels]

    averaged = _compute_averaged_scores(y_true, y_pred, labels)
    per_class = _compute_per_class_scores(y_true, y_pred, labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    roc_auc_macro_ovr = float(
        roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=labels)
    )

    return ClassificationMetrics(
        top1_accuracy=float((y_pred == y_true).mean()),
        macro_precision=averaged[0],
        macro_recall=averaged[1],
        macro_f1=averaged[2],
        weighted_f1=averaged[3],
        roc_auc_macro_ovr=roc_auc_macro_ovr,
        per_class_precision=per_class[0],
        per_class_recall=per_class[1],
        per_class_f1=per_class[2],
        per_class_support=per_class[3],
        confusion_matrix=cm,
        class_names=class_names,
    )


def find_confusion_patterns(metrics: ClassificationMetrics, n: int = 3) -> list:
    """Identify the n most common confusion patterns (true class, predicted
    class, count) for the Part 5 "Class-Level Error Analysis" section,
    which requires at least three documented confusion patterns.

    Only off-diagonal cells count as confusions (a correct prediction isn't
    a "pattern of confusion"). Ties are broken by (true, predicted) index
    order for reproducibility.
    """
    cm = np.array(metrics.confusion_matrix)
    num_classes = cm.shape[0]

    confusions = []
    for true_idx in range(num_classes):
        for pred_idx in range(num_classes):
            if true_idx == pred_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            if count > 0:
                confusions.append((true_idx, pred_idx, count))

    confusions.sort(key=lambda item: (-item[2], item[0], item[1]))

    return [
        {
            "true_class": metrics.class_names[true_idx],
            "predicted_class": metrics.class_names[pred_idx],
            "count": count,
        }
        for true_idx, pred_idx, count in confusions[:n]
    ]


def top_and_bottom_classes(metrics: ClassificationMetrics, n: int = 5) -> dict:
    """Identify the n highest- and n lowest-F1 classes by name, for the
    Part 5 "Class-Level Error Analysis" section (five best / five worst).
    """
    f1_by_class = list(zip(metrics.class_names, metrics.per_class_f1))
    f1_by_class.sort(key=lambda pair: pair[1], reverse=True)
    return {
        "highest_f1": f1_by_class[:n],
        "lowest_f1": f1_by_class[-n:][::-1],
    }


def save_per_class_csv(metrics: ClassificationMetrics, output_path: str) -> str:
    """Write the full per-class precision/recall/F1/support breakdown to a
    CSV, sorted by F1 descending -- readable directly (Excel, pandas, or a
    text editor) rather than requiring the caller to parse the full JSON.
    """
    rows = sorted(
        zip(
            metrics.class_names, metrics.per_class_precision,
            metrics.per_class_recall, metrics.per_class_f1, metrics.per_class_support,
        ),
        key=lambda row: row[3],
        reverse=True,
    )
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["class_name", "precision", "recall", "f1_score", "support"])
        writer.writerows(rows)
    return output_path