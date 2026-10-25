"""NumPy-only classification metrics with an explicit rejection column."""

from typing import Any, Dict

import numpy as np


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4
) -> Dict[str, Any]:
    """Count rejected predictions (-1) as false negatives of their true class.

    The confusion matrix has num_classes rows and num_classes + 1 columns;
    its final column contains rejected samples. Empty inputs yield zeros.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    truth = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if truth.ndim != 1 or predicted.ndim != 1 or truth.shape != predicted.shape:
        raise ValueError("y_true and y_pred must be one-dimensional with equal length")
    for values in (truth, predicted):
        if not np.isfinite(values).all() or np.any(values != np.floor(values)):
            raise ValueError("class labels must be finite integers")
    truth = truth.astype(np.int64)
    predicted = predicted.astype(np.int64)
    if np.any((truth < 0) | (truth >= num_classes)):
        raise ValueError("y_true contains an out-of-range class")
    if np.any((predicted < -1) | (predicted >= num_classes)):
        raise ValueError("y_pred contains an out-of-range class")

    confusion = np.zeros((num_classes, num_classes + 1), dtype=np.int64)
    columns = np.where(predicted == -1, num_classes, predicted)
    np.add.at(confusion, (truth, columns), 1)
    true_positive = np.diag(confusion[:, :num_classes]).astype(np.float64)
    actual = confusion.sum(axis=1)
    positive = confusion[:, :num_classes].sum(axis=0)
    recall = np.divide(true_positive, actual, out=np.zeros(num_classes), where=actual > 0)
    precision = np.divide(
        true_positive, positive, out=np.zeros(num_classes), where=positive > 0
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes),
        where=(precision + recall) > 0,
    )
    total = len(truth)
    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_f1": float(f1.mean()),
        "per_class_recall": recall.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_f1": f1.tolist(),
        "support": actual.tolist(),
        "confusion_matrix": confusion.tolist(),
        "coverage": float(np.count_nonzero(predicted != -1) / total) if total else 0.0,
        "num_samples": total,
    }
