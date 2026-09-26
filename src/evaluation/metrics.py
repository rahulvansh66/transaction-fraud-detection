"""Shared classification metrics and threshold selection.

Used by both training (validation) and evaluation (test) so "good" has a single
definition. Imports only numpy and scikit-learn to keep the training image light.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_THRESHOLD = 0.5


def compute_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict[str, float]:
    """Computes AUPRC, ROC-AUC and precision/recall/F1 at a threshold.

    Ranking metrics are NaN when ``y_true`` holds a single class (they are undefined).

    Args:
        y_true: Binary labels.
        y_score: Predicted fraud probabilities.
        threshold: Decision threshold applied to ``y_score``.

    Returns:
        Mapping of metric name to value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_score) >= threshold
    single_class = len(np.unique(y_true)) < 2
    return {
        "aucpr": float("nan") if single_class else float(average_precision_score(y_true, y_score)),
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def select_threshold(
    y_true: np.ndarray, y_score: np.ndarray, min_precision: float
) -> float:
    """Picks the threshold with the best recall whose precision is at least ``min_precision``.

    Must be called on VALIDATION data only. Falls back to 0.5 when there are no
    positives or no threshold reaches the precision floor.

    Args:
        y_true: Binary validation labels.
        y_score: Predicted fraud probabilities.
        min_precision: Minimum acceptable precision.

    Returns:
        The chosen decision threshold.
    """
    y_true = np.asarray(y_true)
    if y_true.sum() == 0:
        return DEFAULT_THRESHOLD
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall have one more entry than thresholds; drop the sentinel point
    ok = precision[:-1] >= min_precision
    if not ok.any():
        return DEFAULT_THRESHOLD
    best = np.argmax(np.where(ok, recall[:-1], -1.0))
    return float(thresholds[best])
