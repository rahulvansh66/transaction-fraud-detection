"""Unit tests for shared metrics and threshold selection."""

import math

import numpy as np

from src.evaluation.metrics import DEFAULT_THRESHOLD, compute_metrics, select_threshold


def test_perfect_ranking_metrics() -> None:
    """A perfect ranking gives AUPRC and ROC-AUC of 1."""
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert m["aucpr"] == 1.0 and m["roc_auc"] == 1.0
    assert m["precision"] == 1.0 and m["recall"] == 1.0


def test_single_class_is_nan() -> None:
    """Ranking metrics are undefined (NaN) with one class."""
    m = compute_metrics(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4]), 0.5)
    assert math.isnan(m["aucpr"]) and math.isnan(m["roc_auc"])


def test_select_threshold_respects_precision_floor() -> None:
    """Chosen threshold keeps precision >= floor with max recall."""
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9])
    thr = select_threshold(y, s, min_precision=1.0)
    assert thr == 0.6
    assert compute_metrics(y, s, thr)["precision"] == 1.0


def test_select_threshold_fallbacks() -> None:
    """No positives, or an unreachable floor, falls back to the default."""
    assert select_threshold(np.zeros(3), np.array([0.1, 0.2, 0.3]), 0.9) == DEFAULT_THRESHOLD
    assert select_threshold(np.array([1, 0]), np.array([0.1, 0.9]), 0.9) == DEFAULT_THRESHOLD
