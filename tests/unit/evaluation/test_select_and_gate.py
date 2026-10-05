# =======================================================================
# Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
# =======================================================================

"""Unit tests for winner selection (validation only) and the promotion gate."""

import pytest

from src.evaluation.evaluate import apply_gate
from src.evaluation.select_winner import comparison_table, pick_best, varying_params

GATE = {"min_aucpr": 0.5, "min_recall": 0.3, "no_worse_than_production": True}


def _run(run_id: str, aucpr: float | None) -> dict:
    """Builds a fake MLflow run record."""
    metrics = {} if aucpr is None else {"validation_aucpr": aucpr}
    return {"run_id": run_id, "metrics": metrics, "params": {"max_depth": "5"}}


def test_pick_best_takes_max_validation_metric() -> None:
    """The highest validation AUPRC wins; runs without the metric are skipped."""
    runs = [_run("aaaaaaaa1", 0.7), _run("bbbbbbbb2", 0.9), _run("cccccccc3", None)]
    assert pick_best(runs, "validation_aucpr")["run_id"] == "bbbbbbbb2"


def test_pick_best_ignores_nan_and_raises_when_empty() -> None:
    """NaN scores are not eligible and an empty field is an error."""
    assert pick_best([_run("a", float("nan")), _run("b", 0.1)], "validation_aucpr")["run_id"] == "b"
    with pytest.raises(ValueError):
        pick_best([_run("a", None)], "validation_aucpr")


def test_comparison_table_is_ranked() -> None:
    """The first data row is the best trial."""
    table = comparison_table([_run("lowlowlow", 0.1), _run("highhighh", 0.9)], "validation_aucpr")
    assert table.splitlines()[2].startswith("| highhigh")


def test_comparison_table_shows_recall_and_only_varying_params() -> None:
    """Columns are the params that differ between trials, plus validation recall."""
    runs = [{"run_id": "aaaaaaaa1", "metrics": {"validation_aucpr": 0.7, "validation_recall": 0.5},
             "params": {"max_depth": "2", "eta": "0.1", "best_iteration": "10"}},
            {"run_id": "bbbbbbbb2", "metrics": {"validation_aucpr": 0.9},
             "params": {"max_depth": "5", "eta": "0.1", "best_iteration": "30"}}]
    assert varying_params(runs) == ["max_depth"]
    lines = comparison_table(runs, "validation_aucpr").splitlines()
    assert lines[0] == "| run_id | validation_aucpr | validation_recall | max_depth |"
    assert lines[2] == "| bbbbbbbb | 0.9000 |  | 5 |" and lines[3] == "| aaaaaaaa | 0.7000 | 0.5000 | 2 |"


def test_gate_absolute_and_relative_checks() -> None:
    """Passes above the floors, fails below them or below production."""
    good = {"aucpr": 0.8, "recall": 0.6}
    assert apply_gate(good, GATE, None) == (True, [])
    assert apply_gate({"aucpr": 0.4, "recall": 0.6}, GATE, None)[0] is False
    assert apply_gate({"aucpr": 0.8, "recall": 0.1}, GATE, None)[0] is False
    passed, reasons = apply_gate(good, GATE, {"aucpr": 0.9})
    assert not passed and "production" in reasons[0]
