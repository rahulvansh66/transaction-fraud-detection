# =======================================================================
# Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
# =======================================================================

"""Unit tests for experiment validation, strategy rules and the shipped experiment files."""

import copy
from pathlib import Path
from typing import Any

import pytest

from src.config_loader.config_loader import load_yaml
from src.training.experiment_config import (
    SINGLE,
    classify_params,
    validate_experiment,
    validate_production,
)
from src.training.run_training_job import build_control_args, run_mode

EXPERIMENTS = Path("config/experiments")
EXPECTED_STRATEGY = {
    "experiment-001.yaml": "bayesian",
    "experiment-002-manual-params.yaml": SINGLE,
    "experiment-003-search-space.yaml": "bayesian",
    "experiment-004-depth-sensitivity.yaml": "grid",
}
RANGE = {"type": "integer", "min": 3, "max": 9}


def make_cfg(params: dict[str, Any] | None = None, tuning: dict[str, Any] | None = None) -> dict[str, Any]:
    """Builds a minimal valid config.

    Args:
        params: Extra params merged over ``{"eta": 0.1}``.
        tuning: Tuning block, or ``None`` for a single run.

    Returns:
        Config dict accepted by ``validate_experiment``.
    """
    cfg: dict[str, Any] = {"experiment": {"name": "exp-x"}, "params": {"eta": 0.1, **(params or {})}}
    if tuning is not None:
        cfg["tuning"] = {"objective_metric": "validation:aucpr", "objective_type": "Maximize",
                         "max_parallel_jobs": 2, **tuning}
    return cfg


@pytest.mark.parametrize("filename,strategy", EXPECTED_STRATEGY.items())
def test_shipped_experiment_files_validate(filename: str, strategy: str) -> None:
    """Every checked-in experiment file validates and resolves to its documented strategy."""
    assert validate_experiment(load_yaml(EXPERIMENTS / filename)).strategy == strategy


def test_depth_sensitivity_grid_has_three_trials() -> None:
    """Experiment 004 is a 3-point grid over max_depth with everything else fixed."""
    plan = validate_experiment(load_yaml(EXPERIMENTS / "experiment-004-depth-sensitivity.yaml"))
    assert plan.lists == {"max_depth": [2, 5, 7]} and plan.combinations == 3 and plan.trials == 3
    assert "max_depth" not in plan.fixed


def test_all_scalars_is_a_single_run() -> None:
    """Scalars only means one training job and no tuning block."""
    plan = validate_experiment(make_cfg({"max_depth": 6}))
    assert plan.strategy == SINGLE and plan.fixed == {"eta": 0.1, "max_depth": 6} and plan.trials == 1


def test_stray_tuning_block_rejected() -> None:
    """A tuning block with nothing to tune is ambiguous intent."""
    with pytest.raises(ValueError, match="nothing|no list or range"):
        validate_experiment(make_cfg({"max_depth": 6}, {"strategy": "grid"}))


def test_tunable_params_need_strategy() -> None:
    """Lists or ranges without a tuning block or strategy are rejected, never guessed."""
    with pytest.raises(ValueError, match="tuning block"):
        validate_experiment(make_cfg({"max_depth": [2, 5]}))
    cfg = make_cfg({"max_depth": [2, 5]}, {"strategy": "grid"})
    del cfg["tuning"]["strategy"]
    with pytest.raises(ValueError, match="strategy"):
        validate_experiment(cfg)


def test_unknown_strategy_rejected() -> None:
    """Only grid, random and bayesian are accepted, case-insensitively."""
    with pytest.raises(ValueError, match="not one of"):
        validate_experiment(make_cfg({"max_depth": [2, 5]}, {"strategy": "hyperband"}))
    assert validate_experiment(make_cfg({"max_depth": [2, 5]}, {"strategy": "Grid"})).strategy == "grid"


def test_grid_rejects_ranges_and_mismatched_max_jobs() -> None:
    """Grid supports lists only, and max_jobs (if set) must equal the combination count."""
    with pytest.raises(ValueError, match="lists only"):
        validate_experiment(make_cfg({"max_depth": RANGE}, {"strategy": "grid"}))
    with pytest.raises(ValueError, match="2 combinations"):
        validate_experiment(make_cfg({"max_depth": [2, 5]}, {"strategy": "grid", "max_jobs": 3}))
    plan = validate_experiment(make_cfg({"max_depth": [2, 5], "gamma": [0, 1, 2]}, {"strategy": "grid"}))
    assert plan.combinations == 6 and plan.trials == 6
    assert validate_experiment(make_cfg({"max_depth": [2, 5]}, {"strategy": "grid", "max_jobs": 2})).trials == 2


@pytest.mark.parametrize("strategy", ["random", "bayesian"])
def test_random_and_bayesian_need_max_jobs_and_allow_mixed(strategy: str) -> None:
    """Sampling strategies require a budget and accept lists, ranges or both."""
    params = {"max_depth": [2, 5, 7], "gamma": {"type": "continuous", "min": 0, "max": 3}}
    with pytest.raises(ValueError, match="max_jobs is required"):
        validate_experiment(make_cfg(params, {"strategy": strategy}))
    plan = validate_experiment(make_cfg(params, {"strategy": strategy, "max_jobs": 4}))
    assert plan.strategy == strategy and plan.trials == 4 and plan.tunable == {"max_depth", "gamma"}


def test_unknown_top_level_key_rejected() -> None:
    """A mis-indented parameter fails instead of being silently ignored."""
    cfg = make_cfg()
    cfg["gamma"] = {"type": "continuous", "min": 0, "max": 3}
    with pytest.raises(ValueError, match="unknown top-level keys: gamma"):
        validate_experiment(cfg)


def test_missing_name_and_params_rejected() -> None:
    """experiment.name tags every run and params must not be empty."""
    cfg = make_cfg()
    del cfg["experiment"]["name"]
    with pytest.raises(ValueError, match="experiment.name"):
        validate_experiment(cfg)
    cfg = make_cfg()
    cfg["params"] = {}
    with pytest.raises(ValueError, match="params"):
        validate_experiment(cfg)


def test_missing_tuning_keys_rejected() -> None:
    """Every required tuning key is checked before submission."""
    cfg = make_cfg({"max_depth": [2, 5]}, {"strategy": "grid"})
    del cfg["tuning"]["objective_metric"]
    with pytest.raises(ValueError, match="objective_metric"):
        validate_experiment(cfg)


@pytest.mark.parametrize("bad", [[], [2, 2], [2, True], [2, None], [[1], 2], None])
def test_bad_list_values_rejected(bad: Any) -> None:
    """Empty, duplicated, bool, null and nested entries are refused."""
    with pytest.raises(ValueError):
        classify_params({"max_depth": bad})


def test_one_element_list_collapses_to_fixed() -> None:
    """A one-point list is a fixed value, not a degenerate search."""
    fixed, lists, ranges = classify_params({"max_depth": [6], "eta": 0.1})
    assert fixed == {"max_depth": 6, "eta": 0.1} and not lists and not ranges


@pytest.mark.parametrize("spec", [
    {"min": 0, "max": 3},
    {"type": "weird", "min": 0, "max": 1},
    {"type": "integer", "min": 5, "max": 5},
    {"type": "integer", "min": 1.5, "max": 5},
    {"type": "continuous", "min": 0, "max": 1, "scaling": "Logarithmic"},
    {"type": "continuous", "min": 0, "max": 1, "scaling": "Weird"},
    {"type": "continuous", "min": 0, "max": 1, "step": 0.1},
    {"type": "continuous", "min": "a", "max": 1},
])
def test_bad_range_specs_rejected(spec: dict[str, Any]) -> None:
    """Missing or unknown type, bad bounds, scaling and stray keys fail at launch."""
    with pytest.raises(ValueError):
        classify_params({"x": spec})


def test_range_with_continuous_int_bounds_is_valid() -> None:
    """Whole-number bounds are fine for a continuous range (gamma 0..3)."""
    _, _, ranges = classify_params({"gamma": {"type": "continuous", "min": 0, "max": 3}})
    assert "gamma" in ranges


def test_production_params_must_be_scalars_and_non_empty() -> None:
    """Production retrains one approved point; empty or searchable params are errors."""
    assert validate_production({"params": {"eta": 0.1}}) == {"eta": 0.1}
    with pytest.raises(ValueError, match="empty"):
        validate_production({"params": {}})
    with pytest.raises(ValueError, match="scalars"):
        validate_production({"params": {"max_depth": [2, 5]}})


def test_run_mode_and_experiment_name_reach_train_args() -> None:
    """The launcher tags the strategy as mode and forwards experiment.name for MLflow tagging."""
    cfg = make_cfg({"max_depth": [2, 5]}, {"strategy": "grid"})
    plan = validate_experiment(copy.deepcopy(cfg))
    assert run_mode("experiment", plan) == "grid" and run_mode("production", plan) == "production"
    control = build_control_args(cfg, "grid", "dev", "abc123", "synthetic-v0")
    assert control["experiment-name"] == "exp-x" and control["mode"] == "grid"
