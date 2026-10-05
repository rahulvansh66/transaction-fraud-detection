# =======================================================================
# Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
# =======================================================================

"""Unit tests for experiment config validation and the shipped experiment files."""

from pathlib import Path
from typing import Any

import pytest

from src.config_loader.config_loader import load_yaml
from src.training.experiment_config import validate_experiment
from src.training.run_training_job import build_control_args

EXPERIMENTS = Path("config/experiments")
FILE_KINDS = {
    "experiment-001.yaml": "hpo",
    "experiment-002-manual-params.yaml": "manual",
    "experiment-003-search-space.yaml": "hpo",
}


def manual_cfg() -> dict[str, Any]:
    """Builds a minimal valid manual config.

    Returns:
        Config dict accepted by ``validate_experiment(cfg, "manual")``.
    """
    return {"experiment": {"name": "exp-x", "mode": "manual"},
            "static_params": {"eta": 0.1}, "manual_params": {"max_depth": 6}}


def hpo_cfg() -> dict[str, Any]:
    """Builds a minimal valid HPO config.

    Returns:
        Config dict accepted by ``validate_experiment(cfg, "hpo")``.
    """
    tuning = {"objective_metric": "validation:aucpr", "objective_type": "Maximize",
              "strategy": "Bayesian", "max_jobs": 3, "max_parallel_jobs": 2}
    return {"experiment": {"name": "exp-y", "mode": "hpo"}, "static_params": {"eta": 0.1},
            "search_space": {"max_depth": {"type": "integer", "min": 3, "max": 9}}, "tuning": tuning}


@pytest.mark.parametrize("filename,kind", FILE_KINDS.items())
def test_shipped_experiment_files_validate(filename: str, kind: str) -> None:
    """Every checked-in experiment file passes validation for the kind it declares."""
    validate_experiment(load_yaml(EXPERIMENTS / filename), kind)


def test_valid_configs_pass() -> None:
    """Minimal manual and HPO configs validate."""
    validate_experiment(manual_cfg(), "manual")
    validate_experiment(hpo_cfg(), "hpo")


def test_production_kind_is_not_validated() -> None:
    """Production configs come from model.yaml and skip experiment validation."""
    validate_experiment({}, "production")


def test_mode_must_match_kind() -> None:
    """An hpo file cannot be launched as a manual run."""
    with pytest.raises(ValueError, match="does not match"):
        validate_experiment(hpo_cfg(), "manual")


def test_missing_name_rejected() -> None:
    """experiment.name is required for lineage tagging."""
    cfg = manual_cfg()
    del cfg["experiment"]["name"]
    with pytest.raises(ValueError, match="experiment.name"):
        validate_experiment(cfg, "manual")


def test_manual_requires_manual_params() -> None:
    """A manual file without manual_params would silently train on defaults."""
    cfg = manual_cfg()
    del cfg["manual_params"]
    with pytest.raises(ValueError, match="manual_params"):
        validate_experiment(cfg, "manual")


def test_hpo_requires_search_space_and_tuning_keys() -> None:
    """An hpo file needs a non-empty search space and every tuning key."""
    cfg = hpo_cfg()
    del cfg["search_space"]
    with pytest.raises(ValueError, match="search_space"):
        validate_experiment(cfg, "hpo")
    cfg = hpo_cfg()
    del cfg["tuning"]["max_jobs"]
    with pytest.raises(ValueError, match="max_jobs"):
        validate_experiment(cfg, "hpo")


def test_params_must_not_repeat_static() -> None:
    """A parameter declared in both static_params and the run block is ambiguous."""
    cfg = manual_cfg()
    cfg["manual_params"]["eta"] = 0.3
    with pytest.raises(ValueError, match="repeats static_params"):
        validate_experiment(cfg, "manual")


def test_experiment_name_reaches_train_args() -> None:
    """The launcher forwards experiment.name so train.py can tag the run."""
    control = build_control_args(manual_cfg(), "manual", "dev", "abc123", "synthetic-v0")
    assert control["experiment-name"] == "exp-x"
