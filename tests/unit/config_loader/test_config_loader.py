"""Unit tests for config merging and loading helpers."""

from pathlib import Path

import pytest

from src.config_loader.config_loader import load_yaml, merge_configs
from src.training.lineage import config_hash

EXPERIMENT = Path("config/experiments/experiment-001.yaml")


def test_merge_configs_recursive_and_pure() -> None:
    """Nested keys merge, overrides win, inputs are not mutated."""
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    out = merge_configs(base, {"a": {"y": 9}, "c": 3})
    assert out == {"a": {"x": 1, "y": 9}, "b": 1, "c": 3}
    assert base == {"a": {"x": 1, "y": 2}, "b": 1}


def test_config_hash_is_stable() -> None:
    """Same content in a different key order gives the same hash."""
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_load_yaml_missing_raises() -> None:
    """A missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_yaml("config/experiments/nope.yaml")


def test_experiment_has_no_environment_values() -> None:
    """Experiment configs must not hold buckets, ARNs or tracking URIs."""
    text = EXPERIMENT.read_text(encoding="utf-8")
    for forbidden in ("arn:aws", "s3://", "dagshub", "tracking_uri"):
        assert forbidden not in text
