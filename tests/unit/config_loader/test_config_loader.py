# =======================================================================
# Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
# =======================================================================

"""Unit tests for config merging and loading helpers."""

import re
from pathlib import Path

import pytest

from src.config_loader.config_loader import load_env_config, load_yaml, merge_configs
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


def test_env_config_expands_placeholders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``${VAR}`` placeholders are substituted from the environment, nested and in strings."""
    (tmp_path / "x.yaml").write_text(
        'a: "${MLFLOW_TRACKING_URI}"\nb:\n  - "arn:aws:iam::${AWS_ACCOUNT_ID}:role/r"\n', encoding="utf-8"
    )
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.test/x.mlflow")
    monkeypatch.setenv("AWS_ACCOUNT_ID", "123456789012")
    cfg = load_env_config("x", config_dir=tmp_path)
    assert cfg == {"a": "https://example.test/x.mlflow", "b": ["arn:aws:iam::123456789012:role/r"]}


def test_env_config_unset_placeholder_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset non-account placeholder fails loudly instead of leaving ``${VAR}`` in place."""
    (tmp_path / "x.yaml").write_text('a: "${NOT_SET_ANYWHERE}"\n', encoding="utf-8")
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    with pytest.raises(KeyError):
        load_env_config("x", config_dir=tmp_path)


def test_committed_env_configs_hold_no_account_ids() -> None:
    """Committed env configs use placeholders, never a literal 12-digit account id."""
    for path in Path("config/env").glob("*.yaml"):
        assert not re.search(r"\b\d{12}\b", path.read_text(encoding="utf-8")), path
