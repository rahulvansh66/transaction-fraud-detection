# =======================================================================
# Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
# =======================================================================

"""Unit tests for SageMaker request builders, AMT ranges and launch-time guards (no AWS calls)."""

import gzip
import io
import re
import tarfile
from pathlib import Path

import pytest

from src.training import hpo_tuner, sagemaker_jobs

SEARCH_SPACE = {
    "max_depth": {"type": "integer", "min": 3, "max": 10},
    "gamma": {"type": "continuous", "min": 0, "max": 5},
}
TRAINING = {"instance_type": "ml.m5.xlarge", "max_run": 3600}


def _request() -> dict:
    """Builds a training request from fixed inputs."""
    return sagemaker_jobs.build_training_job_request(
        "job", "img", "arn:role", TRAINING, {"eta": "0.1", "max_depth": "6"},
        sagemaker_jobs.data_channels("bkt", "run-1"), "s3://bkt/models/job/", {"A": "b"},
    )


def test_build_ranges_splits_types() -> None:
    """Integer and continuous parameters land in separate range lists with string bounds."""
    ranges = hpo_tuner.build_ranges(SEARCH_SPACE)
    assert ranges["IntegerParameterRanges"][0]["MinValue"] == "3"
    assert ranges["ContinuousParameterRanges"][0]["Name"] == "gamma"


def test_build_ranges_rejects_bad_specs() -> None:
    """Unknown types and inverted bounds fail loudly."""
    with pytest.raises(ValueError):
        hpo_tuner.build_ranges({"x": {"type": "weird", "min": 0, "max": 1}})
    with pytest.raises(ValueError):
        hpo_tuner.build_ranges({"x": {"type": "integer", "min": 5, "max": 5}})


def test_tuning_request_static_excludes_tunable_and_sets_parent() -> None:
    """Tunable keys are removed from static params and the MLflow parent id reaches trials."""
    cfg = {"search_space": SEARCH_SPACE, "tuning": {
        "strategy": "Bayesian", "objective_type": "Maximize", "objective_metric": "validation:aucpr",
        "max_jobs": 3, "max_parallel_jobs": 2}}
    req = hpo_tuner.build_tuning_request("hpo-x", cfg, _request(), "parent-1")
    definition = req["TrainingJobDefinition"]
    assert "max_depth" not in definition["StaticHyperParameters"]
    assert definition["Environment"][hpo_tuner.PARENT_RUN_ENV] == "parent-1"
    assert re.search(hpo_tuner.METRIC_REGEX, "validation-aucpr:0.8123").group(1) == "0.8123"


def test_tuning_job_name_length_limit() -> None:
    """Names stay within AMT's 32-character limit."""
    assert len(hpo_tuner.tuning_job_name("a" * 64, "0926103015")) <= 32


def test_training_request_requires_max_run() -> None:
    """A missing max_run raises instead of allowing an unbounded job."""
    with pytest.raises(KeyError):
        sagemaker_jobs.build_training_job_request(
            "j", "i", "r", {"instance_type": "x"}, {}, [], "s3://o", {})
    assert _request()["StoppingCondition"]["MaxRuntimeInSeconds"] == 3600


@pytest.mark.parametrize("bad", ["", "latest", " Latest "])
def test_latest_data_run_id_rejected(bad: str) -> None:
    """Moving aliases are refused so re-runs read identical data."""
    with pytest.raises(ValueError):
        sagemaker_jobs.validate_data_run_id(bad)


def test_source_tar_is_deterministic_and_complete() -> None:
    """Identical sources give identical bytes, and the bundle carries entrypoint and env config."""
    first, second = sagemaker_jobs.build_source_tar(), sagemaker_jobs.build_source_tar()
    assert first == second
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(first))) as tar:
        names = set(tar.getnames())
    assert {"entrypoint.py", "requirements.txt", "src/training/train.py", "config/env/dev.yaml"} <= names


def test_train_never_reads_test_split() -> None:
    """train.py must not reference the test split (test scoring belongs to evaluate.py)."""
    source = Path(sagemaker_jobs.REPO_ROOT, "src/training/train.py").read_text(encoding="utf-8")
    assert not re.search(r"['\"/]test['\"/]|test_dir|SM_CHANNEL_TEST", source)
