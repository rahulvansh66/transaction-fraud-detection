# =======================================================================
# Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
# =======================================================================

"""Unit tests for SageMaker request builders, AMT ranges and launch-time guards (no AWS calls)."""

import gzip
import io
import re
import tarfile
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

from src.training import hpo_tuner, sagemaker_jobs
from src.training.experiment_config import ExperimentPlan

RANGES = {
    "max_depth": {"type": "integer", "min": 3, "max": 10},
    "gamma": {"type": "continuous", "min": 0, "max": 5},
}
TUNING = {"objective_type": "Maximize", "objective_metric": "validation:aucpr", "max_parallel_jobs": 2}
TRAINING = {"instance_type": "ml.m5.xlarge", "max_run": 3600}


def _request() -> dict:
    """Builds a training request from fixed inputs."""
    return sagemaker_jobs.build_training_job_request(
        "job", "img", "arn:role", TRAINING, {"eta": "0.1", "max_depth": "6"},
        sagemaker_jobs.data_channels("bkt", "run-1"), "s3://bkt/models/job/", {"A": "b"},
    )


def test_build_ranges_splits_types() -> None:
    """Integer, continuous and categorical parameters land in separate lists; values are strings."""
    ranges = hpo_tuner.build_ranges({"depth": [2, 5, 7]}, RANGES)
    assert ranges["IntegerParameterRanges"][0]["MinValue"] == "3"
    assert ranges["ContinuousParameterRanges"][0]["Name"] == "gamma"
    assert ranges["CategoricalParameterRanges"] == [{"Name": "depth", "Values": ["2", "5", "7"]}]


def test_build_ranges_omits_empty_kinds_and_rejects_unknown_type() -> None:
    """Empty range kinds are dropped from the request and an unknown type fails loudly."""
    assert list(hpo_tuner.build_ranges({"depth": [2, 5]}, {})) == ["CategoricalParameterRanges"]
    with pytest.raises(ValueError):
        hpo_tuner.build_ranges({}, {"x": {"type": "weird", "min": 0, "max": 1}})


def test_tuning_request_static_excludes_tunable_and_sets_parent() -> None:
    """Tunable keys are removed from static params and the MLflow parent id reaches trials."""
    plan = ExperimentPlan(strategy="bayesian", fixed={"eta": 0.1}, ranges=RANGES, trials=3)
    request = _request()
    request["HyperParameters"]["max_depth"] = "6"
    req = hpo_tuner.build_tuning_request("hpo-x", TUNING, plan, request, "parent-1")
    definition = req["TrainingJobDefinition"]
    config = req["HyperParameterTuningJobConfig"]
    assert "max_depth" not in definition["StaticHyperParameters"]
    assert definition["Environment"][hpo_tuner.PARENT_RUN_ENV] == "parent-1"
    assert config["Strategy"] == "Bayesian" and config["ResourceLimits"]["MaxNumberOfTrainingJobs"] == 3


def test_grid_request_omits_max_training_jobs() -> None:
    """AMT derives the Grid job count, so the request must not send MaxNumberOfTrainingJobs."""
    plan = ExperimentPlan(strategy="grid", fixed={}, lists={"max_depth": [2, 5, 7]}, combinations=3, trials=3)
    config = hpo_tuner.build_tuning_request("hpo-x", TUNING, plan, _request(), "p")["HyperParameterTuningJobConfig"]
    assert config["Strategy"] == "Grid"
    assert config["ResourceLimits"] == {"MaxParallelTrainingJobs": 2}
    assert config["ParameterRanges"]["CategoricalParameterRanges"][0]["Values"] == ["2", "5", "7"]


def test_random_request_sends_max_training_jobs() -> None:
    """Random is capped by the configured budget."""
    plan = ExperimentPlan(strategy="random", fixed={}, lists={"max_depth": [2, 5, 7]}, combinations=3, trials=2)
    config = hpo_tuner.build_tuning_request("hpo-x", TUNING, plan, _request(), "p")["HyperParameterTuningJobConfig"]
    assert config["Strategy"] == "Random" and config["ResourceLimits"]["MaxNumberOfTrainingJobs"] == 2


@pytest.mark.parametrize("line,value", [("validation-aucpr:0.8123", "0.8123"), ("validation-aucpr:1e-05", "1e-05")])
def test_metric_regex_handles_scientific_notation(line: str, value: str) -> None:
    """The AMT objective regex captures plain and scientific-notation values."""
    assert re.search(hpo_tuner.METRIC_REGEX, line).group(1) == value


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


class _FakeEcr:
    """ECR stub returning a fixed digest or raising a configured error."""

    def __init__(self, error: bool = False) -> None:
        """Configures whether ``batch_get_image`` fails.

        Args:
            error: If true, raise ``ClientError`` (e.g. no cross-account permission).
        """
        self.error, self.calls = error, []

    def batch_get_image(self, **kwargs: object) -> dict:
        """Returns a digest response, or raises when configured to fail."""
        self.calls.append(kwargs)
        if self.error:
            raise ClientError({"Error": {"Code": "AccessDeniedException", "Message": "x"}}, "BatchGetImage")
        return {"images": [{"imageId": {"imageDigest": "sha256:abc"}}]}


XGB_IMAGE = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1"


def test_image_digest_resolved_from_tagged_uri() -> None:
    """A tagged ECR URI is split into registry, repo and tag, and the digest is returned."""
    ecr = _FakeEcr()
    assert sagemaker_jobs.resolve_image_digest(ecr, XGB_IMAGE) == "sha256:abc"
    assert ecr.calls[0]["registryId"] == "683313688378"
    assert ecr.calls[0]["repositoryName"] == "sagemaker-xgboost"
    assert ecr.calls[0]["imageIds"] == [{"imageTag": "1.7-1"}]


def test_image_digest_failure_never_blocks_launch() -> None:
    """Permission errors and non-ECR URIs return the sentinel instead of raising."""
    assert sagemaker_jobs.resolve_image_digest(_FakeEcr(error=True), XGB_IMAGE) == sagemaker_jobs.UNRESOLVED_DIGEST
    assert sagemaker_jobs.resolve_image_digest(_FakeEcr(), "img") == sagemaker_jobs.UNRESOLVED_DIGEST


def test_requirements_are_exactly_pinned() -> None:
    """The container requirements must be == pins, never ranges, so the bundle is reproducible."""
    text = Path(sagemaker_jobs.REPO_ROOT, sagemaker_jobs.REQUIREMENTS_FILE).read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    assert lines and all("==" in line for line in lines), lines


def test_train_never_reads_test_split() -> None:
    """train.py must not reference the test split (test scoring belongs to evaluate.py)."""
    source = Path(sagemaker_jobs.REPO_ROOT, "src/training/train.py").read_text(encoding="utf-8")
    assert not re.search(r"['\"/]test['\"/]|test_dir|SM_CHANNEL_TEST", source)


class _FakeS3:
    """S3 stand-in holding a set of object keys."""

    def __init__(self, keys: set[str]) -> None:
        """Stores the keys that exist.

        Args:
            keys: Object keys present in the fake bucket.
        """
        self.keys = keys

    def head_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        """Returns for an existing key, raises ``ClientError`` otherwise.

        Args:
            Bucket: Bucket name (ignored).
            Key: Object key.

        Returns:
            Empty dict when the key exists.
        """
        if Key not in self.keys:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {}

    def list_objects_v2(self, Bucket: str, Prefix: str, MaxKeys: int) -> dict[str, int]:
        """Counts keys under a prefix.

        Args:
            Bucket: Bucket name (ignored).
            Prefix: Key prefix.
            MaxKeys: Ignored.

        Returns:
            ``{"KeyCount": n}``.
        """
        return {"KeyCount": sum(k.startswith(Prefix) for k in self.keys)}


def test_processed_run_complete_passes_with_all_parts() -> None:
    """A run with metadata plus both splits is accepted."""
    keys = {"processed/r1/metadata.json", "processed/r1/train/p.parquet", "processed/r1/val/p.parquet"}
    sagemaker_jobs.assert_processed_run_complete(_FakeS3(keys), "b", "r1")


def test_processed_run_incomplete_names_what_is_missing() -> None:
    """A partial run (config only, no splits) fails and names the absent parts."""
    with pytest.raises(FileNotFoundError, match=r"metadata\.json, train/, val/"):
        sagemaker_jobs.assert_processed_run_complete(_FakeS3({"processed/r1/config/x.yaml"}), "b", "r1")
