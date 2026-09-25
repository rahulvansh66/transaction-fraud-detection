"""End-to-end job test and parity check against the pandas notebook output on ``data-v0``."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from src.preprocessing import spark_job

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "dataset" / "raw" / "data-v0"
NOTEBOOK_OUT = REPO_ROOT / "dataset" / "processed" / "data-v0"
CONFIG = REPO_ROOT / "config" / "preprocessing" / "preprocessing.yaml"

pytestmark = pytest.mark.skipif(not RAW.exists(), reason="data-v0 sample not present")


@pytest.fixture(scope="module")
def job_output(spark: SparkSession, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    """Runs the whole job locally on data-v0.

    Args:
        spark: Shared SparkSession.
        tmp_path_factory: pytest factory for a temp output directory.

    Returns:
        Tuple of (output directory, metadata dict returned by the job).
    """
    out = tmp_path_factory.mktemp("spark-v0") / "run"
    args = spark_job.parse_args([
        "--input-uri", str(RAW), "--output-uri", str(out), "--config", str(CONFIG),
        "--git-commit", "test", "--run-id", "test",
    ])
    metadata = spark_job.run(spark, args, spark_job.load_config(CONFIG))
    return out, metadata


def test_job_writes_contract(job_output: tuple[Path, dict[str, Any]]) -> None:
    """The job writes all splits and a metadata.json consistent with the return value."""
    out, metadata = job_output
    assert json.loads((out / "metadata.json").read_text())["run_id"] == "test"
    assert sum(metadata["row_counts"].values()) == 8
    assert metadata["thresholds"] == {"train_max_step": 8, "val_max_step": 10}


def test_job_refuses_to_overwrite(spark: SparkSession, job_output: tuple[Path, dict[str, Any]]) -> None:
    """An existing output prefix is never overwritten (immutable runs)."""
    out, _ = job_output
    args = spark_job.parse_args([
        "--input-uri", str(RAW), "--output-uri", str(out), "--config", str(CONFIG),
    ])
    with pytest.raises(Exception, match="already exists"):
        spark_job.run(spark, args, spark_job.load_config(CONFIG))


@pytest.mark.skipif(not NOTEBOOK_OUT.exists(), reason="notebook output not generated (run the notebook)")
def test_matches_notebook_output(job_output: tuple[Path, dict[str, Any]]) -> None:
    """Per-split rows, thresholds and class weight equal the pandas notebook's."""
    out, metadata = job_output
    expected = json.loads((NOTEBOOK_OUT / "metadata.json").read_text())
    for key in ("features", "thresholds", "scale_pos_weight", "row_counts", "fraud_counts"):
        assert metadata[key] == expected[key], key
    sort_cols = ["step", "amount", "is_transfer"]
    for split in spark_job.SPLITS:
        got = pd.read_parquet(out / split).sort_values(sort_cols).reset_index(drop=True)
        want = pd.read_parquet(NOTEBOOK_OUT / split).sort_values(sort_cols).reset_index(drop=True)
        pd.testing.assert_frame_equal(got, want, check_dtype=False)
