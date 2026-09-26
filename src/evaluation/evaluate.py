"""Scores the validation-selected winner once on the TEST split, applies the gate, registers.

This is the only place the test split is read. The winner reuses the threshold chosen on
validation during training. A pass registers a candidate version in the MLflow Model
Registry (no alias, so it awaits manual approval); promotion is never automatic. A failed
winner is final: do not pick another trial, run a new experiment instead.
"""

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import boto3
import mlflow
import mlflow.xgboost
import xgboost as xgb
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config_loader.config_loader import load_env_config, load_yaml
from src.evaluation.metrics import compute_metrics
from src.mlflow_tracking.mlflow_tracking import configure_mlflow_tracking
from src.training.data import load_split

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE = REPO_ROOT / "config" / "evaluation" / "gate.yaml"
EVALUATED_TAG = "evaluated"


def apply_gate(
    report: dict[str, float], gate: dict[str, Any], prod_report: dict[str, float] | None
) -> tuple[bool, list[str]]:
    """Decides whether the winner may become a registered candidate.

    Args:
        report: Test metrics of the winner (``aucpr``, ``recall``, ...).
        gate: Gate config (``min_aucpr``, ``min_recall``, ``no_worse_than_production``).
        prod_report: Test metrics of the current production model on the same data, or
            ``None`` if there is no production model yet.

    Returns:
        Tuple of (passed, list of failure reasons).
    """
    reasons = []
    if not report["aucpr"] >= gate["min_aucpr"]:
        reasons.append(f"aucpr {report['aucpr']:.4f} < min_aucpr {gate['min_aucpr']}")
    if not report["recall"] >= gate["min_recall"]:
        reasons.append(f"recall {report['recall']:.4f} < min_recall {gate['min_recall']}")
    if (gate.get("no_worse_than_production") and prod_report is not None
            and not report["aucpr"] >= prod_report["aucpr"]):
        reasons.append(f"aucpr {report['aucpr']:.4f} < production {prod_report['aucpr']:.4f}")
    return not reasons, reasons


def download_split(bucket: str, prefix: str, dest: Path) -> Path:
    """Downloads every object under an S3 prefix into ``dest``.

    Args:
        bucket: Bucket name.
        prefix: Key prefix (e.g. ``processed/<run_id>/test/``).
        dest: Local directory.

    Returns:
        ``dest``.
    """
    dest.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("/"):
                s3.download_file(bucket, obj["Key"], str(dest / Path(obj["Key"]).name))
    return dest


def score(model_uri: str, x_test: Any, y_test: Any, threshold: float) -> dict[str, float]:
    """Loads a logged XGBoost model and computes test metrics.

    Args:
        model_uri: MLflow model URI.
        x_test: Test features.
        y_test: Test labels.
        threshold: Decision threshold chosen on validation.

    Returns:
        Metrics from :func:`compute_metrics`.
    """
    booster = mlflow.xgboost.load_model(model_uri)
    return compute_metrics(y_test, booster.predict(xgb.DMatrix(x_test)), threshold)


def production_report(
    client: MlflowClient, gate: dict[str, Any], x_test: Any, y_test: Any
) -> dict[str, float] | None:
    """Scores the current production model on the same test data, if one exists.

    Args:
        client: MLflow client.
        gate: Gate config (``model_name``, ``production_alias``).
        x_test: Test features.
        y_test: Test labels.

    Returns:
        Test metrics, or ``None`` when no production alias exists yet.
    """
    try:
        version = client.get_model_version_by_alias(gate["model_name"], gate["production_alias"])
    except MlflowException:
        logger.warning("step=evaluate status=no_production_model model=%s", gate["model_name"])
        return None
    threshold = client.get_run(version.run_id).data.metrics["threshold"]
    return score(f"models:/{gate['model_name']}@{gate['production_alias']}", x_test, y_test, threshold)


def evaluate(run_id: str, gate: dict[str, Any], test_dir: Path, allow_rescore: bool = False) -> bool:
    """Scores the winner on test, logs the report, applies the gate and registers on pass.

    Args:
        run_id: Winner MLflow run id.
        gate: Gate config.
        test_dir: Directory with the test split Parquet files.
        allow_rescore: Permit a second test scoring of the same run (discouraged).

    Returns:
        ``True`` if the gate passed and a version was registered.

    Raises:
        RuntimeError: If this run was already scored on test and ``allow_rescore`` is false.
    """
    client = MlflowClient()
    run = client.get_run(run_id)
    if run.data.tags.get(EVALUATED_TAG) == "true" and not allow_rescore:
        raise RuntimeError(f"run {run_id} was already scored on test; the test set is single-use per winner")

    schema = json.loads(Path(client.download_artifacts(run_id, "features.json")).read_text(encoding="utf-8"))
    x_test, y_test = load_split(test_dir, schema["features"], schema["label"])

    report = score(f"runs:/{run_id}/model", x_test, y_test, run.data.metrics["threshold"])
    prod = production_report(client, gate, x_test, y_test)
    passed, reasons = apply_gate(report, gate, prod)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"test_{k}": v for k, v in report.items()})
        mlflow.set_tags({EVALUATED_TAG: "true", "gate_passed": str(passed).lower()})
        mlflow.log_dict({"metrics": report, "production": prod, "passed": passed, "reasons": reasons},
                        "evaluation.json")
    logger.info("step=evaluate status=complete run_id=%s passed=%s test_aucpr=%.4f reasons=%s",
                run_id, passed, report["aucpr"], reasons)

    if passed:
        version = mlflow.register_model(f"runs:/{run_id}/model", gate["model_name"])
        logger.info("step=register status=registered model=%s version=%s awaiting_approval=true",
                    gate["model_name"], version.version)
    return passed


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Raises:
        SystemExit: With code 1 if the gate fails, so CI marks the job failed.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Winner run id from select_winner.")
    parser.add_argument("--env", default="dev")
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--test-dir", type=Path, default=None, help="Local test split; otherwise read from S3.")
    parser.add_argument("--allow-rescore", action="store_true")
    args = parser.parse_args(argv)

    configure_mlflow_tracking(args.env)
    gate = load_yaml(args.gate)
    test_dir = args.test_dir
    if test_dir is None:
        data_run_id = MlflowClient().get_run(args.run_id).data.tags["data_run_id"]
        bucket = load_env_config(args.env, config_dir=REPO_ROOT / "config" / "env")["aws"]["data_bucket"]
        test_dir = download_split(bucket, f"processed/{data_run_id}/test/", Path(tempfile.mkdtemp()))
    if not evaluate(args.run_id, gate, test_dir, args.allow_rescore):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
