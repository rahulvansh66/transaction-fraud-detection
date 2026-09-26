"""SageMaker/local training entry script: trains XGBoost on train+val and logs lineage to MLflow.

This is the only module that knows about SageMaker (``SM_CHANNEL_*``, ``SM_MODEL_DIR``).
Hyperparameters arrive as ``--key value`` CLI args (from AMT, or from the launcher for
manual runs). It never reads the test split; test scoring belongs to ``evaluate.py``.
"""

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import xgboost as xgb
from mlflow.models import infer_signature

from src.evaluation.metrics import compute_metrics, select_threshold
from src.mlflow_tracking.mlflow_tracking import (
    configure_mlflow_tracking,
    fetch_dagshub_secret,
    set_lineage_tags,
)
from src.training.data import load_split, read_metadata
from src.training.model import build_params, fit

logger = logging.getLogger(__name__)

PARENT_RUN_ENV = "MLFLOW_PARENT_RUN_ID"
MODEL_FILE = "xgboost-model"


def coerce(value: str) -> int | float | str:
    """Converts a CLI string into int, float or leaves it as a string.

    Args:
        value: Raw argument value.

    Returns:
        The most specific numeric type that parses, else the original string.
    """
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, dict[str, Any]]:
    """Parses control arguments and collects every other ``--key value`` as a hyperparameter.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Tuple of (control namespace, hyperparameter dict).
    """
    parser = argparse.ArgumentParser(description="Fraud XGBoost training")
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--env", default="dev")
    parser.add_argument("--mode", default="manual", choices=["manual", "hpo", "production"])
    parser.add_argument("--git-sha", default="unknown")
    parser.add_argument("--config-hash", default="unknown")
    parser.add_argument("--data-run-id", default="unknown")
    parser.add_argument("--experiment-name", default=None, help="MLflow run name prefix.")
    parser.add_argument("--expected-xgboost-version", default=None,
                        help="Fail fast if the installed XGBoost differs (local vs container drift).")
    args, unknown = parser.parse_known_args(argv)
    hyperparameters: dict[str, Any] = {}
    for i in range(0, len(unknown) - 1, 2):
        hyperparameters[unknown[i].lstrip("-").replace("-", "_")] = coerce(unknown[i + 1].strip('"'))
    return args, hyperparameters


def check_xgboost_version(expected: str | None) -> None:
    """Fails fast when the installed XGBoost differs from the pinned version.

    Args:
        expected: Version required by the experiment config, or ``None`` to skip the check.

    Raises:
        RuntimeError: If the installed version does not match, since metrics would not
            be reproducible across environments.
    """
    if expected and xgb.__version__ != expected:
        raise RuntimeError(f"XGBoost version mismatch: installed={xgb.__version__} expected={expected}")


def pip_freeze() -> str:
    """Returns the installed package list for the environment lineage artifact.

    Returns:
        ``pip freeze``-style text, or an empty string if it cannot be produced.
    """
    try:
        return subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        from importlib.metadata import distributions

        return "\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions()))


def log_run(
    booster: xgb.Booster,
    params: dict[str, Any],
    resolved: dict[str, Any],
    metrics: dict[str, float],
    features: list[str],
    label: str,
    x_val: Any,
    val_scores: Any,
    args: argparse.Namespace,
    train_dir: str,
    job_name: str,
) -> str:
    """Writes params, metrics, lineage tags and artifacts to the active MLflow run.

    Args:
        booster: Trained booster.
        params: Full booster parameters actually used.
        resolved: Resolved config (static + hyperparameters + control values) for the artifact.
        metrics: Metrics to log (already prefixed).
        features: Ordered feature names.
        label: Label column name.
        x_val: Validation features (for the model input example).
        val_scores: Validation predictions (for the model signature).
        args: Control arguments.
        train_dir: Training data location logged as the run input.
        job_name: SageMaker job name or local run name.

    Returns:
        The MLflow run id.
    """
    run = mlflow.active_run()
    set_lineage_tags(
        git_sha=args.git_sha,
        config_hash=args.config_hash,
        data_run_id=args.data_run_id,
        mode=args.mode,
        image_uri=os.environ.get("TRAINING_IMAGE_URI", "local"),
        pipeline_execution_arn=os.environ.get("PIPELINE_EXECUTION_ARN", ""),
        python_version=platform.python_version(),
        xgboost_version=xgb.__version__,
        job_name=job_name,
    )
    mlflow.log_params(params)
    mlflow.log_param("seed", resolved["seed"])
    mlflow.log_param("num_round", resolved["num_round"])
    mlflow.log_param("best_iteration", booster.best_iteration)
    mlflow.log_metrics(metrics)
    mlflow.log_input(
        mlflow.data.from_pandas(x_val.head(1000), source=str(train_dir), name="processed-val"),
        context="validation",
    )
    mlflow.log_dict(resolved, "resolved_config.json")
    mlflow.log_dict({"features": features, "label": label}, "features.json")
    mlflow.log_text(pip_freeze(), "pip_freeze.txt")
    mlflow.xgboost.log_model(
        booster,
        "model",
        signature=infer_signature(x_val, val_scores),
        input_example=x_val.head(5),
    )
    return run.info.run_id


def main(argv: list[str] | None = None) -> None:
    """Trains one model, logs it to MLflow and saves it to ``SM_MODEL_DIR``.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Raises:
        FileNotFoundError: If the data or metadata files are missing.
        RuntimeError: If the installed XGBoost differs from the pinned version.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args, hp = parse_args(argv)
    check_xgboost_version(args.expected_xgboost_version)
    start = time.time()
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", args.train_dir)
    val_dir = os.environ.get("SM_CHANNEL_VAL", args.val_dir)
    model_dir = os.environ.get("SM_MODEL_DIR", args.model_dir or "model_out")
    job_name = os.environ.get("TRAINING_JOB_NAME", f"local-{int(start)}")
    logger.info("step=train status=start job=%s git_sha=%s data_run_id=%s mode=%s",
                job_name, args.git_sha, args.data_run_id, args.mode)

    meta = read_metadata(os.environ.get("SM_CHANNEL_META", train_dir))
    features, label = meta["features"], meta["label"]
    x_train, y_train = load_split(train_dir, features, label)
    x_val, y_val = load_split(val_dir, features, label)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)

    num_round = int(hp.get("num_round", 100))
    seed = int(hp.get("seed", 42))
    min_precision = float(hp.get("min_precision", 0.9))
    params = build_params(hp, {}, meta["scale_pos_weight"])
    booster = fit(dtrain, dval, params, num_round, seed, int(hp.get("early_stopping_rounds", 20)))

    val_scores = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
    threshold = select_threshold(y_val, val_scores, min_precision)
    val_metrics = compute_metrics(y_val, val_scores, threshold)
    train_metrics = compute_metrics(
        y_train, booster.predict(dtrain, iteration_range=(0, booster.best_iteration + 1)), threshold
    )
    metrics = {f"validation_{k}": v for k, v in val_metrics.items()}
    metrics.update({f"train_{k}": v for k, v in train_metrics.items()})
    metrics["threshold"] = threshold

    if os.environ.get("DAGSHUB_SECRET_ID"):
        fetch_dagshub_secret(os.environ["DAGSHUB_SECRET_ID"])
    configure_mlflow_tracking(args.env)
    resolved = {**hp, "scale_pos_weight": params["scale_pos_weight"], "features": features}
    parent = os.environ.get(PARENT_RUN_ENV)
    with mlflow.start_run(run_name=job_name, parent_run_id=parent, nested=bool(parent)):
        run_id = log_run(booster, params, resolved, metrics, features, label, x_val,
                         val_scores, args, train_dir, job_name)

    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / MODEL_FILE))
    (out / "threshold.json").write_text(json.dumps({"threshold": threshold}), encoding="utf-8")
    logger.info("step=train status=complete job=%s run_id=%s validation_aucpr=%.4f threshold=%.4f "
                "duration_s=%.2f", job_name, run_id, val_metrics["aucpr"], threshold, time.time() - start)
    # AMT parses this line from the logs
    print(f"validation-aucpr:{val_metrics['aucpr']:.6f}")


if __name__ == "__main__":
    main()
