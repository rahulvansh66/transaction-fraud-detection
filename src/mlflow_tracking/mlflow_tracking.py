# =======================================================================
# Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
# =======================================================================

"""MLflow tracking setup against the project's DagsHub-hosted server.

Pipeline steps and training scripts import :func:`configure_mlflow_tracking`
as a single entry point to point MLflow at DagsHub. It keeps the split the
repo's config conventions require: the tracking URI and experiment name are
non-secret environment values loaded from ``config/env/{env}.yaml``, while
the DagsHub username/token used to authenticate are secrets loaded from the
process environment (populated from a gitignored ``.env`` file).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from src.config_loader.config_loader import load_env_config

logger = logging.getLogger(__name__)

_REQUIRED_ENV_VARS = ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")


def configure_mlflow_tracking(
    env: str = "dev", config_dir: str | Path = "config/env"
) -> str:
    """Points MLflow at the DagsHub tracking server for the given environment.

    Loads DagsHub credentials from the process environment (populated from
    ``.env`` via python-dotenv, never hardcoded) and the non-secret tracking
    URI and experiment name from ``config/env/{env}.yaml``, then sets the
    active MLflow experiment so subsequent ``mlflow.start_run()`` calls log
    to it.

    Args:
        env: Environment name selecting which ``config/env/*.yaml`` to read.
        config_dir: Directory containing the per-environment YAML files.

    Returns:
        The name of the MLflow experiment now active.

    Raises:
        FileNotFoundError: If no config file exists for the given environment.
        RuntimeError: If ``MLFLOW_TRACKING_USERNAME`` / ``MLFLOW_TRACKING_PASSWORD``
            are not set in the environment after loading ``.env``.
    """
    load_dotenv()

    missing = [var for var in _REQUIRED_ENV_VARS if not os.environ.get(var)]
    if missing:
        raise RuntimeError(
            f"Missing MLflow credentials in the environment: {', '.join(missing)}. "
            f"Copy .env.example to .env and fill in your DagsHub token."
        )

    cfg = load_env_config(env, config_dir)
    mlflow_cfg = cfg["mlflow"]

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    logger.info(
        "MLflow tracking configured: env=%s uri=%s experiment=%s",
        env,
        mlflow_cfg["tracking_uri"],
        mlflow_cfg["experiment_name"],
    )
    return mlflow_cfg["experiment_name"]


def set_lineage_tags(
    git_sha: str, config_hash: str, data_run_id: str, mode: str, **extra: str
) -> None:
    """Tags the active MLflow run with the identifiers needed to reproduce it.

    Args:
        git_sha: Commit the run was launched from (with a dirty suffix if applicable).
        config_hash: Hash of the resolved experiment config.
        data_run_id: Immutable processed run id the run read.
        mode: One of single, grid, random, bayesian, production.
        **extra: Additional string tags (e.g. image URI, pipeline execution ARN).
    """
    mlflow.set_tags(
        {"git_sha": git_sha, "config_hash": config_hash, "data_run_id": data_run_id,
         "mode": mode, **extra}
    )


def fetch_dagshub_secret(secret_id: str, region: str = "us-east-1") -> None:
    """Loads DagsHub MLflow credentials from Secrets Manager into the process environment.

    Used on SageMaker, where no .env exists. The secret must be JSON with
    ``MLFLOW_TRACKING_USERNAME`` and ``MLFLOW_TRACKING_PASSWORD`` (the shape written by
    ``infrastructure/dagshub_secret.tf``). The values are never logged.

    Args:
        secret_id: Secrets Manager id, e.g. fraud-detection/dev/dagshub-mlflow.
        region: AWS region of the secret.

    Raises:
        KeyError: If the secret lacks either credential key.
    """
    import json

    import boto3

    payload = boto3.client("secretsmanager", region_name=region).get_secret_value(
        SecretId=secret_id
    )["SecretString"]
    secret = json.loads(payload)
    for var in _REQUIRED_ENV_VARS:
        os.environ[var] = secret[var]
    logger.info("step=secrets status=loaded secret_id=%s", secret_id)
