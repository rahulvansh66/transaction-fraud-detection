"""MLflow tracking setup against the project's DagsHub-hosted server.

Pipeline steps and training scripts import :func:`configure_mlflow_tracking`
as a single entry point to point MLflow at DagsHub. It keeps the split the
repo's config conventions require: the tracking URI and experiment name are
non-secret environment values loaded from ``config/env/{env}.yaml``, while
the DagsHub username/token used to authenticate are secrets loaded from the
process environment (populated from a gitignored ``.env`` file).
"""

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
