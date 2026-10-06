# =======================================================================
# Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
# =======================================================================

"""Shared config-file loading utilities.

Every pipeline step (preprocessing, training, evaluation, inference,
monitoring) needs to read YAML config from ``config/`` — this module is the
single place that logic lives, so steps don't each reimplement file
resolution and parsing.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_PLACEHOLDER = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")
ACCOUNT_ID_VAR = "AWS_ACCOUNT_ID"


def _resolve_account_id() -> str:
    """Returns the AWS account id from the environment, falling back to STS.

    Returns:
        The 12-digit AWS account id.

    Raises:
        RuntimeError: If ``AWS_ACCOUNT_ID`` is unset and STS cannot be reached.
    """
    account_id = os.environ.get(ACCOUNT_ID_VAR)
    if account_id:
        return account_id
    try:
        import boto3

        return boto3.client("sts").get_caller_identity()["Account"]
    except Exception as err:
        raise RuntimeError(
            f"{ACCOUNT_ID_VAR} is not set and could not be resolved via STS. "
            f"Set it in .env (see .env.example) or configure AWS credentials."
        ) from err


def _expand_placeholders(value: Any) -> Any:
    """Recursively replaces ``${VAR}`` placeholders in string values of a parsed config.

    Keeps account-specific and deployment-specific values (account id, MLflow
    tracking URI) out of the committed YAML; they come from the environment
    (``.env`` locally, GitHub Variables in CI, the job environment on AWS).

    Args:
        value: A parsed YAML node (dict, list, string or scalar).

    Returns:
        The same structure with every ``${VAR}`` substituted.

    Raises:
        KeyError: If a referenced variable is not set (``AWS_ACCOUNT_ID`` is
            resolved via STS before giving up).
    """
    if isinstance(value, dict):
        return {k: _expand_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_placeholders(v) for v in value]
    if not isinstance(value, str):
        return value

    def substitute(match: re.Match[str]) -> str:
        """Resolves one ``${VAR}`` match from the environment."""
        name = match.group(1)
        if name == ACCOUNT_ID_VAR:
            return _resolve_account_id()
        if name not in os.environ:
            raise KeyError(
                f"Config references ${{{name}}} but it is not set. "
                f"Add it to .env (see .env.example) or the job environment."
            )
        return os.environ[name]

    return _PLACEHOLDER.sub(substitute, value)


def load_env_config(env: str, config_dir: str | Path = "config/env") -> dict[str, Any]:
    """Loads the per-environment config file for the given environment name.

    ``${VAR}`` placeholders in values (e.g. ``${AWS_ACCOUNT_ID}``,
    ``${MLFLOW_TRACKING_URI}``) are substituted from the environment; a local
    ``.env`` file is loaded first if present.

    Args:
        env: Environment name, e.g. ``"dev"`` or ``"prod"``. Selects the file
            ``{config_dir}/{env}.yaml``.
        config_dir: Directory containing the per-environment YAML files.

    Returns:
        The parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If no config file exists for the given environment.
        KeyError: If a ``${VAR}`` placeholder refers to an unset variable.
        RuntimeError: If ``AWS_ACCOUNT_ID`` is needed but cannot be resolved.
    """
    load_dotenv()
    config_path = Path(config_dir) / f"{env}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No environment config found at {config_path}. "
            f"Expected one of config/env/{{dev,prod}}.yaml."
        )

    with config_path.open("r", encoding="utf-8") as f:
        return _expand_placeholders(yaml.safe_load(f))


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Loads any YAML file (e.g. an experiment config) into a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed YAML content.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No config file found at {path}.")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merges ``override`` into ``base`` and returns a new dict.

    Args:
        base: Base configuration; not mutated.
        override: Values that win over ``base``; nested dicts are merged key by key.

    Returns:
        A new merged dictionary.
    """
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_configs(out[key], value)
        else:
            out[key] = value
    return out
