"""Shared config-file loading utilities.

Every pipeline step (preprocessing, training, evaluation, inference,
monitoring) needs to read YAML config from ``config/`` — this module is the
single place that logic lives, so steps don't each reimplement file
resolution and parsing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_env_config(env: str, config_dir: str | Path = "config/env") -> dict[str, Any]:
    """Loads the per-environment config file for the given environment name.

    Args:
        env: Environment name, e.g. ``"dev"`` or ``"prod"``. Selects the file
            ``{config_dir}/{env}.yaml``.
        config_dir: Directory containing the per-environment YAML files.

    Returns:
        The parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If no config file exists for the given environment.
    """
    config_path = Path(config_dir) / f"{env}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No environment config found at {config_path}. "
            f"Expected one of config/env/{{dev,prod}}.yaml."
        )

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
