"""Shared config-file loading utilities.

Every pipeline step (preprocessing, training, evaluation, inference,
monitoring) needs to read YAML config from ``config/`` — this module is the
single place that logic lives, so steps don't each reimplement file
resolution and parsing.
"""

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
