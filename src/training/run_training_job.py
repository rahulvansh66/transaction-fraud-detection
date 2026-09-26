"""Client-side launcher for training. ``--mode local`` runs ``train.py`` as a subprocess.

Contains no ML logic: it resolves the experiment config, derives lineage identifiers
and builds the argument list. ``--mode sagemaker`` is implemented in a later step.

Run from the repository root: ``python -m src.training.run_training_job``.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.config_loader.config_loader import load_yaml
from src.training.lineage import config_hash, current_git_sha

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = "src.training.train"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses launcher arguments.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Namespace with mode, kind, experiment, env, data_dir and data_run_id.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["local", "sagemaker"], default="local")
    parser.add_argument("--kind", choices=["manual", "hpo", "production"], default="manual")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--env", default="dev")
    parser.add_argument("--data-run-id", default=None, help="Overrides data.run_id from the experiment.")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Local processed root; defaults to dataset/processed/<run_id>.")
    return parser.parse_args(argv)


def build_train_args(cfg: dict[str, Any], kind: str, data_dir: Path, data_run_id: str,
                     env: str, git_sha: str) -> list[str]:
    """Builds the ``train.py`` argument list from the experiment config.

    Args:
        cfg: Parsed experiment config.
        kind: ``manual``, ``hpo`` or ``production``.
        data_dir: Directory holding ``train/``, ``val/`` and ``metadata.json``.
        data_run_id: Immutable data run identifier.
        env: Environment name.
        git_sha: Code version to tag.

    Returns:
        CLI arguments for ``train.py``.
    """
    hp = {**cfg["static_params"], **cfg.get("manual_params", {})}
    args = [
        "--train-dir", str(data_dir / "train"), "--val-dir", str(data_dir / "val"),
        "--env", env, "--mode", kind, "--git-sha", git_sha, "--data-run-id", data_run_id,
        "--config-hash", config_hash(cfg),
    ]
    for key, value in hp.items():
        args += [f"--{key}", str(value)]
    return args


def main(argv: list[str] | None = None) -> None:
    """Resolves the experiment and launches training.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Raises:
        NotImplementedError: For ``--mode sagemaker`` until step 9.
        subprocess.CalledProcessError: If the training subprocess fails.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args(argv)
    if args.mode != "local":
        raise NotImplementedError("SageMaker mode is implemented in step 9.")

    cfg = load_yaml(args.experiment)
    data_run_id = args.data_run_id or cfg["data"]["run_id"]
    data_dir = args.data_dir or REPO_ROOT / "dataset" / "processed" / data_run_id
    git_sha = current_git_sha()
    logger.info("step=launch status=start mode=local kind=%s experiment=%s data_run_id=%s git_sha=%s",
                args.kind, cfg["experiment"]["name"], data_run_id, git_sha)

    train_args = build_train_args(cfg, args.kind, data_dir, data_run_id, args.env, git_sha)
    # MLflow prints an emoji on run end, which crashes cp1252 consoles on Windows
    env = {**os.environ, "PYTHONUTF8": "1"}
    subprocess.run([sys.executable, "-m", TRAIN_SCRIPT, *train_args], cwd=REPO_ROOT, check=True, env=env)


if __name__ == "__main__":
    main()
