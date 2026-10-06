# =======================================================================
# Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
# =======================================================================

"""Client-side launcher for training: local subprocess, one SageMaker job, or an AMT search.

Contains no ML logic: it resolves the experiment config, derives lineage identifiers and
submits the work. ``--mode local`` runs ``train.py`` as a subprocess; ``--mode sagemaker``
submits a training job or, when the experiment's ``params`` contain lists or ranges, a
tuning job whose strategy comes from ``tuning.strategy``. ``--kind production`` reads
``config/production/model.yaml`` and always trains one fixed point (never AMT).

Run from the repository root: ``python -m src.training.run_training_job``.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from src.config_loader.config_loader import load_env_config, load_yaml
from src.training.experiment_config import (
    SINGLE,
    ExperimentPlan,
    validate_experiment,
    validate_production,
)
from src.training.lineage import config_hash, current_git_sha, full_git_sha
from src.training.sagemaker_jobs import validate_data_run_id

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = "src.training.train"
PRODUCTION_CONFIG = REPO_ROOT / "config" / "production" / "model.yaml"
PRODUCTION_MODE = "production"
SOURCE_PREFIX = "models/code/"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses launcher arguments.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Namespace with mode, kind, experiment, env, data_dir, data_run_id, allow_dirty, no_wait and output_json.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["local", "sagemaker"], default="local")
    parser.add_argument("--kind", choices=["experiment", "production"], default="experiment")
    parser.add_argument("--experiment", type=Path, default=None,
                        help="Experiment YAML; ignored for --kind production.")
    parser.add_argument("--env", default="dev")
    parser.add_argument("--data-run-id", default=None, help="Overrides data.run_id from the config.")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Local processed root; defaults to dataset/processed/<run_id>.")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Permit a dirty git tree for SageMaker runs (smoke tests only).")
    parser.add_argument("--no-wait", action="store_true", help="Submit and return immediately.")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Write {run_id} (single/production) or {parent_run_id} (search) for later steps.")
    return parser.parse_args(argv)


def resolve_config(kind: str, experiment: Path | None) -> tuple[dict[str, Any], ExperimentPlan]:
    """Loads and validates the config that defines this run.

    Args:
        kind: ``experiment`` or ``production``.
        experiment: Experiment YAML path (required unless ``kind == "production"``).

    Returns:
        Tuple ``(config, plan)``. For ``production`` the approved ``config/production/model.yaml``
        replaces ``params`` and is layered over the ``data``/``runtime`` defaults of the
        experiment if given; the plan is always a single run of its scalar params.

    Raises:
        SystemExit: If no experiment is given for the ``experiment`` kind.
        ValueError: If the config is invalid (see
            :func:`src.training.experiment_config.validate_experiment`).
    """
    if kind == "production":
        base = load_yaml(experiment) if experiment else {}
        production = load_yaml(PRODUCTION_CONFIG)
        cfg = {**{k: v for k, v in base.items() if k not in ("params", "tuning")}, **production}
        return cfg, ExperimentPlan(strategy=SINGLE, fixed=validate_production(cfg))
    if experiment is None:
        raise SystemExit("--experiment is required for --kind experiment.")
    cfg = load_yaml(experiment)
    return cfg, validate_experiment(cfg)


def run_mode(kind: str, plan: ExperimentPlan) -> str:
    """Returns the lineage ``mode`` tag for a run.

    Args:
        kind: ``experiment`` or ``production``.
        plan: The resolved plan.

    Returns:
        ``production`` for production runs, else the plan strategy
        (``single``, ``grid``, ``random`` or ``bayesian``).
    """
    return PRODUCTION_MODE if kind == "production" else plan.strategy


def build_hyperparameters(plan: ExperimentPlan) -> dict[str, Any]:
    """Selects the hyperparameters passed to ``train.py`` by the launcher.

    Args:
        plan: The resolved plan.

    Returns:
        The fixed (scalar) params. Lists and ranges are supplied per trial by AMT.
    """
    return dict(plan.fixed)


def build_control_args(cfg: dict[str, Any], mode: str, env: str, git_sha: str, data_run_id: str) -> dict[str, str]:
    """Builds the non-hyperparameter arguments ``train.py`` expects.

    Args:
        cfg: Parsed config.
        mode: Lineage mode from :func:`run_mode`.
        env: Environment name.
        git_sha: Code version to tag.
        data_run_id: Immutable data run identifier.

    Returns:
        Mapping of CLI option (without ``--``) to value.
    """
    control = {"env": env, "mode": mode, "git-sha": git_sha, "data-run-id": data_run_id,
               "config-hash": config_hash(cfg)}
    version = cfg.get("runtime", {}).get("xgboost_version")
    if version:
        control["expected-xgboost-version"] = version
    experiment_name = cfg.get("experiment", {}).get("name")
    if experiment_name:
        control["experiment-name"] = experiment_name
    return control


def build_train_args(cfg: dict[str, Any], plan: ExperimentPlan, mode: str, data_dir: Path, data_run_id: str,
                     env: str, git_sha: str) -> list[str]:
    """Builds the ``train.py`` argument list for a local run.

    Args:
        cfg: Parsed config.
        plan: The resolved (single) plan.
        mode: Lineage mode from :func:`run_mode`.
        data_dir: Directory holding ``train/``, ``val/`` and ``metadata.json``.
        data_run_id: Immutable data run identifier.
        env: Environment name.
        git_sha: Code version to tag.

    Returns:
        CLI arguments for ``train.py``.
    """
    args = ["--train-dir", str(data_dir / "train"), "--val-dir", str(data_dir / "val")]
    for key, value in build_control_args(cfg, mode, env, git_sha, data_run_id).items():
        args += [f"--{key}", value]
    for key, value in build_hyperparameters(plan).items():
        args += [f"--{key}", str(value)]
    return args


def run_local(cfg: dict[str, Any], plan: ExperimentPlan, args: argparse.Namespace, data_run_id: str,
              git_sha: str) -> None:
    """Runs ``train.py`` as a subprocess on local Parquet splits.

    Args:
        cfg: Parsed config.
        plan: The resolved plan; only ``single`` runs are supported locally.
        args: Launcher arguments.
        data_run_id: Immutable data run identifier.
        git_sha: Code version to tag.

    Raises:
        SystemExit: If the plan is a search (lists or ranges); use ``--mode sagemaker``.
        FileNotFoundError: If the processed data directory does not exist.
        subprocess.CalledProcessError: If the training subprocess fails.
    """
    if plan.strategy != SINGLE:
        raise SystemExit(f"--mode local supports single runs only; this experiment is a {plan.strategy} search "
                         "(use --mode sagemaker, or make params all scalars).")
    data_dir = args.data_dir or REPO_ROOT / "dataset" / "processed" / data_run_id
    if not data_dir.is_dir():
        raise FileNotFoundError(f"processed data not found: {data_dir}")
    train_args = build_train_args(cfg, plan, run_mode(args.kind, plan), data_dir, data_run_id, args.env, git_sha)
    # MLflow prints an emoji on run end, which crashes cp1252 consoles on Windows
    env = {**os.environ, "PYTHONUTF8": "1"}
    subprocess.run([sys.executable, "-m", TRAIN_SCRIPT, *train_args], cwd=REPO_ROOT, check=True, env=env)


def wait_for_tuning(client: Any, name: str, poll_s: int = 60) -> str:
    """Polls an AMT job until it reaches a terminal state.

    Args:
        client: A boto3 SageMaker client.
        name: Tuning job name.
        poll_s: Seconds between polls.

    Returns:
        The terminal ``HyperParameterTuningJobStatus``.
    """
    while True:
        status = client.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=name)["HyperParameterTuningJobStatus"]
        if status in ("Completed", "Failed", "Stopped"):
            return status
        time.sleep(poll_s)


def write_output(path: Path | None, payload: dict[str, Any]) -> None:
    """Writes run identifiers for downstream workflow steps.

    Args:
        path: Output file, or ``None`` to skip.
        payload: JSON-serialisable identifiers.
    """
    if path:
        path.write_text(json.dumps(payload), encoding="utf-8")


def run_sagemaker(cfg: dict[str, Any], plan: ExperimentPlan, args: argparse.Namespace, data_run_id: str,
                  git_sha: str) -> None:
    """Submits a SageMaker training job or an AMT tuning job.

    Args:
        cfg: Parsed config.
        plan: The resolved plan; ``single`` submits one training job, anything else an AMT job.
        args: Launcher arguments.
        data_run_id: Immutable data run identifier.
        git_sha: Code version to tag (must be clean unless ``--allow-dirty``).

    Raises:
        SystemExit: If the git tree is dirty without ``--allow-dirty``.
        KeyError: If the env config lacks ``training.max_run`` or ``training.instance_type``.
    """
    import boto3
    import mlflow
    from sagemaker.core import image_uris

    from src.mlflow_tracking.mlflow_tracking import configure_mlflow_tracking
    from src.training import hpo_tuner, sagemaker_jobs

    if "dirty" in git_sha and not args.allow_dirty:
        raise SystemExit("Refusing to launch on SageMaker with a dirty git tree; commit or pass --allow-dirty.")

    env_cfg = load_env_config(args.env, config_dir=REPO_ROOT / "config" / "env")
    region, bucket = env_cfg["aws"]["region"], env_cfg["aws"]["data_bucket"]
    session = boto3.Session(region_name=region)
    image_uri = image_uris.retrieve("xgboost", region=region, version=cfg["runtime"]["sagemaker_framework_version"])

    source_uri = sagemaker_jobs.stage_source(session.client("s3"), bucket, SOURCE_PREFIX,
                                             sagemaker_jobs.build_source_tar())
    mode = run_mode(args.kind, plan)
    hp = {**build_hyperparameters(plan), **build_control_args(cfg, mode, args.env, git_sha, data_run_id)}
    stamp = time.strftime("%m%d%H%M%S")
    name = f"fraud-{mode}-{config_hash(cfg)[:8]}-{stamp}"
    environment = {"TRAINING_IMAGE_URI": image_uri, "TRAINING_JOB_NAME": name,
                   "TRAINING_IMAGE_DIGEST": sagemaker_jobs.resolve_image_digest(
                       session.client("ecr"), image_uri),
                   "TRAINING_SOURCE_URI": source_uri, "GIT_SHA_FULL": full_git_sha(),
                   "DAGSHUB_SECRET_ID": f"fraud-detection/{args.env}/dagshub-mlflow",
                   "AWS_ACCOUNT_ID": env_cfg["aws"]["account_id"],
                   "MLFLOW_TRACKING_URI": env_cfg["mlflow"]["tracking_uri"]}
    request = sagemaker_jobs.build_training_job_request(
        job_name=name, image_uri=image_uri,
        role_arn=env_cfg["aws"]["iam_roles"]["sagemaker_execution_role_arn"],
        training=env_cfg["training"],
        hyperparameters=sagemaker_jobs.script_mode_hyperparameters(source_uri, region, hp),
        channels=sagemaker_jobs.data_channels(bucket, data_run_id),
        output_uri=f"s3://{bucket}/models/{name}/", environment=environment,
    )
    client = session.client("sagemaker")

    if plan.strategy != SINGLE:
        configure_mlflow_tracking(args.env)
        tuning_name = hpo_tuner.tuning_job_name(config_hash(cfg), stamp)
        with mlflow.start_run(run_name=tuning_name) as parent:
            mlflow.set_tags({"mode": mode, "experiment_name": cfg["experiment"]["name"],
                             "git_sha": git_sha, "config_hash": config_hash(cfg),
                             "data_run_id": data_run_id, "tuning_job_name": tuning_name,
                             "env": args.env, "git_sha_full": environment["GIT_SHA_FULL"],
                             "source_uri": source_uri,
                             "image_digest": environment["TRAINING_IMAGE_DIGEST"]})
            client.create_hyper_parameter_tuning_job(
                **hpo_tuner.build_tuning_request(tuning_name, cfg["tuning"], plan, request, parent.info.run_id))
            logger.info("step=launch status=submitted mode=%s strategy=%s combinations=%d trials=%d "
                        "tuning_job=%s parent_run_id=%s", mode, plan.strategy, plan.combinations, plan.trials,
                        tuning_name, parent.info.run_id)
        if not args.no_wait:
            status = wait_for_tuning(client, tuning_name)
            logger.info("step=launch status=finished tuning_job=%s job_status=%s", tuning_name, status)
            if status != "Completed":
                raise SystemExit(f"tuning job {tuning_name} ended with status {status}")
        write_output(args.output_json, {"parent_run_id": parent.info.run_id})
        return

    client.create_training_job(**request)
    logger.info("step=launch status=submitted mode=%s training_job=%s image=%s", mode, name, image_uri)
    if not args.no_wait:
        client.get_waiter("training_job_completed_or_stopped").wait(TrainingJobName=name)
        status = client.describe_training_job(TrainingJobName=name)["TrainingJobStatus"]
        logger.info("step=launch status=finished training_job=%s job_status=%s", name, status)
        if status != "Completed":
            raise SystemExit(f"training job {name} ended with status {status}")
        configure_mlflow_tracking(args.env)
        runs = mlflow.search_runs(filter_string=f"tags.job_name = '{name}'", max_results=1)
        write_output(args.output_json, {"run_id": runs.iloc[0].run_id if len(runs) else None})


def main(argv: list[str] | None = None) -> None:
    """Resolves the config and launches training locally or on SageMaker.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Raises:
        ValueError: If the data run id is empty or a moving alias like ``latest``.
        SystemExit: On invalid argument combinations or a dirty tree for SageMaker.
        subprocess.CalledProcessError: If the local training subprocess fails.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args(argv)
    cfg, plan = resolve_config(args.kind, args.experiment)
    data_run_id = validate_data_run_id(args.data_run_id or cfg["data"]["run_id"])
    git_sha = current_git_sha()
    logger.info("step=launch status=start mode=%s kind=%s strategy=%s data_run_id=%s git_sha=%s",
                args.mode, args.kind, plan.strategy, data_run_id, git_sha)
    if args.mode == "local":
        run_local(cfg, plan, args, data_run_id, git_sha)
    else:
        run_sagemaker(cfg, plan, args, data_run_id, git_sha)


if __name__ == "__main__":
    main()
