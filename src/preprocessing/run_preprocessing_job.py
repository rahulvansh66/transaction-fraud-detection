"""Launcher: starts the PySpark preprocessing job as a SageMaker Processing job.

Runs on the developer/CI machine (or later as a pipeline step). It derives the
immutable run identifier (``<git_sha>-<config_hash>-<data_version>``), stages the
config and a versioned deps zip (``features.py`` + ``lineage.py``) under the run's S3
prefix, and submits ``spark_job.py`` to a SageMaker Spark container. The job itself
reads/writes S3 directly and writes ``processed/<run_id>/{train,val,test}``.

Run from the repository root: ``python -m src.preprocessing.run_preprocessing_job``.
"""

import argparse
import hashlib
import io
import logging
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import boto3
import yaml
from botocore.exceptions import ClientError
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.shapes import ProcessingInput, ProcessingS3Input
from sagemaker.core.spark.processing import PySparkProcessor

from src.config_loader.config_loader import load_env_config
from src.preprocessing.lineage import build_run_id, config_hash

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PREPROCESSING_DIR = Path(__file__).resolve().parent
SPARK_JOB_FILE = PREPROCESSING_DIR / "spark_job.py"
DEPS_MODULES: tuple[str, ...] = ("features.py", "lineage.py")
DEPS_ZIP_NAME = "preprocessing_deps.zip"
DEFAULT_CONFIG = REPO_ROOT / "config" / "preprocessing" / "preprocessing.yaml"
CONFIG_LOCAL_PATH = "/opt/ml/processing/input/config"
DEPS_LOCAL_PATH = "/opt/ml/processing/input/deps"
PRECONDITION_ERROR_CODES = frozenset({"PreconditionFailed", "ConditionalRequestConflict"})
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Namespace with ``env``, ``config``, ``data_version``, ``allow_dirty`` and ``no_wait``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="dev", help="Selects config/env/<env>.yaml.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-version", default="data-v0", help="Raw dataset version under raw/.")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Permit a dirty git tree (smoke runs only); the run_id gets a -dirty-<hash> suffix.")
    parser.add_argument("--no-wait", action="store_true", help="Submit the job and return immediately.")
    return parser.parse_args()


def run_git(*args: str) -> str:
    """Runs a git command in the repository root.

    Args:
        *args: Arguments passed to ``git``.

    Returns:
        Stripped stdout.

    Raises:
        subprocess.CalledProcessError: If git is unavailable or this is not a repository.
    """
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def git_state() -> tuple[str, str | None]:
    """Reads the current git commit and, if the tree is dirty, a hash of the uncommitted changes.

    Returns:
        Tuple of (short commit sha, dirty_hash or ``None`` for a clean tree). The dirty hash
        covers the status listing and the diff against HEAD so two dirty runs on different
        edits get different run ids.

    Raises:
        subprocess.CalledProcessError: If git is unavailable or this is not a repository.
    """
    sha = run_git("rev-parse", "--short", "HEAD")
    status = run_git("status", "--porcelain")
    if not status:
        return sha, None
    diff = run_git("diff", "HEAD")
    return sha, hashlib.sha256(f"{status}\n{diff}".encode()).hexdigest()[:8]


def build_deps_zip() -> bytes:
    """Builds the deterministic deps zip holding the modules ``spark_job.py`` imports.

    Timestamps and entry order are fixed, so identical sources always produce identical bytes
    (and therefore a stable checksum for lineage).

    Returns:
        The zip archive content.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(DEPS_MODULES):
            info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (PREPROCESSING_DIR / name).read_bytes())
    return buffer.getvalue()


def put_immutable(s3_client: Any, bucket: str, key: str, body: bytes) -> None:
    """Writes an S3 object only if the key does not exist yet (atomic, no check-then-write race).

    Args:
        s3_client: A boto3 S3 client.
        bucket: Bucket name.
        key: Object key.
        body: Object content.

    Raises:
        SystemExit: If the key already exists (the run was already launched).
        ClientError: On other S3 errors.
    """
    try:
        s3_client.put_object(Bucket=bucket, Key=key, Body=body, IfNoneMatch="*")
    except ClientError as err:
        if err.response["Error"]["Code"] in PRECONDITION_ERROR_CODES:
            raise SystemExit(f"s3://{bucket}/{key} already exists; runs are immutable.") from err
        raise


def stage_artifacts(
    s3_client: Any, bucket: str, out_prefix: str, config_path: Path
) -> tuple[str, str]:
    """Uploads the config and the deps zip under the run prefix.

    The config upload doubles as the atomic launch marker: a second launch of the same
    run id fails here before anything else is written.

    Args:
        s3_client: A boto3 S3 client.
        bucket: Bucket name.
        out_prefix: Run prefix, e.g. ``processed/<run_id>/``.
        config_path: Local preprocessing config file.

    Returns:
        Tuple of (config S3 key, deps zip S3 key).

    Raises:
        SystemExit: If the run prefix was already launched.
    """
    config_key = f"{out_prefix}config/{config_path.name}"
    put_immutable(s3_client, bucket, config_key, config_path.read_bytes())

    deps_zip = build_deps_zip()
    deps_key = f"{out_prefix}deps/{DEPS_ZIP_NAME}"
    put_immutable(s3_client, bucket, deps_key, deps_zip)
    logger.info("step=launch status=staged config=s3://%s/%s deps=s3://%s/%s deps_sha256=%s",
                bucket, config_key, bucket, deps_key, hashlib.sha256(deps_zip).hexdigest()[:12])
    return config_key, deps_key


def s3_input(name: str, uri: str, local_path: str) -> ProcessingInput:
    """Builds a file-mode S3 ``ProcessingInput``.

    Args:
        name: Input channel name.
        uri: Source S3 URI.
        local_path: Mount path inside the container.

    Returns:
        The processing input definition.
    """
    return ProcessingInput(
        input_name=name,
        s3_input=ProcessingS3Input(
            s3_uri=uri, local_path=local_path, s3_data_type="S3Prefix", s3_input_mode="File",
        ),
    )


def build_processor(
    boto_session: boto3.Session, env: dict[str, Any], proc: dict[str, Any], bucket: str
) -> PySparkProcessor:
    """Builds the SageMaker PySpark processor from environment and job-sizing config.

    Args:
        boto_session: Shared boto3 session (region and credentials).
        env: Parsed ``config/env/<env>.yaml``.
        proc: The ``processing`` section of the preprocessing config.
        bucket: Project data bucket (used for the SDK's script staging).

    Returns:
        A configured, not yet started, ``PySparkProcessor``.
    """
    return PySparkProcessor(
        base_job_name=proc["base_job_name"],
        framework_version=proc["spark_framework_version"],
        role=env["aws"]["iam_roles"]["sagemaker_execution_role_arn"],
        instance_type=proc["instance_type"],
        instance_count=proc["instance_count"],
        volume_size_in_gb=proc["volume_size_gb"],
        max_runtime_in_seconds=proc["max_runtime_s"],
        # Stage the SDK's script upload in the project bucket under a prefix the role can read
        sagemaker_session=Session(
            boto_session=boto_session,
            default_bucket=bucket,
            default_bucket_prefix=proc["sdk_staging_prefix"],
        ),
    )


def job_name(processor: PySparkProcessor) -> str:
    """Best-effort lookup of the submitted Processing job name for log correlation.

    Args:
        processor: The processor after ``run()`` was called.

    Returns:
        The job name, or ``"unknown"`` if the SDK does not expose it.
    """
    latest = getattr(processor, "latest_job", None)
    for attr in ("processing_job_name", "job_name", "name"):
        value = getattr(latest, attr, None)
        if value:
            return str(value)
    return "unknown"


def main() -> None:
    """Validates preconditions, stages artifacts and submits the Processing job.

    Raises:
        SystemExit: If the git tree is dirty (without ``--allow-dirty``) or the run already exists.
        ClientError: On AWS API failures.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args()

    env = load_env_config(args.env, config_dir=REPO_ROOT / "config" / "env")
    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    proc = cfg["processing"]
    bucket = env["aws"]["data_bucket"]

    # Reproducibility guards: the run id must identify exactly the code + config + data that ran
    sha, dirty_hash = git_state()
    if dirty_hash and not args.allow_dirty:
        raise SystemExit("Refusing to launch with a dirty git tree; commit first or pass --allow-dirty.")
    run_id = build_run_id(sha, config_hash(cfg), args.data_version, dirty_hash)
    if dirty_hash:
        logger.warning("step=launch status=dirty_tree run_id=%s", run_id)

    boto_session = boto3.Session(region_name=env["aws"]["region"])
    out_prefix = f"processed/{run_id}/"
    config_key, deps_key = stage_artifacts(boto_session.client("s3"), bucket, out_prefix, args.config)

    processor = build_processor(boto_session, env, proc, bucket)
    processor.run(
        submit_app=str(SPARK_JOB_FILE),
        inputs=[
            s3_input("deps", f"s3://{bucket}/{deps_key}", DEPS_LOCAL_PATH),
            s3_input("config", f"s3://{bucket}/{config_key}", CONFIG_LOCAL_PATH),
        ],
        arguments=[
            "--input-uri", f"s3://{bucket}/raw/{args.data_version}/",
            "--output-uri", f"s3://{bucket}/processed/{run_id}",
            "--config", f"{CONFIG_LOCAL_PATH}/{args.config.name}",
            "--git-commit", sha,
            "--run-id", run_id,
        ],
        wait=not args.no_wait,
        logs=not args.no_wait,
    )
    logger.info("step=launch status=submitted run_id=%s processing_job=%s output=s3://%s/processed/%s",
                run_id, job_name(processor), bucket, run_id)


if __name__ == "__main__":
    main()
