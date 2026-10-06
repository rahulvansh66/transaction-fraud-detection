# =======================================================================
# Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
# =======================================================================

"""Builds SageMaker training-job requests and the script-mode source bundle.

The installed SageMaker SDK (v3) no longer ships the v2 ``XGBoost`` estimator, so jobs are
submitted with boto3 directly. This module holds the pure request builders (unit-tested
without AWS) and the deterministic source packaging used by both single training jobs and
AMT trials. Used by ``run_training_job.py`` and ``hpo_tuner.py``.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import logging
import re
import tarfile
from pathlib import Path
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS: tuple[str, ...] = ("src/training", "src/evaluation", "src/mlflow_tracking", "src/config_loader")
SOURCE_FILES: tuple[str, ...] = ("src/__init__.py",)
CONFIG_ENV_DIR = "config/env"
REQUIREMENTS_FILE = "src/training/requirements.txt"
ENTRY_POINT = "entrypoint.py"
ENTRY_POINT_SOURCE = "from src.training.train import main\n\nif __name__ == \"__main__\":\n    main()\n"
FIXED_MTIME = 0
PRECONDITION_ERROR_CODES = frozenset({"PreconditionFailed", "ConditionalRequestConflict"})
FORBIDDEN_RUN_IDS = frozenset({"", "latest", "current"})
UNRESOLVED_DIGEST = "unresolved"
ECR_IMAGE_URI = re.compile(
    r"^(?P<account>\d+)\.dkr\.ecr\.[\w-]+\.amazonaws\.com/(?P<repo>[^:@]+):(?P<tag>[^@]+)$"
)


def resolve_image_digest(ecr_client: Any, image_uri: str) -> str:
    """Resolves a tagged ECR image URI to its immutable content digest.

    Tags such as ``1.7-1`` can be re-pushed, so the digest is what actually pins the
    environment. Best-effort: lineage must not block a launch, so any failure (a URI that
    is not a tagged ECR image, or missing ``ecr:BatchGetImage`` on the AWS-owned repo)
    is logged and returns a sentinel instead of raising.

    Args:
        ecr_client: A boto3 ECR client for the image's region.
        image_uri: Tagged ECR image URI from ``image_uris.retrieve``.

    Returns:
        The ``sha256:...`` digest, or :data:`UNRESOLVED_DIGEST`.
    """
    match = ECR_IMAGE_URI.match(image_uri)
    if not match:
        logger.warning("step=launch status=image_digest_skipped reason=not_a_tagged_ecr_uri uri=%s", image_uri)
        return UNRESOLVED_DIGEST
    try:
        response = ecr_client.batch_get_image(
            registryId=match["account"], repositoryName=match["repo"],
            imageIds=[{"imageTag": match["tag"]}],
        )
        return response["images"][0]["imageId"]["imageDigest"]
    except (ClientError, BotoCoreError, KeyError, IndexError):
        logger.warning("step=launch status=image_digest_unresolved uri=%s", image_uri, exc_info=True)
        return UNRESOLVED_DIGEST


def validate_data_run_id(run_id: str) -> str:
    """Rejects mutable or empty data run ids so a re-run always reads the same bytes.

    Args:
        run_id: Processed run id from the experiment config or CLI.

    Returns:
        The validated run id.

    Raises:
        ValueError: If the id is empty or a moving alias such as ``latest``.
    """
    if run_id.strip().lower() in FORBIDDEN_RUN_IDS:
        raise ValueError(f"data run_id must be an immutable processed/<run_id>, got '{run_id}'")
    return run_id


def _iter_source_files(repo_root: Path) -> list[Path]:
    """Lists the files shipped to the container, in a stable order.

    Args:
        repo_root: Repository root.

    Returns:
        Sorted absolute paths of Python modules, env configs and the requirements file.
    """
    files = [repo_root / f for f in SOURCE_FILES]
    for directory in SOURCE_DIRS:
        files += [p for p in (repo_root / directory).rglob("*.py")]
    files += (repo_root / CONFIG_ENV_DIR).glob("*.yaml")
    return sorted(set(files))


def build_source_tar(repo_root: Path = REPO_ROOT) -> bytes:
    """Builds the deterministic ``sourcedir.tar.gz`` for script mode.

    Fixed mtimes, owners and gzip header make identical sources produce identical bytes,
    so the S3 key (which embeds the checksum) is stable.

    Args:
        repo_root: Repository root to package.

    Returns:
        The gzipped tar content: ``entrypoint.py``, ``requirements.txt``, the training-side
        ``src`` packages and ``config/env``.
    """
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        def add(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.uid, info.gid, info.uname, info.gname = len(data), FIXED_MTIME, 0, 0, "", ""
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))

        add(ENTRY_POINT, ENTRY_POINT_SOURCE.encode())
        add("requirements.txt", (repo_root / REQUIREMENTS_FILE).read_bytes())
        for path in _iter_source_files(repo_root):
            add(path.relative_to(repo_root).as_posix(), path.read_bytes())
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=FIXED_MTIME) as gz:
        gz.write(raw.getvalue())
    return out.getvalue()


def stage_source(s3_client: Any, bucket: str, prefix: str, body: bytes) -> str:
    """Uploads the source bundle under a content-addressed key (idempotent).

    Args:
        s3_client: A boto3 S3 client.
        bucket: Bucket name.
        prefix: Key prefix, e.g. ``models/code/``.
        body: Tarball bytes from :func:`build_source_tar`.

    Returns:
        The ``s3://`` URI of the bundle.

    Raises:
        ClientError: On S3 errors other than the object already existing.
    """
    key = f"{prefix}{hashlib.sha256(body).hexdigest()[:16]}/sourcedir.tar.gz"
    try:
        s3_client.put_object(Bucket=bucket, Key=key, Body=body, IfNoneMatch="*")
    except ClientError as err:
        if err.response["Error"]["Code"] not in PRECONDITION_ERROR_CODES:
            raise
    logger.info("step=launch status=source_staged uri=s3://%s/%s", bucket, key)
    return f"s3://{bucket}/{key}"


def script_mode_hyperparameters(source_uri: str, region: str, hp: dict[str, Any]) -> dict[str, str]:
    """Adds the script-mode keys to the hyperparameters and stringifies every value.

    Args:
        source_uri: S3 URI of the source bundle.
        region: AWS region.
        hp: Hyperparameters and control values for ``train.py``.

    Returns:
        SageMaker ``HyperParameters`` (all values are strings).
    """
    base = {
        "sagemaker_program": ENTRY_POINT,
        "sagemaker_submit_directory": source_uri,
        "sagemaker_region": region,
    }
    return {**base, **{k: str(v) for k, v in hp.items()}}


def assert_processed_run_complete(s3_client: Any, bucket: str, data_run_id: str) -> None:
    """Fails fast if ``processed/<run_id>/`` is missing, partial or never finished.

    Checked before any training or tuning job is submitted, so a typo or an interrupted
    preprocessing run is an immediate readable error instead of a failed SageMaker job
    (or, worse, an AMT search whose trials all fail).

    Args:
        s3_client: A boto3 S3 client.
        bucket: Data bucket.
        data_run_id: Validated processed run id.

    Raises:
        FileNotFoundError: If ``metadata.json`` or the ``train/`` or ``val/`` split is absent.
    """
    base = f"processed/{validate_data_run_id(data_run_id)}"
    missing: list[str] = []
    try:
        s3_client.head_object(Bucket=bucket, Key=f"{base}/metadata.json")
    except ClientError:
        missing.append("metadata.json")
    for split in ("train", "val"):
        listing = s3_client.list_objects_v2(Bucket=bucket, Prefix=f"{base}/{split}/", MaxKeys=1)
        if not listing.get("KeyCount"):
            missing.append(f"{split}/")
    if missing:
        raise FileNotFoundError(
            f"s3://{bucket}/{base}/ is incomplete, missing: {', '.join(missing)}; "
            "check data.run_id or re-run preprocessing")
    logger.info("step=launch status=data_run_verified data_run_id=%s", data_run_id)


def data_channels(bucket: str, data_run_id: str) -> list[dict[str, Any]]:
    """Builds the train, val and metadata input channels for an immutable run id.

    Args:
        bucket: Data bucket.
        data_run_id: Validated processed run id.

    Returns:
        ``InputDataConfig`` list. The ``meta`` channel carries ``metadata.json`` (features,
        label, ``scale_pos_weight``), which lives beside the split folders.
    """
    base = f"s3://{bucket}/processed/{validate_data_run_id(data_run_id)}"
    return [
        {"ChannelName": name, "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": uri, "S3DataDistributionType": "FullyReplicated"}}}
        for name, uri in (("train", f"{base}/train/"), ("val", f"{base}/val/"),
                          ("meta", f"{base}/metadata.json"))
    ]


def build_training_job_request(
    job_name: str,
    image_uri: str,
    role_arn: str,
    training: dict[str, Any],
    hyperparameters: dict[str, str],
    channels: list[dict[str, Any]],
    output_uri: str,
    environment: dict[str, str],
) -> dict[str, Any]:
    """Builds the ``CreateTrainingJob`` request.

    Args:
        job_name: Unique training job name (<= 63 chars).
        image_uri: XGBoost container image URI.
        role_arn: SageMaker execution role.
        training: ``training`` block of ``config/env/<env>.yaml``.
        hyperparameters: String hyperparameters including script-mode keys.
        channels: Input channels from :func:`data_channels`.
        output_uri: S3 URI for ``model.tar.gz``.
        environment: Container environment variables (no secrets, only ids).

    Returns:
        Keyword arguments for ``sagemaker_client.create_training_job``.

    Raises:
        KeyError: If ``instance_type`` or ``max_run`` is missing, since an unbounded job
            can keep running and billing.
    """
    return {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {"TrainingImage": image_uri, "TrainingInputMode": "File"},
        "RoleArn": role_arn,
        "HyperParameters": hyperparameters,
        "InputDataConfig": channels,
        "OutputDataConfig": {"S3OutputPath": output_uri},
        "ResourceConfig": {"InstanceType": training["instance_type"], "InstanceCount": 1,
                           "VolumeSizeInGB": int(training.get("volume_size_gb", 30))},
        "StoppingCondition": {"MaxRuntimeInSeconds": int(training["max_run"])},
        "Environment": environment,
    }
