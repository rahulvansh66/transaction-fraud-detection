"""One-off script: uploads a local dataset directory to S3 under a versioned prefix.

Runs after ``csv_to_hive_parquet.py``. Objects land at
``s3://<bucket>/<prefix>/<data_version>/...`` and existing keys are never
overwritten, so a data version stays immutable and can be referenced from
MLflow lineage. The bucket itself must be provisioned via Terraform.
"""

import argparse
import logging
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from src.config_loader.config_loader import load_env_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Namespace with ``env``, optional ``bucket``, ``local_dir``, ``prefix`` and ``data_version``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="dev", help="Selects config/env/<env>.yaml (region).")
    parser.add_argument("--bucket", default=None, help="Target S3 bucket; defaults to aws.data_bucket in config/env/<env>.yaml.")
    parser.add_argument("--local-dir", type=Path, default=Path("dataset/raw/data-v0"))
    parser.add_argument("--prefix", default="raw", help="Top-level S3 prefix.")
    parser.add_argument("--data-version", default="data-v0", help="Immutable version folder.")
    return parser.parse_args()


def key_exists(client: "boto3.client", bucket: str, key: str) -> bool:
    """Checks whether an object already exists in S3.

    Args:
        client: A boto3 S3 client.
        bucket: Bucket name.
        key: Object key.

    Returns:
        True if the object exists, False if it does not.

    Raises:
        ClientError: For any error other than "not found".
    """
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as err:
        if err.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def upload_dir(client: "boto3.client", local_dir: Path, bucket: str, base_key: str) -> tuple[int, int]:
    """Uploads every file under ``local_dir``, preserving relative paths.

    Args:
        client: A boto3 S3 client.
        local_dir: Directory to upload.
        bucket: Destination bucket.
        base_key: S3 key prefix the relative paths are appended to.

    Returns:
        Tuple of (files_uploaded, files_skipped_because_they_exist).
    """
    uploaded = skipped = 0
    for path in sorted(p for p in local_dir.rglob("*") if p.is_file()):
        key = f"{base_key}/{path.relative_to(local_dir).as_posix()}"
        if key_exists(client, bucket, key):
            skipped += 1
            continue
        client.upload_file(str(path), bucket, key)
        uploaded += 1
    return uploaded, skipped


def main() -> None:
    """Entry point: uploads the dataset and logs the resulting S3 URI."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args()

    if not args.local_dir.is_dir():
        raise SystemExit(f"{args.local_dir} not found; run csv_to_hive_parquet.py first.")

    aws_config = load_env_config(args.env)["aws"]
    region = aws_config["region"]
    bucket = args.bucket or aws_config["data_bucket"]
    client = boto3.client("s3", region_name=region)
    base_key = f"{args.prefix}/{args.data_version}"

    uploaded, skipped = upload_dir(client, args.local_dir, bucket, base_key)
    logger.info(
        "step=upload status=complete uploaded=%d skipped=%d uri=s3://%s/%s/",
        uploaded, skipped, bucket, base_key,
    )


if __name__ == "__main__":
    main()
