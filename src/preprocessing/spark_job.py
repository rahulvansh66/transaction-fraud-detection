"""Entry point of the PySpark preprocessing job (SageMaker Spark Processing step).

Reads raw Hive-partitioned parquet, applies the feature logic in ``features.py``,
writes ``train``/``val``/``test`` parquet folders plus ``metadata.json`` under one
immutable output prefix, and validates the written output contract.

Everything variable (input/output URIs, config path, run identifiers) arrives as
command-line arguments so the same script runs locally (``local[*]``) and inside
SageMaker. Data is read/written with ``s3://`` URIs directly (never through
``ProcessingInput``) because a multi-node Spark cluster cannot share node-local disks.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

DEPS_ZIP_PATH = "/opt/ml/processing/input/deps/preprocessing_deps.zip"

try:  # imported as ``src.preprocessing.spark_job`` locally, as a top-level script on SageMaker
    from src.preprocessing import features
    from src.preprocessing.lineage import config_hash
except ImportError:  # pragma: no cover - SageMaker mounts the versioned deps zip via the "deps" channel
    sys.path.insert(0, DEPS_ZIP_PATH)
    import features  # type: ignore[no-redef]
    from lineage import config_hash  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

SPLITS: tuple[str, ...] = ("train", "val", "test")
LEAKY_COLUMNS: frozenset[str] = frozenset({
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "nameOrig", "nameDest", "type", "isFlaggedFraud",
})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the job's command-line arguments.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        Namespace with ``input_uri``, ``output_uri``, ``config``, ``git_commit``
        and ``run_id``.
    """
    parser = argparse.ArgumentParser(description="Fraud preprocessing PySpark job")
    parser.add_argument("--input-uri", required=True, help="Raw parquet dataset (local path or s3://)")
    parser.add_argument("--output-uri", required=True, help="Immutable output prefix (local path or s3://)")
    parser.add_argument("--config", required=True, help="Path to preprocessing.yaml")
    parser.add_argument("--git-commit", default="unknown", help="Git commit the job was launched from")
    parser.add_argument("--run-id", default="local", help="Run identifier (e.g. <git_sha>-<config_hash>)")
    return parser.parse_args(argv)


def load_config(path: str | Path) -> dict[str, Any]:
    """Loads the preprocessing YAML config.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration as a nested dict.
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def join_uri(base: str, *parts: str) -> str:
    """Joins path segments onto a local path or ``s3://`` URI.

    Args:
        base: Base local path or URI.
        *parts: Segments to append.

    Returns:
        Joined location using ``/`` separators (Spark accepts these on Windows too).
    """
    return "/".join([base.rstrip("/"), *parts])


def write_metadata(output_uri: str, metadata: dict[str, Any]) -> None:
    """Writes ``metadata.json`` next to the splits (S3 via boto3, else local file).

    Args:
        output_uri: Output prefix (local path or ``s3://bucket/prefix``).
        metadata: JSON-serialisable lineage information.
    """
    body = json.dumps(metadata, indent=2, default=str)
    target = join_uri(output_uri, "metadata.json")
    if output_uri.startswith("s3://"):
        import boto3

        parsed = urlparse(target)
        boto3.client("s3").put_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"), Body=body.encode())
    else:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text(body, encoding="utf-8")
    logger.info("step=metadata status=written path=%s", target)


def validate_outputs(
    spark: SparkSession, output_uri: str, feature_names: list[str], label: str
) -> dict[str, dict[str, int | None]]:
    """Re-reads the written splits and asserts the output contract.

    Args:
        spark: Active SparkSession.
        output_uri: Output prefix the splits were written to.
        feature_names: Expected ordered model feature names.
        label: Name of the label column.

    Returns:
        Per-split stats: ``rows``, ``fraud``, ``min_step`` and ``max_step``.

    Raises:
        ValueError: If columns, nulls, leakage or step ordering violate the contract.
    """
    expected = ["step", *feature_names, label]
    stats: dict[str, dict[str, int | None]] = {}
    for name in SPLITS:
        part = spark.read.parquet(join_uri(output_uri, name))
        if part.columns != expected:
            raise ValueError(f"{name}: unexpected columns {part.columns}")
        if LEAKY_COLUMNS & set(part.columns):
            raise ValueError(f"{name}: leaky column present")
        row = part.agg(
            F.count(F.lit(1)).alias("rows"),
            F.sum(label).alias("fraud"),
            F.min("step").alias("min_step"),
            F.max("step").alias("max_step"),
            F.sum(F.greatest(*[F.col(c).isNull().cast("int") for c in expected])).alias("null_rows"),
        ).first()
        if row["null_rows"]:
            raise ValueError(f"{name}: contains nulls")
        stats[name] = {
            "rows": int(row["rows"]),
            "fraud": int(row["fraud"] or 0),
            "min_step": row["min_step"],
            "max_step": row["max_step"],
        }
        logger.info("step=split part=%s rows=%d fraud=%d", name, stats[name]["rows"], stats[name]["fraud"])
        if not stats[name]["rows"] or not stats[name]["fraud"]:
            logger.warning("step=split part=%s has_no_rows_or_no_positives=true", name)
    for earlier, later in (("train", "val"), ("val", "test")):
        a, b = stats[earlier]["max_step"], stats[later]["min_step"]
        if a is not None and b is not None and a >= b:
            raise ValueError(f"step ranges overlap: {earlier} max={a} >= {later} min={b}")
    logger.info("step=validate status=ok n_features=%d", len(feature_names))
    return stats


def run(spark: SparkSession, args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Runs the full preprocessing pipeline and returns the metadata it wrote.

    Args:
        spark: Active SparkSession.
        args: Parsed command-line arguments.
        config: Parsed ``preprocessing.yaml``.

    Returns:
        The ``metadata.json`` content.

    Raises:
        ValueError: If the raw schema or the written output violates its contract.
    """
    label = config["data"]["label_col"]
    logger.info(
        "step=start run_id=%s git_commit=%s input=%s output=%s config_hash=%s spark_version=%s",
        args.run_id, args.git_commit, args.input_uri, args.output_uri, config_hash(config), spark.version,
    )

    df: DataFrame = spark.read.option("basePath", args.input_uri).parquet(args.input_uri)
    df = features.clean(features.select_raw(df))
    df = features.add_past_counts(df)
    df = features.filter_types(df, config["data"]["keep_types"])
    df = features.add_basic_features(df, config["features"]["night_hours"])
    df, feature_names = features.apply_leakage_gate(
        df, config["features"]["include_balance_error_features"], label
    )

    train_max, val_max = features.split_thresholds(
        df, config["split"]["train_frac_of_max_step"], config["split"]["val_frac_of_max_step"]
    )
    splits = features.time_split(df, train_max, val_max)
    weight = features.compute_scale_pos_weight(splits["train"], label)

    for name, part in splits.items():
        part.write.mode("errorifexists").parquet(join_uri(args.output_uri, name))
        logger.info("step=write part=%s status=complete", name)

    stats = validate_outputs(spark, args.output_uri, feature_names, label)
    metadata = {
        "run_id": args.run_id,
        "git_commit": args.git_commit,
        "spark_version": spark.version,
        "source_path": args.input_uri,
        "features": feature_names,
        "label": label,
        "thresholds": {"train_max_step": train_max, "val_max_step": val_max},
        "scale_pos_weight": weight,
        "row_counts": {k: v["rows"] for k, v in stats.items()},
        "fraud_counts": {k: v["fraud"] for k, v in stats.items()},
        "config": config,
        "config_hash": config_hash(config),
    }
    write_metadata(args.output_uri, metadata)
    return metadata


def main(argv: list[str] | None = None) -> None:
    """Configures logging, builds the SparkSession and runs the job.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]``.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args(argv)
    config = load_config(args.config)
    builder = SparkSession.builder.appName("fraud-preprocessing")
    # Applied here (not via the SDK's ``configuration`` arg, which is broken in SDK v3)
    for key, value in config.get("processing", {}).get("spark_conf", {}).items():
        builder = builder.config(key, value)
    spark = builder.getOrCreate()
    try:
        run(spark, args, config)
    except Exception:
        logger.exception("step=job status=failed run_id=%s", args.run_id)
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
