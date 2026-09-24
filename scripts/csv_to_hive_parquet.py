"""One-off data-prep script: raw CSV -> Hive-partitioned Parquet on local disk.

Runs *before* the SageMaker pipeline. The pipeline's preprocessing step assumes
Parquet already exists in S3, so all CSV handling lives here. Uses an explicit
schema (no inference) so the small dev file and the full 15GB+ dataset produce
identical dtypes.
"""

import argparse
import logging
import shutil
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA: dict[str, str] = {
    "step": "int32",
    "type": "string",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}
HOURS_PER_DAY = 24


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Namespace with ``input_csv``, ``output_dir`` and ``overwrite``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=Path("dataset/raw-csv/data-v0.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/raw/data-v0"))
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output_dir.")
    return parser.parse_args()


def convert(input_csv: Path, output_dir: Path) -> int:
    """Reads the CSV with the fixed schema and writes day-partitioned Parquet.

    The dataset has no timestamp; ``step`` is an hour counter, so the partition
    key ``day`` is derived as ``(step - 1) // 24``.

    Args:
        input_csv: Path to the raw CSV file.
        output_dir: Root directory for the ``day=<n>/`` Hive layout.

    Returns:
        Number of rows written.
    """
    df = pd.read_csv(input_csv, dtype=SCHEMA)
    df["day"] = ((df["step"] - 1) // HOURS_PER_DAY).astype("int16")
    df = df.sort_values(["step", "nameOrig"], kind="stable").reset_index(drop=True)

    df.to_parquet(output_dir, engine="pyarrow", partition_cols=["day"], index=False)
    return len(df)


def validate(input_csv: Path, output_dir: Path, expected_rows: int) -> None:
    """Reads the Parquet back and checks it matches the source.

    Args:
        input_csv: Path to the raw CSV (used for the source row count).
        output_dir: Root of the written Hive-partitioned dataset.
        expected_rows: Row count returned by ``convert``.

    Raises:
        ValueError: If the row count differs from the source or no partition
            directories were written.
    """
    partitions = sorted(p.name for p in output_dir.glob("day=*"))
    back = pd.read_parquet(output_dir, engine="pyarrow")
    source_rows = sum(1 for _ in input_csv.open("rb")) - 1
    if not partitions or len(back) != source_rows or len(back) != expected_rows:
        raise ValueError(
            f"validation failed: csv_rows={source_rows} parquet_rows={len(back)} "
            f"partitions={len(partitions)}"
        )
    logger.info("step=validate status=ok rows=%d partitions=%d", len(back), len(partitions))


def main() -> None:
    """Entry point: converts the CSV and validates the result."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args()

    if args.output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.output_dir} exists; pass --overwrite to replace it.")
        shutil.rmtree(args.output_dir)

    start = time.time()
    rows = convert(args.input_csv, args.output_dir)
    logger.info(
        "step=convert status=complete rows=%d output=%s duration_s=%.2f",
        rows, args.output_dir, time.time() - start,
    )
    validate(args.input_csv, args.output_dir, rows)


if __name__ == "__main__":
    main()
