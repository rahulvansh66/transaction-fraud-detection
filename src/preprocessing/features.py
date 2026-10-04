# =======================================================================
# Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
# =======================================================================

"""PySpark feature-engineering functions for the fraud preprocessing step.

Port of the pandas prototype in ``notebooks/01_preprocessing_feature_engineering.ipynb``.
Every function is a pure ``DataFrame -> DataFrame`` (or a small aggregate over one),
so it can be unit-tested with a local SparkSession and reused unchanged by the
SageMaker PySpark Processing job (``spark_job.py``) and, later, a pipeline step.

The feature definitions here are mirrored in ``config/inference/feature_contract.yaml``
for the serving step (notably the account-count semantics, the main train/serve skew
risk); update that file whenever a feature changes.

Functions avoid Spark actions (``count``, ``collect``) except where a value must
reach the driver (split thresholds, class weight), so the job stays lazy until
the final write.
"""

import logging
import math

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

RAW_COLUMNS: list[str] = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]

BASE_FEATURES: list[str] = [
    "is_transfer", "amount", "log_amount", "hour_of_day",
    "is_night", "day_of_month", "orig_txn_count", "dest_txn_count",
]
ERROR_FEATURES: list[str] = ["errorBalanceOrig", "errorBalanceDest"]


def select_raw(df: DataFrame) -> DataFrame:
    """Validates the raw schema and keeps only the raw columns.

    Args:
        df: Raw transactions as read from the (Hive-partitioned) parquet dataset.

    Returns:
        DataFrame with exactly ``RAW_COLUMNS`` (drops the ``day`` partition column).

    Raises:
        ValueError: If any expected raw column is missing.
    """
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df.select(*RAW_COLUMNS)


def clean(df: DataFrame) -> DataFrame:
    """Drops rows with nulls in required columns and exact duplicate rows.

    Args:
        df: Raw transactions with ``RAW_COLUMNS``.

    Returns:
        Cleaned DataFrame. Merchant ``0.0`` balances are kept as-is (no imputation).
    """
    return df.dropna(subset=RAW_COLUMNS).dropDuplicates()


def add_past_counts(df: DataFrame) -> DataFrame:
    """Adds order-free "up to this step" transaction counts per account.

    ``orig_txn_count`` / ``dest_txn_count`` count the transactions an account has
    made / received with ``step`` earlier than or equal to the current row's.
    ``step`` (one simulated hour) is the only time axis, so rows in the same step are
    simultaneous: they share one count and no order between them is invented.
    Must run before the type filter so the counts cover all transaction types.

    Args:
        df: Transactions with ``step``, ``nameOrig`` and ``nameDest``.

    Returns:
        DataFrame with ``orig_txn_count`` and ``dest_txn_count`` (long) added.
    """

    def running_count(account_col: str) -> Column:
        # RANGE frame: peers with the same ``step`` are all included in the current row's frame.
        window = (
            Window.partitionBy(account_col)
            .orderBy("step")
            .rangeBetween(Window.unboundedPreceding, Window.currentRow)
        )
        return F.count(F.lit(1)).over(window)

    return (
        df.withColumn("orig_txn_count", running_count("nameOrig"))
        .withColumn("dest_txn_count", running_count("nameDest"))
    )


def filter_types(df: DataFrame, keep_types: list[str]) -> DataFrame:
    """Keeps only the configured transaction types.

    Args:
        df: Transactions with a ``type`` column.
        keep_types: Transaction types to retain (e.g. ``TRANSFER``, ``CASH_OUT``).

    Returns:
        Filtered DataFrame.
    """
    return df.filter(F.col("type").isin(keep_types))


def add_basic_features(df: DataFrame, night_hours: list[int]) -> DataFrame:
    """Adds type, amount and time-derived features.

    Args:
        df: Filtered transactions with ``type``, ``amount`` and ``step``.
        night_hours: ``[start, end]`` inclusive hour range flagged as night.

    Returns:
        DataFrame with ``is_transfer``, ``log_amount``, ``hour_of_day``,
        ``is_night`` and ``day_of_month`` added.
    """
    hour = F.col("step") % 24
    return (
        df.withColumn("is_transfer", (F.col("type") == "TRANSFER").cast("byte"))
        .withColumn("log_amount", F.log1p("amount"))
        .withColumn("hour_of_day", hour.cast("short"))
        .withColumn("is_night", hour.between(night_hours[0], night_hours[1]).cast("byte"))
        .withColumn("day_of_month", F.floor(F.col("step") / 24).cast("short"))
    )


def apply_leakage_gate(
    df: DataFrame, include_error_features: bool, label: str
) -> tuple[DataFrame, list[str]]:
    """Optionally builds balance-error features, then keeps only model columns.

    Args:
        df: Featured transactions still holding the raw balance columns.
        include_error_features: If True, add ``errorBalanceOrig`` and
            ``errorBalanceDest`` (leaky post-transaction features; experiments only).
        label: Name of the label column.

    Returns:
        Tuple of (DataFrame with ``step``, the model features and the label only,
        ordered list of model feature names).
    """
    features = list(BASE_FEATURES)
    if include_error_features:
        df = df.withColumn(
            "errorBalanceOrig", F.col("oldbalanceOrg") - F.col("amount") - F.col("newbalanceOrig")
        ).withColumn(
            "errorBalanceDest", F.col("newbalanceDest") - F.col("oldbalanceDest") - F.col("amount")
        )
        features += ERROR_FEATURES
        logger.warning("step=leakage_gate include_error_features=true model_will_use_leaky_features")
    logger.info("step=leakage_gate n_features=%d features=%s", len(features), features)
    return df.select("step", *features, label), features


def split_thresholds(df: DataFrame, train_frac: float, val_frac: float) -> tuple[int, int]:
    """Computes the ``step`` cut-offs for the time-based split.

    Args:
        df: Model DataFrame containing ``step``.
        train_frac: Fraction of the max step where the train split ends.
        val_frac: Fraction of the max step where the validation split ends.

    Returns:
        Tuple ``(train_max_step, val_max_step)``.

    Raises:
        ValueError: If the fractions are not ``0 < train_frac < val_frac < 1``.
    """
    if not 0 < train_frac < val_frac < 1:
        raise ValueError("require 0 < train_frac < val_frac < 1")
    max_step = int(df.agg(F.max("step")).first()[0])
    return math.floor(train_frac * max_step), math.floor(val_frac * max_step)


def time_split(df: DataFrame, train_max_step: int, val_max_step: int) -> dict[str, DataFrame]:
    """Splits rows into train/val/test by ``step`` thresholds.

    Args:
        df: Model DataFrame containing ``step``.
        train_max_step: Last ``step`` (inclusive) of the train split.
        val_max_step: Last ``step`` (inclusive) of the validation split.

    Returns:
        Dict with ``train``, ``val`` and ``test`` DataFrames (disjoint, covering ``df``).
    """
    return {
        "train": df.filter(F.col("step") <= train_max_step),
        "val": df.filter((F.col("step") > train_max_step) & (F.col("step") <= val_max_step)),
        "test": df.filter(F.col("step") > val_max_step),
    }


def compute_scale_pos_weight(train: DataFrame, label: str) -> float | None:
    """Computes XGBoost's ``scale_pos_weight`` from the training split only.

    Args:
        train: Training split containing the label column.
        label: Name of the label column.

    Returns:
        ``n_negative / n_positive``, or None when there are no positives.
    """
    n_pos, n_total = train.agg(F.sum(label), F.count(F.lit(1))).first()
    if not n_pos:
        logger.warning("step=imbalance train_has_no_positives=true")
        return None
    weight = (n_total - n_pos) / n_pos
    logger.info("step=imbalance n_pos=%d n_neg=%d scale_pos_weight=%.4f", n_pos, n_total - n_pos, weight)
    return float(weight)
