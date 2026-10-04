# =======================================================================
# Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
# =======================================================================

"""Unit tests for the PySpark feature functions in ``src/preprocessing/features.py``."""

import pytest
from pyspark.sql import DataFrame, SparkSession

from src.preprocessing import features as f

RAW_SCHEMA = (
    "step int, type string, amount double, nameOrig string, oldbalanceOrg double, "
    "newbalanceOrig double, nameDest string, oldbalanceDest double, newbalanceDest double, "
    "isFraud int, isFlaggedFraud int"
)


def raw_df(spark: SparkSession, rows: list[tuple]) -> DataFrame:
    """Builds a raw-schema DataFrame from tuples.

    Args:
        spark: Active SparkSession.
        rows: Tuples ordered like ``RAW_COLUMNS``.

    Returns:
        DataFrame with ``RAW_COLUMNS``.
    """
    return spark.createDataFrame(rows, RAW_SCHEMA)


def txn(step: int, type_: str, amount: float | None, orig: str, dest: str, fraud: int = 0) -> tuple:
    """Builds one raw transaction tuple with zero balances.

    Args:
        step: Hour index.
        type_: Transaction type.
        amount: Transaction amount.
        orig: Originating account.
        dest: Destination account.
        fraud: Fraud label.

    Returns:
        Tuple ordered like ``RAW_COLUMNS``.
    """
    return (step, type_, amount, orig, 0.0, 0.0, dest, 0.0, 0.0, fraud, 0)


def test_select_raw_rejects_missing_column(spark: SparkSession) -> None:
    """A missing raw column fails loudly."""
    df = raw_df(spark, [txn(1, "TRANSFER", 1.0, "C1", "C2")]).drop("amount")
    with pytest.raises(ValueError, match="amount"):
        f.select_raw(df)


def test_clean_drops_duplicates_and_nulls(spark: SparkSession) -> None:
    """Exact duplicates and null-containing rows are removed."""
    rows = [txn(1, "TRANSFER", 1.0, "C1", "C2")] * 2 + [txn(2, "TRANSFER", None, "C3", "C4")]
    assert f.clean(raw_df(spark, rows)).count() == 1


def test_past_counts_same_step_rows_share_a_count(spark: SparkSession) -> None:
    """Rows in the same step are simultaneous, so no order is invented: both get the same count."""
    df = raw_df(spark, [
        txn(11, "PAYMENT", 1.0, "C1", "M1"),
        txn(12, "TRANSFER", 5.0, "C1", "C2"),
        txn(12, "CASH_OUT", 5.0, "C1", "C3"),
    ])
    out = {(r.step, r.type): r.orig_txn_count for r in f.add_past_counts(df).collect()}
    assert out == {(11, "PAYMENT"): 1, (12, "TRANSFER"): 3, (12, "CASH_OUT"): 3}


def test_past_counts_are_past_only_and_include_non_kept_types(spark: SparkSession) -> None:
    """Counts use only earlier-or-equal steps and include types dropped later by the filter."""
    df = raw_df(spark, [
        txn(1, "PAYMENT", 1.0, "C1", "M1"),
        txn(2, "TRANSFER", 2.0, "C1", "C9"),
        txn(3, "CASH_OUT", 3.0, "C1", "C9"),
    ])
    counted = f.add_past_counts(df)
    out = {r.step: (r.orig_txn_count, r.dest_txn_count) for r in counted.collect()}
    assert out == {1: (1, 1), 2: (2, 1), 3: (3, 2)}
    kept = f.filter_types(counted, ["TRANSFER", "CASH_OUT"])
    assert kept.filter("step = 2").first().orig_txn_count == 2   # PAYMENT row still counted


def test_basic_features(spark: SparkSession) -> None:
    """Type, amount and time features match the notebook definitions."""
    df = raw_df(spark, [txn(27, "TRANSFER", 100.0, "C1", "C2"), txn(30, "CASH_OUT", 0.0, "C3", "C4")])
    rows = {r.step: r for r in f.add_basic_features(df, [0, 6]).collect()}
    assert (rows[27].is_transfer, rows[27].hour_of_day, rows[27].is_night, rows[27].day_of_month) == (1, 3, 1, 1)
    assert (rows[30].is_transfer, rows[30].hour_of_day, rows[30].is_night, rows[30].day_of_month) == (0, 6, 1, 1)
    assert rows[30].log_amount == 0.0


def test_leakage_gate_columns(spark: SparkSession) -> None:
    """Only step, model features and the label survive; error features are opt-in."""
    df = raw_df(spark, [txn(1, "TRANSFER", 5.0, "C1", "C2", 1)])
    df = f.add_basic_features(f.add_past_counts(df), [0, 6])
    out, feats = f.apply_leakage_gate(df, False, "isFraud")
    assert out.columns == ["step", *f.BASE_FEATURES, "isFraud"] and feats == f.BASE_FEATURES
    out2, feats2 = f.apply_leakage_gate(df, True, "isFraud")
    assert feats2 == [*f.BASE_FEATURES, *f.ERROR_FEATURES] and out2.columns[-1] == "isFraud"


def test_split_thresholds_and_partition(spark: SparkSession) -> None:
    """Thresholds floor the fractions of max step and the splits are disjoint and complete."""
    df = spark.createDataFrame([(s, s % 2) for s in range(1, 13)], ["step", "isFraud"])
    t1, t2 = f.split_thresholds(df, 0.7, 0.85)
    assert (t1, t2) == (8, 10)
    parts = f.time_split(df, t1, t2)
    assert [parts[k].count() for k in ("train", "val", "test")] == [8, 2, 2]


@pytest.mark.parametrize("train_frac,val_frac", [(0.0, 0.5), (0.8, 0.7), (0.5, 1.0)])
def test_split_thresholds_rejects_bad_fractions(spark: SparkSession, train_frac: float, val_frac: float) -> None:
    """Fractions must satisfy 0 < train < val < 1."""
    df = spark.createDataFrame([(1, 0)], ["step", "isFraud"])
    with pytest.raises(ValueError):
        f.split_thresholds(df, train_frac, val_frac)


def test_scale_pos_weight(spark: SparkSession) -> None:
    """Weight is n_neg / n_pos, and None when there are no positives."""
    df = spark.createDataFrame([(1, 1), (2, 0), (3, 0), (4, 0)], ["step", "isFraud"])
    assert f.compute_scale_pos_weight(df, "isFraud") == 3.0
    assert f.compute_scale_pos_weight(df.filter("isFraud = 0"), "isFraud") is None
