# =======================================================================
# Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
# =======================================================================

"""Generates synthetic processed splits (same contract as the preprocessing job) for local training runs.

The committed sample data has only a handful of rows, too few to train on. This writes
``train/``, ``val/``, ``test/`` parquet folders and ``metadata.json`` under
``dataset/processed/<run_id>`` (gitignored). Local, throwaway script.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["is_transfer", "amount", "log_amount", "hour_of_day", "is_night",
            "day_of_month", "orig_txn_count", "dest_txn_count"]
LABEL = "isFraud"


def make_split(rng: np.random.Generator, n: int, fraud_rate: float) -> pd.DataFrame:
    """Builds one synthetic split with a learnable fraud signal.

    Args:
        rng: Seeded random generator.
        n: Row count.
        fraud_rate: Approximate fraction of fraud rows.

    Returns:
        DataFrame with FEATURES and LABEL columns.
    """
    y = (rng.random(n) < fraud_rate).astype(int)
    amount = np.where(y == 1, rng.lognormal(9, 1.0, n), rng.lognormal(7.5, 1.3, n))
    hour = np.where(y == 1, rng.integers(0, 8, n), rng.integers(0, 24, n))
    return pd.DataFrame({
        "is_transfer": np.where(y == 1, rng.random(n) < 0.6, rng.random(n) < 0.3).astype(int),
        "amount": amount,
        "log_amount": np.log1p(amount),
        "hour_of_day": hour,
        "is_night": ((hour >= 0) & (hour <= 6)).astype(int),
        "day_of_month": rng.integers(1, 31, n),
        "orig_txn_count": rng.poisson(np.where(y == 1, 1.5, 4), n),
        "dest_txn_count": rng.poisson(np.where(y == 1, 6, 3), n),
        LABEL: y,
    })


def main() -> None:
    """Writes the synthetic splits and metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="synthetic-v0")
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path("dataset/processed") / args.run_id
    sizes = {"train": args.rows, "val": args.rows // 4, "test": args.rows // 4}
    counts: dict[str, int] = {}
    fraud: dict[str, int] = {}
    for name, n in sizes.items():
        df = make_split(rng, n, 0.05)
        (root / name).mkdir(parents=True, exist_ok=True)
        df.to_parquet(root / name / "part-0.parquet", index=False)
        counts[name], fraud[name] = len(df), int(df[LABEL].sum())
    spw = (counts["train"] - fraud["train"]) / max(fraud["train"], 1)
    meta = {"features": FEATURES, "label": LABEL, "scale_pos_weight": spw,
            "row_counts": counts, "fraud_counts": fraud, "synthetic": True}
    (root / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {root} rows={counts} fraud={fraud}")


if __name__ == "__main__":
    main()
