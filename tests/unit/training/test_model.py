"""Unit tests for parameter building, data loading and determinism of training."""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.evaluation.metrics import compute_metrics
from src.training.data import load_split
from src.training.model import build_params, fit
from src.training.run_training_job import build_train_args
from src.training.train import coerce, parse_args


def synthetic(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Builds a small learnable dataset.

    Args:
        n: Row count.
        seed: Random seed.

    Returns:
        Tuple of (features, labels).
    """
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = (x["a"] + 0.5 * x["b"] + rng.normal(scale=0.5, size=n) > 1).astype(int).to_numpy()
    return x, y


def test_build_params_overrides_and_weight() -> None:
    """Overrides win, control keys are dropped, weight applied, ints cast."""
    p = build_params({"eta": 0.1, "num_round": 50, "max_depth": 3}, {"eta": 0.3, "max_depth": 6.0}, 12.5)
    assert p["eta"] == 0.3 and p["max_depth"] == 6 and isinstance(p["max_depth"], int)
    assert p["scale_pos_weight"] == 12.5 and "num_round" not in p


def test_parse_args_collects_hyperparameters() -> None:
    """Unknown --key value pairs become typed hyperparameters."""
    args, hp = parse_args(["--mode", "hpo", "--max_depth", "5", "--eta", "0.2", "--objective", "binary:logistic"])
    assert args.mode == "hpo"
    assert hp == {"max_depth": 5, "eta": 0.2, "objective": "binary:logistic"}
    assert coerce("3") == 3 and coerce("x") == "x"


def test_same_seed_same_metrics() -> None:
    """Two fits with the same seed produce identical predictions."""
    x, y = synthetic()
    dtrain, dval = xgb.DMatrix(x[:150], label=y[:150]), xgb.DMatrix(x[150:], label=y[150:])
    params = build_params({"objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist"}, {}, 1.0)
    scores = [fit(dtrain, dval, params, 20, seed=1).predict(dval) for _ in range(2)]
    assert np.array_equal(scores[0], scores[1])
    assert compute_metrics(y[150:], scores[0], 0.5)["aucpr"] > 0.5


def test_load_split_selects_columns(tmp_path: Path) -> None:
    """load_split returns the requested feature columns and label."""
    x, y = synthetic(10)
    pd.concat([x, pd.Series(y, name="isFraud")], axis=1).to_parquet(tmp_path / "part-0.parquet")
    xs, ys = load_split(tmp_path, ["a", "b"], "isFraud")
    assert list(xs.columns) == ["a", "b"] and len(ys) == 10


def test_launcher_args_carry_lineage(tmp_path: Path) -> None:
    """The launcher passes data ids, mode and config hash to train.py."""
    cfg = {"static_params": {"seed": 7}, "manual_params": {"max_depth": 4}}
    args = build_train_args(cfg, "manual", tmp_path, "run-1", "dev", "abc")
    assert args[args.index("--data-run-id") + 1] == "run-1"
    assert args[args.index("--max_depth") + 1] == "4"
    assert "--config-hash" in args
