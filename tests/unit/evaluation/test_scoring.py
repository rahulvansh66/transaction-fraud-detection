# =======================================================================
# Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
# =======================================================================

"""Unit tests that test scoring uses the same trees training used for threshold selection."""

import numpy as np
import pytest
import xgboost as xgb

from src.evaluation import evaluate
from src.training.model import build_params, fit


def test_best_iteration_param_parsing() -> None:
    """Logged params parse to an int; missing or null values mean an older run."""
    assert evaluate.best_iteration_of({"best_iteration": "7"}) == 7
    assert evaluate.best_iteration_of({"best_iteration": "None"}) is None
    assert evaluate.best_iteration_of({}) is None


def test_score_uses_best_iteration_trees(monkeypatch: pytest.MonkeyPatch) -> None:
    """With early stopping, scoring is restricted to the trees up to best_iteration."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 3))
    y = (x[:, 0] + rng.normal(scale=1.0, size=300) > 0).astype(int)
    dtrain, dval = xgb.DMatrix(x[:200], label=y[:200]), xgb.DMatrix(x[200:], label=y[200:])
    params = build_params({"objective": "binary:logistic", "eval_metric": "aucpr", "eta": 0.5}, {}, 1.0)
    booster = fit(dtrain, dval, params, 200, seed=1, early_stopping_rounds=3)
    assert booster.num_boosted_rounds() > booster.best_iteration + 1  # trees exist past the best round
    monkeypatch.setattr(evaluate.mlflow.xgboost, "load_model", lambda uri: booster)

    seen: list[np.ndarray] = []
    monkeypatch.setattr(evaluate, "compute_metrics", lambda yt, scores, thr: seen.append(scores) or {})
    evaluate.score("models:/m", x[200:], y[200:], 0.5, booster.best_iteration)
    evaluate.score("models:/m", x[200:], y[200:], 0.5, None)

    expected = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
    assert np.allclose(seen[0], expected)
    assert not np.allclose(seen[0], seen[1])  # all-trees scores differ, so the range matters
