"""XGBoost parameter building and fitting (pure logic, no I/O or MLflow)."""

from __future__ import annotations

from typing import Any

import xgboost as xgb

INT_PARAMS = frozenset({"max_depth"})
CONTROL_KEYS = frozenset({"num_round", "early_stopping_rounds", "min_precision"})


def build_params(
    static_params: dict[str, Any], hp_overrides: dict[str, Any], scale_pos_weight: float
) -> dict[str, Any]:
    """Merges static params, tuner-provided hyperparameters and the class weight.

    Overrides win over static params. Control keys (``num_round``, ...) are removed
    because they are not XGBoost booster parameters.

    Args:
        static_params: Fixed parameters from the experiment config.
        hp_overrides: Hyperparameters supplied by AMT or the manual config.
        scale_pos_weight: Negative/positive ratio computed by preprocessing.

    Returns:
        Booster parameter dictionary.
    """
    merged = {**static_params, **hp_overrides}
    params = {k: v for k, v in merged.items() if k not in CONTROL_KEYS}
    for key in INT_PARAMS & params.keys():
        params[key] = int(params[key])
    params.setdefault("scale_pos_weight", scale_pos_weight)
    return params


def fit(
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    params: dict[str, Any],
    num_round: int,
    seed: int,
    early_stopping_rounds: int = 20,
) -> xgb.Booster:
    """Trains a booster with early stopping on the validation set.

    Args:
        dtrain: Training matrix.
        dval: Validation matrix (last eval set drives early stopping).
        params: Booster parameters from :func:`build_params`.
        num_round: Maximum boosting rounds.
        seed: Random seed, set explicitly for reproducibility.
        early_stopping_rounds: Rounds without validation improvement before stopping.

    Returns:
        The trained booster.
    """
    return xgb.train(
        {**params, "seed": seed},
        dtrain,
        num_round,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=10,
    )
