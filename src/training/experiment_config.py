# =======================================================================
# Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
# =======================================================================

"""Validates an experiment YAML and turns its ``params`` block into a run plan.

Every parameter under ``params`` is one of three shapes: a scalar (fixed), a list of discrete
values, or a ``{type, min, max}`` range. ``tuning.strategy`` (``grid``, ``random`` or
``bayesian``) states how the lists and ranges are searched; a file with only scalars is a
single training job and carries no ``tuning`` block. Checking all of this before any job is
submitted turns a typo, a mis-indented parameter or an impossible combination into an
immediate, readable error instead of a silently wrong or wasted SageMaker run.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

SINGLE = "single"
STRATEGIES = ("grid", "random", "bayesian")
ALLOWED_TOP_LEVEL = frozenset({"experiment", "data", "runtime", "params", "tuning"})
REQUIRED_TUNING_KEYS = ("strategy", "objective_metric", "objective_type", "max_parallel_jobs")
RANGE_TYPES = ("integer", "continuous")
RANGE_KEYS = frozenset({"type", "min", "max", "scaling"})
SCALINGS = ("Auto", "Linear", "Logarithmic")


@dataclass(frozen=True)
class ExperimentPlan:
    """What an experiment file asks the launcher to run.

    Attributes:
        strategy: ``single``, ``grid``, ``random`` or ``bayesian``.
        fixed: Scalar parameters, identical in every trial.
        lists: Parameter name to its discrete values (two or more).
        ranges: Parameter name to its validated ``{type, min, max[, scaling]}`` spec.
        combinations: Product of the list lengths (1 when there are no lists).
        trials: Number of training jobs this plan will launch.
    """

    strategy: str
    fixed: dict[str, Any]
    lists: dict[str, list[Any]] = field(default_factory=dict)
    ranges: dict[str, dict[str, Any]] = field(default_factory=dict)
    combinations: int = 1
    trials: int = 1

    @property
    def tunable(self) -> set[str]:
        """Names of every parameter the tuner chooses (lists and ranges)."""
        return set(self.lists) | set(self.ranges)


def validate_experiment(cfg: dict[str, Any]) -> ExperimentPlan:
    """Validates an experiment config and resolves it into a run plan.

    Args:
        cfg: Parsed experiment YAML.

    Returns:
        The plan: ``single`` for an all-scalar file, otherwise the explicit
        ``tuning.strategy`` with its fixed, list and range parameters.

    Raises:
        ValueError: If a top-level key is unknown, ``experiment.name`` or ``params`` is
            missing, a list or range is malformed, ``tuning`` is missing, stray or
            incomplete, the strategy is unknown, ``grid`` is used with a range, or
            ``max_jobs`` is missing (random/bayesian) or differs from the grid size.
    """
    unknown = sorted(set(cfg) - ALLOWED_TOP_LEVEL)
    if unknown:
        raise ValueError(
            f"unknown top-level keys: {', '.join(unknown)} (allowed: {', '.join(sorted(ALLOWED_TOP_LEVEL))}); "
            "check indentation, parameters belong under params"
        )
    name = (cfg.get("experiment") or {}).get("name")
    if not name:
        raise ValueError("experiment.name is required (it tags every MLflow run of this experiment)")
    params = cfg.get("params")
    if not isinstance(params, dict) or not params:
        raise ValueError("params is required and must be a non-empty mapping")

    fixed, lists, ranges = classify_params(params)
    combinations = math.prod(len(values) for values in lists.values())
    tuning = cfg.get("tuning")

    if not lists and not ranges:
        if tuning is not None:
            raise ValueError("tuning is set but params has no list or range to tune; remove tuning for a single run")
        plan = ExperimentPlan(strategy=SINGLE, fixed=fixed)
    else:
        strategy, trials = _validate_tuning(tuning, bool(ranges), combinations)
        plan = ExperimentPlan(strategy=strategy, fixed=fixed, lists=lists, ranges=ranges,
                              combinations=combinations, trials=trials)

    logger.info("step=validate_experiment status=ok experiment=%s strategy=%s combinations=%d trials=%d",
                name, plan.strategy, plan.combinations, plan.trials)
    return plan


def validate_production(cfg: dict[str, Any]) -> dict[str, Any]:
    """Returns the fixed parameters of the approved production config.

    Args:
        cfg: Config merged from the experiment defaults and ``config/production/model.yaml``.

    Returns:
        The scalar parameters to retrain with.

    Raises:
        ValueError: If ``params`` is empty (training would silently use library defaults)
            or contains a list or range (production retrains one approved point).
    """
    params = cfg.get("params")
    if not isinstance(params, dict) or not params:
        raise ValueError("production params are empty; copy the approved winner's values into config/production/model.yaml")
    fixed, lists, ranges = classify_params(params)
    if lists or ranges:
        raise ValueError(f"production params must be scalars, found lists/ranges: {', '.join(sorted(set(lists) | set(ranges)))}")
    return fixed


def classify_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[Any]], dict[str, dict[str, Any]]]:
    """Splits ``params`` into fixed scalars, discrete lists and ranges by YAML shape.

    A one-element list is treated as a fixed value, so it never creates a one-point search.

    Args:
        params: The ``params`` mapping of an experiment config.

    Returns:
        Tuple ``(fixed, lists, ranges)``.

    Raises:
        ValueError: If a value is null, a list is empty/duplicated/holds a bool, null or
            nested value, or a range spec is malformed.
    """
    fixed: dict[str, Any] = {}
    lists: dict[str, list[Any]] = {}
    ranges: dict[str, dict[str, Any]] = {}
    for name, value in params.items():
        if isinstance(value, list):
            _validate_list(name, value)
            if len(value) == 1:
                logger.info("step=classify_params status=collapsed param=%s reason=single_value_list", name)
                fixed[name] = value[0]
            else:
                lists[name] = value
        elif isinstance(value, dict):
            _validate_range(name, value)
            ranges[name] = value
        elif value is None:
            raise ValueError(f"params.{name} is null")
        else:
            fixed[name] = value
    return fixed, lists, ranges


def _validate_tuning(tuning: Any, has_ranges: bool, combinations: int) -> tuple[str, int]:
    """Checks the ``tuning`` block against the params it must search.

    Args:
        tuning: The raw ``tuning`` value from the config.
        has_ranges: Whether ``params`` contains at least one range.
        combinations: Product of the list lengths.

    Returns:
        Tuple ``(strategy, trials)`` with the strategy lower-cased.

    Raises:
        ValueError: On a missing or incomplete block, unknown strategy, grid over a
            range, a missing ``max_jobs`` for random/bayesian, or a grid ``max_jobs``
            that differs from the combination count.
    """
    if not isinstance(tuning, dict):
        raise ValueError(  # noqa: TRY004 - config errors are ValueError by contract
            "params has lists or ranges, so a tuning block with a strategy is required")
    missing = [key for key in REQUIRED_TUNING_KEYS if key not in tuning]
    if missing:
        raise ValueError(f"tuning is missing required keys: {', '.join(missing)}")
    strategy = str(tuning["strategy"]).lower()
    if strategy not in STRATEGIES:
        raise ValueError(f"tuning.strategy={tuning['strategy']!r} is not one of {', '.join(STRATEGIES)}")

    max_jobs = tuning.get("max_jobs")
    if strategy == "grid":
        if has_ranges:
            raise ValueError("tuning.strategy=grid supports lists only; use random or bayesian for ranges")
        if max_jobs is not None and int(max_jobs) != combinations:
            raise ValueError(f"grid has {combinations} combinations but tuning.max_jobs={max_jobs}; "
                             "remove max_jobs or set it to the combination count")
        return strategy, combinations
    if max_jobs is None:
        raise ValueError(f"tuning.max_jobs is required for strategy={strategy}")
    return strategy, int(max_jobs)


def _validate_list(name: str, values: list[Any]) -> None:
    """Checks one list-valued parameter.

    Args:
        name: Parameter name, used in error messages.
        values: The list from the YAML.

    Raises:
        ValueError: If the list is empty, holds a bool/null/nested value, or has duplicates.
    """
    if not values:
        raise ValueError(f"params.{name} is an empty list")
    bad = [v for v in values if v is None or isinstance(v, (bool, list, dict))]
    if bad:
        raise ValueError(f"params.{name} list entries must be numbers or strings, got {bad!r}")
    if len({str(v) for v in values}) != len(values):
        raise ValueError(f"params.{name} list has duplicate values: {values!r}")


def _validate_range(name: str, spec: dict[str, Any]) -> None:
    """Checks one range-valued parameter.

    Args:
        name: Parameter name, used in error messages.
        spec: The ``{type, min, max[, scaling]}`` mapping from the YAML.

    Raises:
        ValueError: On unknown keys, a missing or unknown ``type``, non-numeric or
            non-integer bounds, ``min >= max``, an unknown ``scaling``, or
            ``Logarithmic`` scaling with ``min <= 0``.
    """
    extra = sorted(set(spec) - RANGE_KEYS)
    if extra:
        raise ValueError(f"params.{name} has unknown range keys: {', '.join(extra)}")
    kind = spec.get("type")
    if kind not in RANGE_TYPES:
        raise ValueError(f"params.{name}.type must be one of {', '.join(RANGE_TYPES)}, got {kind!r}")
    low, high = spec.get("min"), spec.get("max")
    for bound in (low, high):
        if isinstance(bound, bool) or not isinstance(bound, (int, float)):
            raise ValueError(f"params.{name} min and max must be numbers")  # noqa: TRY004
        if kind == "integer" and not isinstance(bound, int):
            raise ValueError(f"params.{name} is an integer range, min and max must be whole numbers")
    if low >= high:
        raise ValueError(f"params.{name}: min must be < max")
    scaling = spec.get("scaling", "Auto")
    if scaling not in SCALINGS:
        raise ValueError(f"params.{name}.scaling must be one of {', '.join(SCALINGS)}, got {scaling!r}")
    if scaling == "Logarithmic" and low <= 0:
        raise ValueError(f"params.{name}: Logarithmic scaling needs min > 0")
