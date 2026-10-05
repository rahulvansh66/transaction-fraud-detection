# =======================================================================
# Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
# =======================================================================

"""Validates an experiment YAML against the run kind it is launched with.

Each file under ``config/experiments/`` is self-contained and serves exactly one kind:
a ``manual`` file carries ``manual_params`` and an ``hpo`` file carries ``search_space``
and ``tuning``. Checking this before any job is submitted turns a mismatched file or a
missing block into an immediate, readable error instead of a silent default-parameter run
or a ``KeyError`` deep inside the AMT request builder.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

REQUIRED_TUNING_KEYS = ("objective_metric", "objective_type", "strategy", "max_jobs", "max_parallel_jobs")
KINDS_WITH_EXPERIMENT_FILE = ("manual", "hpo")


def validate_experiment(cfg: dict[str, Any], kind: str) -> None:
    """Checks that an experiment config is consistent with the run kind.

    Production configs come from ``config/production/model.yaml`` and are not validated here.

    Args:
        cfg: Parsed experiment config.
        kind: ``manual``, ``hpo`` or ``production``.

    Raises:
        ValueError: If ``experiment.name`` is missing, ``experiment.mode`` differs from
            ``kind``, the block the kind needs is missing or empty, a tuning key is
            missing, or a tuned/manual parameter is also declared in ``static_params``.
    """
    if kind not in KINDS_WITH_EXPERIMENT_FILE:
        return

    experiment = cfg.get("experiment") or {}
    if not experiment.get("name"):
        raise ValueError("experiment.name is required (it tags every MLflow run of this experiment)")
    if experiment.get("mode") != kind:
        raise ValueError(
            f"experiment.mode={experiment.get('mode')!r} does not match --kind {kind}; "
            f"use the experiment file written for {kind} runs"
        )

    static = set(cfg.get("static_params") or {})
    if kind == "manual":
        manual = cfg.get("manual_params") or {}
        if not manual:
            raise ValueError("manual_params is required and must be non-empty for --kind manual")
        _reject_overlap("manual_params", set(manual), static)
        ignored = [block for block in ("search_space", "tuning") if block in cfg]
    else:
        space = cfg.get("search_space") or {}
        if not space:
            raise ValueError("search_space is required and must be non-empty for --kind hpo")
        missing = [key for key in REQUIRED_TUNING_KEYS if key not in (cfg.get("tuning") or {})]
        if missing:
            raise ValueError(f"tuning is missing required keys: {', '.join(missing)}")
        _reject_overlap("search_space", set(space), static)
        ignored = [block for block in ("manual_params",) if block in cfg]

    if ignored:
        logger.warning("step=validate_experiment status=ignored_blocks kind=%s blocks=%s",
                       kind, ",".join(ignored))
    logger.info("step=validate_experiment status=ok experiment=%s kind=%s", experiment["name"], kind)


def _reject_overlap(block: str, names: set[str], static: set[str]) -> None:
    """Rejects parameters declared both in a run block and in ``static_params``.

    Args:
        block: Name of the block being checked, used in the error message.
        names: Parameter names declared in that block.
        static: Parameter names declared in ``static_params``.

    Raises:
        ValueError: If the two sets share any parameter name.
    """
    overlap = sorted(names & static)
    if overlap:
        raise ValueError(f"{block} repeats static_params keys: {', '.join(overlap)}")
