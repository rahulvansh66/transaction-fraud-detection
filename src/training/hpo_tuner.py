# =======================================================================
# Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
# =======================================================================

"""Builds and submits SageMaker Automatic Model Tuning (AMT) jobs from an experiment plan.

Used for every experiment whose plan is not ``single``. Lists and ranges come from the
validated experiment YAML (see ``experiment_config``), never hardcoded. One MLflow parent
run represents the whole search; each trial's ``train.py`` opens a child run under it (the
parent id reaches the trial via ``MLFLOW_PARENT_RUN_ID``).
"""

from __future__ import annotations

import logging
from typing import Any

from src.training.experiment_config import ExperimentPlan

logger = logging.getLogger(__name__)

PARENT_RUN_ENV = "MLFLOW_PARENT_RUN_ID"
INTEGER_TYPE = "integer"
CONTINUOUS_TYPE = "continuous"
METRIC_REGEX = r"validation-aucpr:([0-9eE+\-\.]+)"
MAX_TUNING_NAME_LEN = 32
AMT_STRATEGIES = {"grid": "Grid", "random": "Random", "bayesian": "Bayesian"}


def build_ranges(
    lists: dict[str, list[Any]], ranges: dict[str, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Converts validated lists and ranges into AMT ``ParameterRanges``.

    Args:
        lists: Parameter name to its discrete values; sent as categorical ranges, with
            values stringified because AMT requires strings (``train.py`` parses them back).
        ranges: Parameter name to ``{type, min, max}`` (``type`` is ``integer`` or
            ``continuous``; optional ``scaling``). Bounds are assumed already validated.

    Returns:
        Dict with ``IntegerParameterRanges``, ``ContinuousParameterRanges`` and
        ``CategoricalParameterRanges`` lists.

    Raises:
        ValueError: If a range has an unknown type.
    """
    out: dict[str, list[dict[str, Any]]] = {
        "IntegerParameterRanges": [], "ContinuousParameterRanges": [], "CategoricalParameterRanges": []}
    for name, values in lists.items():
        out["CategoricalParameterRanges"].append({"Name": name, "Values": [str(v) for v in values]})
    for name, spec in ranges.items():
        entry = {"Name": name, "MinValue": str(spec["min"]), "MaxValue": str(spec["max"]),
                 "ScalingType": spec.get("scaling", "Auto")}
        if spec["type"] == INTEGER_TYPE:
            out["IntegerParameterRanges"].append(entry)
        elif spec["type"] == CONTINUOUS_TYPE:
            out["ContinuousParameterRanges"].append(entry)
        else:
            raise ValueError(f"params.{name}: unknown type '{spec['type']}'")
    return {kind: entries for kind, entries in out.items() if entries}


def tuning_job_name(config_hash: str, timestamp: str) -> str:
    """Builds a tuning job name within AMT's 32-character limit.

    Args:
        config_hash: Experiment config hash.
        timestamp: Compact timestamp such as ``0926103015``.

    Returns:
        ``hpo-<hash8>-<timestamp>``.

    Raises:
        ValueError: If the name would exceed 32 characters.
    """
    name = f"hpo-{config_hash[:8]}-{timestamp}"
    if len(name) > MAX_TUNING_NAME_LEN:
        raise ValueError(f"tuning job name too long: {name}")
    return name


def build_tuning_request(
    name: str,
    tuning: dict[str, Any],
    plan: ExperimentPlan,
    training_request: dict[str, Any],
    parent_run_id: str,
) -> dict[str, Any]:
    """Builds the ``CreateHyperParameterTuningJob`` request from a training-job request.

    Args:
        name: Tuning job name from :func:`tuning_job_name`.
        tuning: The experiment's ``tuning`` block (objective, parallelism).
        plan: Validated plan from ``validate_experiment``; supplies strategy, lists, ranges
            and the trial count.
        training_request: Request from ``build_training_job_request``; its hyperparameters
            become the static ones and its channels/resources are reused for every trial.
        parent_run_id: MLflow parent run id passed to trials as an environment variable.

    Returns:
        Keyword arguments for ``sagemaker_client.create_hyper_parameter_tuning_job``.
        Grid requests omit ``MaxNumberOfTrainingJobs`` because AMT derives it from the grid.
    """
    static = {k: v for k, v in training_request["HyperParameters"].items() if k not in plan.tunable}
    limits: dict[str, int] = {"MaxParallelTrainingJobs": int(tuning["max_parallel_jobs"])}
    if plan.strategy != "grid":
        limits["MaxNumberOfTrainingJobs"] = plan.trials
    return {
        "HyperParameterTuningJobName": name,
        "HyperParameterTuningJobConfig": {
            "Strategy": AMT_STRATEGIES[plan.strategy],
            "HyperParameterTuningJobObjective": {
                "Type": tuning["objective_type"], "MetricName": tuning["objective_metric"]},
            "ResourceLimits": limits,
            "ParameterRanges": build_ranges(plan.lists, plan.ranges),
        },
        "TrainingJobDefinition": {
            "StaticHyperParameters": static,
            "AlgorithmSpecification": {
                **training_request["AlgorithmSpecification"],
                "MetricDefinitions": [{"Name": tuning["objective_metric"], "Regex": METRIC_REGEX}],
            },
            "RoleArn": training_request["RoleArn"],
            "InputDataConfig": training_request["InputDataConfig"],
            "OutputDataConfig": training_request["OutputDataConfig"],
            "ResourceConfig": training_request["ResourceConfig"],
            "StoppingCondition": training_request["StoppingCondition"],
            "Environment": {**training_request["Environment"], PARENT_RUN_ENV: parent_run_id},
        },
    }
