"""Builds and submits SageMaker Automatic Model Tuning (AMT) jobs from an experiment config.

Used only for ``--kind hpo``. The search space is read from the experiment YAML, never
hardcoded. One MLflow parent run represents the whole search; each trial's ``train.py``
opens a child run under it (the parent id reaches the trial via ``MLFLOW_PARENT_RUN_ID``).
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

PARENT_RUN_ENV = "MLFLOW_PARENT_RUN_ID"
INTEGER_TYPE = "integer"
CONTINUOUS_TYPE = "continuous"
METRIC_REGEX = r"validation-aucpr:([0-9\.]+)"
MAX_TUNING_NAME_LEN = 32


def build_ranges(search_space: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    """Converts the YAML search space into AMT ``ParameterRanges``.

    Args:
        search_space: Mapping of hyperparameter name to ``{type, min, max}``
            (``type`` is ``integer`` or ``continuous``; optional ``scaling``).

    Returns:
        Dict with ``IntegerParameterRanges`` and ``ContinuousParameterRanges`` lists.

    Raises:
        ValueError: If a parameter has an unknown type or ``min >= max``.
    """
    ranges: dict[str, list[dict[str, str]]] = {"IntegerParameterRanges": [], "ContinuousParameterRanges": []}
    for name, spec in search_space.items():
        if spec["min"] >= spec["max"]:
            raise ValueError(f"search_space.{name}: min must be < max")
        entry = {"Name": name, "MinValue": str(spec["min"]), "MaxValue": str(spec["max"]),
                 "ScalingType": spec.get("scaling", "Auto")}
        if spec["type"] == INTEGER_TYPE:
            ranges["IntegerParameterRanges"].append(entry)
        elif spec["type"] == CONTINUOUS_TYPE:
            ranges["ContinuousParameterRanges"].append(entry)
        else:
            raise ValueError(f"search_space.{name}: unknown type '{spec['type']}'")
    return ranges


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
    cfg: dict[str, Any],
    training_request: dict[str, Any],
    parent_run_id: str,
) -> dict[str, Any]:
    """Builds the ``CreateHyperParameterTuningJob`` request from a training-job request.

    Args:
        name: Tuning job name from :func:`tuning_job_name`.
        cfg: Parsed experiment config (``search_space`` and ``tuning`` blocks).
        training_request: Request from ``build_training_job_request``; its hyperparameters
            become the static ones and its channels/resources are reused for every trial.
        parent_run_id: MLflow parent run id passed to trials as an environment variable.

    Returns:
        Keyword arguments for ``sagemaker_client.create_hyper_parameter_tuning_job``.
    """
    tuning = cfg["tuning"]
    tunable = set(cfg["search_space"])
    static = {k: v for k, v in training_request["HyperParameters"].items() if k not in tunable}
    return {
        "HyperParameterTuningJobName": name,
        "HyperParameterTuningJobConfig": {
            "Strategy": tuning["strategy"],
            "HyperParameterTuningJobObjective": {
                "Type": tuning["objective_type"], "MetricName": tuning["objective_metric"]},
            "ResourceLimits": {"MaxNumberOfTrainingJobs": int(tuning["max_jobs"]),
                               "MaxParallelTrainingJobs": int(tuning["max_parallel_jobs"])},
            "ParameterRanges": build_ranges(cfg["search_space"]),
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
