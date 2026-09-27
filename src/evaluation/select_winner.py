"""Selects the best AMT trial by VALIDATION metric and tags it ``winner=true``.

Runs after a tuning job. It never reads test data, never gates and never registers: ranking
on test would tune the model to the test set and make its score optimistic.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from mlflow.tracking import MlflowClient

from src.mlflow_tracking.mlflow_tracking import configure_mlflow_tracking

logger = logging.getLogger(__name__)

DEFAULT_METRIC = "validation_aucpr"
TABLE_PARAMS = ("max_depth", "min_child_weight", "gamma")


def pick_best(runs: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Returns the run with the highest value of ``metric``.

    Args:
        runs: Dicts with ``run_id``, ``metrics`` and ``params``.
        metric: Metric name to maximise.

    Returns:
        The best run. Runs missing the metric (or with NaN) are ignored.

    Raises:
        ValueError: If no run has a usable value for the metric.
    """
    scored = [r for r in runs if r["metrics"].get(metric) == r["metrics"].get(metric)
              and metric in r["metrics"]]
    if not scored:
        raise ValueError(f"no child run has metric '{metric}'")
    return max(scored, key=lambda r: r["metrics"][metric])


def comparison_table(runs: list[dict[str, Any]], metric: str, top: int = 10) -> str:
    """Renders the top trials as a Markdown table (for the CI job summary).

    Args:
        runs: Dicts with ``run_id``, ``metrics`` and ``params``.
        metric: Ranking metric.
        top: Number of rows.

    Returns:
        Markdown table text.
    """
    ranked = sorted((r for r in runs if metric in r["metrics"]), key=lambda r: -r["metrics"][metric])[:top]
    lines = [f"| run_id | {metric} | " + " | ".join(TABLE_PARAMS) + " |",
             "|---|---|" + "---|" * len(TABLE_PARAMS)]
    for r in ranked:
        params = " | ".join(str(r["params"].get(p, "")) for p in TABLE_PARAMS)
        lines.append(f"| {r['run_id'][:8]} | {r['metrics'][metric]:.4f} | {params} |")
    return "\n".join(lines)


def select_winner(parent_run_id: str, metric: str = DEFAULT_METRIC) -> str:
    """Finds the best child run of an AMT parent, tags it ``winner=true`` and prints the table.

    Args:
        parent_run_id: MLflow parent run of the tuning job.
        metric: Validation metric to maximise.

    Returns:
        The winning run id.

    Raises:
        ValueError: If the parent has no scored child runs.
    """
    client = MlflowClient()
    parent = client.get_run(parent_run_id)
    children = client.search_runs(
        [parent.info.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'")
    runs = [{"run_id": r.info.run_id, "metrics": r.data.metrics, "params": r.data.params} for r in children]
    best = pick_best(runs, metric)
    client.set_tag(best["run_id"], "winner", "true")
    client.set_tag(parent_run_id, "winner_run_id", best["run_id"])
    logger.info("step=select_winner status=complete parent=%s winner=%s %s=%.4f trials=%d",
                parent_run_id, best["run_id"], metric, best["metrics"][metric], len(runs))
    sys.stdout.write(comparison_table(runs, metric) + "\n")
    return best["run_id"]


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run-id", required=True)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--env", default="dev")
    parser.add_argument("--output-json", type=Path, default=None, help="Write {run_id} of the winner.")
    args = parser.parse_args(argv)
    configure_mlflow_tracking(args.env)
    winner = select_winner(args.parent_run_id, args.metric)
    if args.output_json:
        args.output_json.write_text(json.dumps({"run_id": winner}), encoding="utf-8")


if __name__ == "__main__":
    main()
