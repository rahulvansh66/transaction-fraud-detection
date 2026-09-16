---
name: ml-lineage-reproducibility
description: Checklist for making a training/pipeline run traceable and reproducible — what to log in MLflow, which S3 paths must be immutable, and how runs should be structured and tagged. Use whenever touching train.py, pipeline.py, an evaluation or processing step, or any MLflow logging code, and whenever adding a new pipeline step to check what lineage it should record. Also use when a model can't be traced back to the exact code/config/data/hyperparameters that produced it, or when deciding how a training run should reference its data snapshot.
---

# ML lineage and reproducibility

Source: [DOCS/ok/mlflow-and-reproducibility.md](../../../DOCS/ok/mlflow-and-reproducibility.md)
(current gaps, kept up to date there) and
[DOCS/ok/production_ml_experimentation_git_sagemaker_amt_mlflow.md](../../../DOCS/ok/production_ml_experimentation_git_sagemaker_amt_mlflow.md)
§15–16 (the target lineage model). This skill is the checklist version —
check the source doc for current wiring status before assuming something is
already done.

## The question every model must be able to answer

> Given this model, what exact code, config, data, environment, and
> hyperparameters produced it, and what did it score?

If you're adding or editing anything that logs to MLflow, or a pipeline step
that produces a model or intermediate artifact, check it captures its slice
of this list:

| Lineage input | Log as |
| --- | --- |
| Code version | git commit SHA, as a run tag |
| Config version | env name + hash of the merged config; attach the resolved config as an artifact |
| Data version | exact input snapshot(s) — see "immutable data paths" below — via `mlflow.log_input` |
| Pipeline version | pipeline-definition hash + `PipelineExecutionArn`, as a tag |
| Environment/container | image URI **with digest**; locked dependency versions (`pip freeze` or hash) |
| Execution parameters | every SageMaker Pipeline Parameter value actually used for this run |
| Hyperparameters | the full set, not a curated subset |
| Random seeds | set and log `random_state` explicitly wherever training has randomness |
| Metrics | train + val + **test**, both threshold-free (AUPRC/ROC-AUC/recall@precision) and at the chosen operating threshold |
| Artifacts | model, plus anything the model depends on to run (scaler, feature list), plus an MLflow model signature + input example |

A run missing several of these isn't "minimal" — it's a run nobody can later
explain or reproduce. If you're adding a new step (a new processing job, a
new evaluation stage), it should log its slice of this table even if it logs
nothing else.

## Immutable data paths — the rule that makes the rest of this meaningful

None of the above matters if a re-run can read different bytes than the
original run did. Never read a moving prefix like `s3://bucket/latest/`.
Instead:

- Read a specific dated/versioned snapshot (e.g. `aggregates/asof=<date>/`),
  passed in as a parameter, not resolved implicitly at run time.
- If reading partitioned data (e.g. `curated/ingestion_date=...`), take an
  explicit list of partitions (or an explicit `<= date` bound) as a
  parameter, and log which partitions were actually read.
- Once a derived dataset (train/val/test split) is written, treat its path as
  permanent and immutable — never overwrite it in place.

If you're writing a step that resolves "the latest X" internally, stop and
make that resolution an explicit, logged parameter instead.

## Run structure

Use one **parent run per pipeline execution**, with each pipeline step
(prepare, train, evaluate, ...) logging as a **nested run** underneath it.
This is what lets you go from "this model in the registry" to "the exact
data-quality outcome, class balance, and split boundaries" in one place,
instead of stitching together unrelated runs.

Tag every run with a `mode` of `manual`, `hpo`, or `production` — this is how
you distinguish a deliberate one-off experiment from an AMT trial from a
scheduled retrain later, without relying on naming conventions.

For an AMT search specifically: each trial should be its own child run under
the tuning job's parent run, carrying the trial's generated hyperparameters —
never hand-write trial values into a config (see [[ml-experimentation-workflow]]).

## Open decision: registry vs. model package group

This repo has not yet settled whether **MLflow Model Registry** or
**SageMaker Model Package Group** is the source of truth for promotion state
(`None → Staging → Production`). Don't silently pick one while implementing
something else — if you're wiring registration/promotion logic, treat this
as a decision to surface, not to make implicitly. Check
[DOCS/ok/mlflow-and-reproducibility.md](../../../DOCS/ok/mlflow-and-reproducibility.md)
§4 for the current state before proceeding.

## Where to look next

- Whether this run should even exist as manual vs. HPO vs. production → [[ml-experimentation-workflow]]
- Where the config/code producing this run should physically live → [[ml-repo-structure]]
