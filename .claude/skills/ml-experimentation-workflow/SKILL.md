---
name: ml-experimentation-workflow
description: Decides whether a new ML experiment should be a manual run, an AMT hyperparameter search, or production retraining, and how Git/Pipeline Parameters/AMT/MLflow/approval fit together. Use whenever writing or editing an experiment config under config/experiments/, a SageMaker pipeline or AMT tuning step, or a GitHub Actions workflow that triggers a pipeline execution. Also use when deciding whether a promising manual result should become an HPO search space, whether an AMT winner is ready to promote, or whether a scheduled retrain needs to rerun HPO.
---

# ML experimentation workflow

Source: [DOCS/my-notes/production_ml_experimentation_git_sagemaker_amt_mlflow.md](../../../DOCS/my-notes/production_ml_experimentation_git_sagemaker_amt_mlflow.md)
and [DOCS/my-notes/best of his repo and our hpo approach.md](../../../DOCS/my-notes/best%20of%20his%20repo%20and%20our%20hpo%20approach.md).
Read those for the full diagrams; this skill is the decision procedure.

## The core mental model

Not every experiment needs AMT. There are three modes, and picking the wrong
one either wastes compute or skips rigor you actually need:

```
MANUAL EXPERIMENTS   — "What should I try?" (a deliberate, specific comparison)
HPO (AMT)            — "Search a promising space systematically."
PRODUCTION RETRAINING — "Train the already-approved config on new data."
```

All three feed the same downstream path: best/approved configuration →
final evaluation → MLflow Model Registry → approval → production.

## Deciding manual vs. HPO

Use **manual** (a new `config/experiments/experiment-NNN-<slug>.yaml` with
`mode: manual`, run via `--kind manual`, no AMT) when the question is a
deliberate, discrete point or comparison an engineer should reason about
directly (a single fixed point is also a valid baseline or smoke test):

```
optimizer = Adam        vs  optimizer = AdamW
feature_set = A         vs  feature_set = B
architecture = A        vs  architecture = B
```

Use **AMT** when you already have a continuous or combinatorial space worth
searching systematically, *and* ideally you already have manual results
narrowing that space. Don't hand AMT a huge uninformed range on day one —
run a few manual points first:

```
Stage 1 (manual): lr=.001 → .72, lr=.0005 → .76, lr=.0001 → .71
                  → promising region is near .0005
Stage 2 (AMT):    lr ∈ [.0003, .002], batch ∈ {32,64,128}, dropout ∈ [0,.2]
                  → 30 trials → best = .80
```

## Experiment file convention (manual stage, then search stage)

**Experiment numbers are globally sequential and never reused: every file under
`config/experiments/` gets the next unused number, whatever its mode.** A manual
file and the search-space file derived from it are two experiments with
consecutive numbers, never two files sharing one number. Before creating a file,
list `config/experiments/` and take highest number + 1.

```
config/experiments/
  experiment-002-manual-params.yaml   # mode: manual, carries manual_params (one fixed point)
  experiment-003-search-space.yaml    # mode: hpo, carries search_space + tuning, ranges
                                      #   informed by the manual result
```

- **Filename = number + stage slug; `experiment.name` = `exp-NNN-<slug>`**, and
  `experiment.mode` (`manual` | `hpo`) must match the `--kind` it is launched
  with. `experiment.name` is tagged on every MLflow run as `experiment_name`, so
  manual runs and AMT trials stay separable. A search-space file's header comment
  names the manual experiment its ranges follow.
- **Files are self-contained**: no `_base.yaml` or `extends`. Each carries its own
  `data`, `runtime` and `static_params`. When copying a file to start a new one,
  diff against the previous so `seed`, `eval_metric` and `data.run_id` only
  change on purpose; keep them identical between a round's manual and search files.
- **A file carries only the block its mode needs**: manual -> `manual_params`;
  hpo -> `search_space` + `tuning` (`objective_metric`, `objective_type`,
  `strategy`, `max_jobs`, `max_parallel_jobs`). Keys must not repeat `static_params`.
  `src/training/experiment_config.py::validate_experiment` enforces this before any
  job is submitted.
- **One manual file = one point.** To try another point or iterate on ranges, start
  the next number (`experiment-003-...`); never edit a file after it has produced runs.
- **Derive ranges from evidence**: centre each range on what the manual runs showed
  to be good, with margin; use `scaling: Logarithmic` for strictly positive
  parameters spanning orders of magnitude (`eta`, `lambda`, `alpha`); drop
  parameters shown to be flat; include the manual best inside the range. Budget
  roughly 10-20 trials per tuned parameter for a real Bayesian search.

The responsibility split, always:

- **Git** stores the search-space *definition* (the YAML under `config/experiments/`).
- **AMT** generates and evaluates the individual *trials* — never write trial
  values by hand into Git.
- **Pipeline Parameters** carry runtime values (dataset version, experiment
  name, HPO limits, search-space bounds) from Git into the pipeline execution
  — but only values that genuinely vary between runs. Don't promote every
  YAML field to a Pipeline Parameter; that turns them into a second,
  untracked config system.

## Never auto-promote

An AMT winner (or a manual result, for that matter) is a candidate, not a
decision. Always route it through:

```
winner → final evaluation → gate
           ├─ FAIL → reject
           └─ PASS → MLflow Model Registry → approval → production
```

Gates can include offline metrics, regression tests, data-quality checks,
robustness checks, business KPI checks, latency/resource checks, and
safety/fairness checks where relevant. If you're wiring a new pipeline and
the register step isn't conditioned on an evaluation step passing, that's a
gap — see [[ml-lineage-reproducibility]] for the `ConditionStep` pattern.

## Production retraining is not HPO

Once a configuration is approved, write it explicitly to
`config/production/model.yaml` and retrain against *that*, on newly
approved data — don't rerun AMT on a schedule "just in case." HPO answers
"what configuration should we use?"; production retraining answers "train
the approved configuration on the latest approved data." Treat a request to
add HPO back into a scheduled retrain job as a deliberate exception, not the
default.

## CI/CD trigger shape

A GitHub Actions workflow that fires a pipeline execution should be passing
runtime values in, not baking decisions into the workflow file itself:

```
Git config change → GitHub Actions → SageMaker Pipeline execution
                                        ├─ Pipeline Parameters (manual mode)
                                        └─ AMT tuning step (hpo mode)
```

If a workflow file starts containing hyperparameter values or search-space
bounds directly, move them back into the relevant `config/experiments/*.yaml`
and pass them through as parameters instead — see [[ml-repo-structure]] for
why that separation matters.

## Where to look next

- Where a new experiment/production config file should physically live → [[ml-repo-structure]]
- What each run (manual or AMT trial) must log in MLflow, and immutable data rules → [[ml-lineage-reproducibility]]
