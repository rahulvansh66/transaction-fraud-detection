---
name: ml-experimentation-workflow
description: Decides whether a new ML experiment should be a single run, a grid sweep, an AMT hyperparameter search, or production retraining, and how Git/AMT/MLflow/approval fit together. Use whenever writing or editing an experiment config under config/experiments/, a SageMaker pipeline or AMT tuning step, or a GitHub Actions workflow that triggers a training run. Also use when deciding which strategy (grid, random, bayesian) a search file should use, whether an AMT winner is ready to promote, or whether a scheduled retrain needs to rerun HPO.
---

# ML experimentation workflow

Source: [DOCS/my-notes/production_ml_experimentation_git_sagemaker_amt_mlflow.md](../../../DOCS/my-notes/production_ml_experimentation_git_sagemaker_amt_mlflow.md)
and [DOCS/my-notes/best of his repo and our hpo approach.md](../../../DOCS/my-notes/best%20of%20his%20repo%20and%20our%20hpo%20approach.md).
Those notes predate the unified `params:` schema below (they describe separate manual/hpo
files); this skill is the current decision procedure.

## The core mental model

Every experiment file is one question. The shape of its `params` block says what kind:

```
SINGLE RUN   — all params scalar. "What does this exact setting score?" (baseline, smoke test)
GRID         — lists + strategy: grid. "What is the impact of these specific values?"
SEARCH       — ranges (and/or lists) + strategy: random|bayesian. "Find a good setting."
PRODUCTION   — config/production/model.yaml. "Train the already-approved point on new data."
```

All of them feed the same downstream path: best/approved configuration →
final evaluation → MLflow Model Registry → approval → production.

## The params block (one mechanism for every kind)

Each entry under `params` is classified by its YAML shape (`src/training/experiment_config.py`):

```yaml
params:
  eta: 0.1                                   # scalar -> fixed in every trial
  max_depth: [2, 5, 7]                       # list   -> discrete values
  gamma: {type: continuous, min: 0, max: 3}  # dict   -> range (`type` required)
tuning:                                      # present only when something is tunable
  strategy: grid                             # grid | random | bayesian, always explicit
  objective_metric: validation:aucpr
  objective_type: Maximize
  max_parallel_jobs: 2
  max_jobs: 3
```

- All scalars and no `tuning` block: one SageMaker training job, no AMT.
- `strategy: grid`: lists only (AMT Grid supports categorical only). Every combination runs once.
  `max_jobs` is optional; if set it must equal the combination count.
- `strategy: random` / `bayesian`: lists, ranges or both; `max_jobs` required. List values are
  sampled, not guaranteed to all be tried; use grid when every value must be tested.
- The strategy is never inferred and never switched automatically. The launcher logs
  `strategy`, `combinations` and `trials` before submitting so cost is visible.
- Only `experiment`, `data`, `runtime`, `params`, `tuning` are allowed at top level; a mis-indented
  parameter is an error, not silently ignored. `validate_experiment` enforces all of this before
  any job is submitted.

## Choosing the kind

- **Single run** for a baseline, a smoke test, or one deliberate comparison point.
- **Grid** for a sensitivity study of a few chosen values (e.g. `max_depth: [2, 5, 7]`, everything
  else fixed). This is a one-factor-at-a-time study, not optimisation. Seed lists
  (`seed: [1, 2, 3]`) measure noise but multiply cost; the winner is still the best single trial.
- **Bayesian** for a systematic search over ranges once you know which parameters matter.
  Prefer wide ranges around sensible defaults over ranges narrowed from one point: a single
  result gives no direction. Budget roughly 10-20 trials per tuned parameter for a real search
  (Bayesian with only a few trials behaves like random search).
- **Random** only when explicitly chosen (cheap exploration, many categorical values).
- Don't hand AMT a huge uninformed range on day one: run a baseline and, if useful, a grid first.

## Experiment file convention

**Experiment numbers are globally sequential and never reused: every file under
`config/experiments/` gets the next unused number, whatever its kind.** Before creating a file,
list `config/experiments/` and take highest number + 1.

```
config/experiments/
  experiment-002-manual-params.yaml       # all scalars: single baseline run
  experiment-003-search-space.yaml        # ranges + strategy: bayesian
  experiment-004-depth-sensitivity.yaml   # list + strategy: grid
```

- **Filename = number + slug; `experiment.name` = `exp-NNN-<slug>`.** `experiment.name` is tagged
  on every MLflow run as `experiment_name`; the MLflow `mode` tag is the resolved strategy
  (`single|grid|random|bayesian|production`), so runs of different kinds stay separable.
- **Files are self-contained**: no `_base.yaml` or `extends`. Each carries its own `data`,
  `runtime` and `params`. Fixed and tuned values live in the same block, so a parameter cannot be
  declared as both. When copying a file to start a new one, diff against the previous so `seed`,
  `eval_metric` and `data.run_id` only change on purpose.
- **One file = one question.** To try different values or ranges, start the next number; never
  edit a file after it has produced runs.
- Use `scaling: Logarithmic` for strictly positive range params spanning orders of magnitude
  (`eta`, `lambda`, `alpha`).

The responsibility split, always:

- **Git** stores the experiment *definition* (the YAML under `config/experiments/`).
- **AMT** generates and evaluates the individual *trials*; never write trial values by hand into Git.
- **Pipeline Parameters** carry runtime values (dataset version, experiment name, HPO limits) from
  Git into the pipeline execution, but only values that genuinely vary between runs. Don't promote
  every YAML field to a Pipeline Parameter; that turns them into a second, untracked config system.

## Never auto-promote

An AMT winner (or a single-run result, for that matter) is a candidate, not a
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
`config/production/model.yaml` (`params`, scalars only) and retrain against *that*, on newly
approved data — don't rerun AMT on a schedule "just in case." HPO answers
"what configuration should we use?"; production retraining answers "train
the approved configuration on the latest approved data." Treat a request to
add HPO back into a scheduled retrain job as a deliberate exception, not the
default.

## CI/CD trigger shape

A GitHub Actions workflow that fires a training run should be passing
runtime values in, not baking decisions into the workflow file itself:

```
Git config change → GitHub Actions (--kind experiment|production)
                       ├─ all-scalar params  → one training job
                       └─ lists/ranges       → AMT tuning job → select_winner (validation only)
                                              → evaluate (test, once) → gate → register
```

If a workflow file starts containing hyperparameter values or search-space
bounds directly, move them back into the relevant `config/experiments/*.yaml`
and pass them through as parameters instead — see [[ml-repo-structure]] for
why that separation matters.

Before launching on SageMaker, commit your changes: the launcher refuses a dirty git tree
(`--allow-dirty` is for smoke tests only) because lineage needs a clean git sha.

## Where to look next

- Where a new experiment/production config file should physically live → [[ml-repo-structure]]
- What each run (single run or AMT trial) must log in MLflow, and immutable data rules → [[ml-lineage-reproducibility]]
