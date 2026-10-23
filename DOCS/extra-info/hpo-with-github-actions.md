# Hyperparameter tuning with GitHub Actions

A model's hyperparameters (tree depth, learning rate and so on) change how good it is, and nobody knows the best values up front. So you try many combinations and keep the best one. Done by hand on a laptop, that gets messy fast: you forget which settings produced which score, the data quietly changes between tries, and "the best model" can't be traced back to anything. Running the search from GitHub Actions fixes that. The settings live in git, the run is triggered from one button, and everything it did is recorded, so anyone can find out exactly how a model came to be.

## The idea in one picture

```
YAML file in git  ->  GitHub Actions  ->  SageMaker tuning job (many trials)
                                              |
                       MLflow records every trial
                                              |
                  pick winner on validation -> test it once -> gate -> registry
```

Nobody types hyperparameters into a workflow. The workflow only says *which file* and *which data*. That is the whole trick.

## 1. The search is just a config file

Each experiment is one YAML file under `config/experiments/`. Ours for the search is [experiment-003-search-space.yaml](../../config/experiments/experiment-003-search-space.yaml). Two blocks matter:

- `params`: every parameter in one place. A scalar (`eta: 0.1`) is fixed, a list (`max_depth: [2, 5, 7]`) is a set of discrete values, and a `{type, min, max}` dict is a range the tuner may explore, like `max_depth` from 5 to 11.
- `tuning`: how to search. `strategy` is always written explicitly: `grid` (every list value once, lists only), `random` or `bayesian` (needs `max_jobs`).

Why a file and not a few lines in the workflow? Because git remembers. Six months later you can still open the exact ranges that were searched. It also keeps the workflow boring, which is what you want.

Before this we run a baseline with all-scalar params ([experiment-002-manual-params.yaml](../../config/experiments/experiment-002-manual-params.yaml), a single training job), and optionally a grid sensitivity study such as [experiment-004-depth-sensitivity.yaml](../../config/experiments/experiment-004-depth-sensitivity.yaml) (`max_depth: [2, 5, 7]`). A single point does not tell you where to search, so prefer wide ranges around sensible defaults. Each file gets its own number, and we never edit one after it has produced runs.

## 2. The button

[train.yml](../../.github/workflows/train.yml) is started manually (Actions -> train -> Run workflow). You give it three things: the experiment file, the data run id, and the kind (`experiment` or `production`). It logs into AWS through OIDC, so there are no stored AWS keys, then calls the launcher:

```
python -m src.training.run_training_job --mode sagemaker --kind experiment ...
```

The data run id is a specific, frozen folder of processed data, never `latest`. If the data could change under you, two runs of the "same" experiment wouldn't be comparable. The check lives in `validate_data_run_id` ([sagemaker_jobs.py:74](../../src/training/sagemaker_jobs.py#L74)).

## 3. The launcher checks before spending money

Tuning jobs cost real money, so we fail early. Two checks run before anything is submitted:

- **Is the file valid?** [validate_experiment](../../src/training/experiment_config.py) rejects unknown top-level keys (a mis-indented parameter), lists or ranges without an explicit `tuning.strategy`, `grid` over a range, a missing `max_jobs` for random/bayesian, and a grid `max_jobs` that differs from the combination count. Without this, a typo would quietly train with default values.
- **Is the git tree clean?** On SageMaker the launcher refuses a dirty tree ([run_training_job.py:227](../../src/training/run_training_job.py#L227)), because otherwise the recorded commit wouldn't match the code that actually ran.

## 4. Handing the search to SageMaker

SageMaker has a built-in tuner called AMT (Automatic Model Tuning). It picks values, runs a training job per trial, and learns from the scores to choose better values next time. Our code only translates the YAML into the request AMT wants:

- [build_ranges](../../src/training/hpo_tuner.py) turns each list into a categorical range (values as strings) and each range entry into an integer or continuous range.
- [build_tuning_request](../../src/training/hpo_tuner.py) adds the strategy, the budget (`max_jobs`, `max_parallel_jobs`; grid omits `max_jobs` because AMT derives it), the metric to maximise, and the fixed values.

The launcher submits it in the search branch of [run_sagemaker](../../src/training/run_training_job.py) and waits for it to finish. An all-scalar file skips AMT and submits one plain training job.

A note on budget: `max_jobs: 3` in our file is only to check the plumbing works. A real search needs far more, roughly 10 to 20 trials for every parameter you tune.

## 5. Every trial is recorded

The launcher opens one MLflow "parent" run for the whole search. Each trial's `train.py` opens a "child" run under it (see [train.py:250](../../src/training/train.py#L250)). Every run is tagged with the git commit, the config hash, the data run id and the experiment name, so a model can always be traced back to what made it.

## 6. Picking the winner, carefully

This is the part newcomers get wrong, so it's worth slowing down.

1. [select_winner.py](../../src/evaluation/select_winner.py) ranks trials using **validation** scores only.
2. [evaluate.py](../../src/evaluation/evaluate.py) then scores that one winner on the **test** set, once.
3. A gate in [gate.yaml](../../config/evaluation/gate.yaml) (minimum AUCPR, minimum recall, not worse than production) decides if it gets registered.

Why not just pick the best test score? Because if you compare 30 trials on the test set, the best one is partly just lucky. The test number then looks better than the model really is. Keeping the test set for a single final look keeps it honest. And if the winner fails the gate, it fails. You start a new experiment, you don't go hunting for another trial that passes.

Passing the gate only registers a *candidate*. Promoting it to production is a separate, manual step.

## Habits worth keeping

- One file per experiment, never edited after it has run.
- Wide ranges around sensible defaults; a single run gives no direction to narrow them.
- Same data, seed and metric across a round's files, so the scores can be compared.
- Commit before any SageMaker run so the recorded git sha is clean.
- Never auto-promote a winner.
- Don't re-run the search on a schedule. Once you have a good config, retrain that approved config on new data instead.

---

> Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
