# Model training plan: XGBoost, AMT, MLflow and CI/CD

## 1. Context
Preprocessing writes time-split Parquet to `processed/<run_id>/{train,val,test}/` plus `metadata.json` (features, label, `scale_pos_weight`, thresholds). `src/training/` is an empty stub, and there is no `config/experiments/`, `config/production/`, `.github/` or training Terraform.

Goal: modular, config-driven training that runs locally first and then on SageMaker. Three AMT experiments are tracked in DagsHub MLflow, the winner is picked by comparing runs, and CI/CD covers training and retraining.

## 2. Decisions
- **Script mode on the SageMaker XGBoost framework container.** Our own `train.py` logs to MLflow live from inside the job.
- **Parquet end to end, no CSV step.** Parquet keeps types, is smaller and faster at 15GB+, and keeps column names tied to the feature contract. CSV would only be needed for the pure built-in algorithm mode.
- **MLflow on DagsHub.** Credentials come from `.env` locally and from Secrets Manager (`fraud-detection/<env>/dagshub-mlflow`) on SageMaker.
- **Registry: MLflow Model Registry** is the source of truth for promotion state. This is an open decision in the lineage skill, so it is flagged for confirmation.
- **Never auto-promote.** A winner is a candidate that goes through evaluation and a gate, then an approval.

## 3. Modularization
The rule is the same as for preprocessing: separate code by why it changes and where it runs.

| Layer | Files | Rule |
|---|---|---|
| Pure logic | `src/training/data.py`, `model.py`; `src/evaluation/metrics.py` | No I/O, no argument parsing, no S3, no MLflow. Takes arrays or DataFrames plus params and returns arrays, a booster or a metrics dict. Unit-testable on tiny synthetic data. |
| Orchestration | `src/training/train.py` | The only file that knows it runs as a SageMaker Training job (`SM_CHANNEL_*`, `SM_MODEL_DIR`). Parses config and hyperparameters, calls the pure functions in order, logs to MLflow, writes the model. |
| Launcher | `src/training/run_training_job.py`, `hpo_tuner.py` | Client-side. `run_training_job.py` is the CLI entry point (`--mode local\|sagemaker`, `--kind manual\|hpo\|production`). It loads the experiment config, builds the estimator and starts a single training job, or delegates to `hpo_tuner.py` when `--kind hpo`. `hpo_tuner.py` builds the AMT tuner and is used only for HPO runs. Neither contains ML logic. |
| Lineage | `src/training/lineage.py` | Dependency-free helpers, reusing `config_hash` and `build_run_id` from preprocessing. |
| Shared | `src/config_loader`, `src/mlflow_tracking` | `load_yaml`, `load_experiment`, `merge_configs`. MLflow helpers: `start_parent_run`, `start_child_run`, `set_lineage_tags`, `log_config_artifact`, `fetch_dagshub_secret`. |
| Evaluation | `src/evaluation/metrics.py`, `evaluate.py`, `select_winner.py` | Owns the definition of "good". `metrics.py` is the pure metric library. `select_winner.py` is a separate consumer of MLflow runs: it ranks trials on validation metrics and tags the winner. `evaluate.py` is a pipeline step that scores the winner on the test split, applies the gate and registers the candidate. |

Keep MLflow out of the pure modules: `metrics.py` only returns dicts, and `train.py` calls the thin `mlflow_tracking` helper. The model logic then runs unchanged in a local run, an AMT trial or production.

Dependency direction is one way: `training/` imports `evaluation/metrics.py`, and `evaluation/` never imports `training/`. This keeps a single metric definition for train and eval time, which avoids train/eval skew. To keep the training image light, `metrics.py` depends only on numpy and scikit-learn, and `evaluation/__init__.py` stays empty so importing `metrics` does not pull in `evaluate.py` or `select_winner.py`.

### Function-level outline
- `data.py`: `load_split(path, features, label)`, `read_metadata(path)`.
- `model.py`: `build_params(cfg, hp_overrides, scale_pos_weight)`, `fit(...)` with early stopping on validation AUPRC and a seed from config.
- `evaluation/metrics.py`: AUPRC, ROC-AUC, recall@precision, threshold selection on validation, and precision/recall/F1 at a given threshold. Shared by `train.py` and `evaluate.py`.
- `train.py`: computes train and val metrics only, and picks the threshold on validation. It never scores the test split. Logs the full lineage table (git SHA, config hash, data run_id, all params, seed, metrics, threshold, model with signature and input example, pip freeze, image URI with digest).
- `evaluation/select_winner.py`: ranks the AMT trials on validation metrics only and tags the best run as the winner. It does not gate and does not register.
- `evaluation/evaluate.py`: pipeline step run as a SageMaker Processing job, after `select_winner.py`. Loads the winning model and the test split, reads the validation-chosen threshold from the training run, computes test metrics with `metrics.py`, writes `evaluation.json` and logs to MLflow. Then applies the gate and, if it passes, registers the candidate. This is the only place the test set is scored. If the gate grows, split it into `gate.py`.

### Winner selection, test scoring and the gate
Two decisions are kept apart on purpose.

1. **Picking the winner** ranks trials against each other. It uses validation metrics only (for example `validation:aucpr`). Ranking by test score would tune the model to the test set, and the reported test score would then be optimistic.
2. **The gate** is a pass/fail check on the winner before it is registered, such as "test AUPRC and recall@precision above the minimums, and no worse than the current production model". It uses the test set, which no trial was tuned against, so it is an honest estimate of real performance.

Flow:
```
AMT trials -> select_winner.py (pick by validation)
           -> evaluate.py (score winner on test)
           -> gate (test metrics + comparison with production)
           -> register candidate -> manual approval
```

Production rules:
- Select and tune on validation. Score on test once, only for the chosen winner.
- Gate on the held-out score and also compare against the current production model on the same test data. Passing an absolute threshold is not enough if the model is worse than production.
- Never pick a different trial because the winner scored badly on test. That turns the test set into a second validation set. If the winner fails, run a new experiment. If this happens repeatedly, get a fresh holdout.
- Keep a human approval after the gate. Nothing is auto-promoted.
- `run_training_job.py`: entry point called by CI and by developers. Parses `--mode` and `--kind`, loads and merges configs, launches one training job for `manual` and `production` (production reads `config/production/model.yaml` and never reruns AMT), and calls `hpo_tuner.py` for `hpo`.
- `hpo_tuner.py`: used only by `--kind hpo`. Builds the `HyperparameterTuner` from the experiment YAML (objective `validation:aucpr`, Bayesian, `max_jobs` and `max_parallel_jobs`). Opens the MLflow parent run for the tuning job. Each trial is a child run tagged `mode=hpo`.

## 4. Configs
- `config/experiments/experiment-001.yaml`: tree structure (`max_depth`, `min_child_weight`, `gamma`).
- `config/experiments/experiment-002.yaml`: regularisation and imbalance (`reg_alpha`, `reg_lambda`, a `scale_pos_weight` scaling factor, `subsample`, `colsample_bytree`).
- `config/experiments/experiment-003.yaml`: boosting speed (`eta`, `num_round` with early stopping).
- Each file holds the search space, static params, the objective metric and the data `run_id` (an immutable path, never "latest").
- `config/production/model.yaml`: a template, filled only after the winner is approved.
- `config/env/{dev,prod}.yaml`: add a `training:` block (instance type, `max_run`, spot off). No model settings go here.
- The experiment skill recommends a few manual points before AMT. Since three AMT experiments were requested directly, each uses modest ranges, and this is noted in the files.

## 5. Terraform (no manual AWS changes)
- `cloudwatch.tf`: add `/aws/sagemaker/TrainingJobs` with explicit retention, and extend the execution role's log permissions.
- New `training.tf`: execution-role permissions for training and tuning (`sagemaker:*HyperParameterTuning*`, `*TrainingJob*`, scoped `iam:PassRole`) and S3 access for `models/*`.
- New `github_oidc.tf`: an OIDC provider and a role for GitHub Actions, scoped to this repo.
- Run `terraform validate` and `plan` for review. `apply` only with explicit approval.

## 6. CI/CD (`.github/workflows/`)
- `ci.yml` on PR: ruff, unit pytest, `terraform fmt -check` and `validate`.
- `train.yml`: `workflow_dispatch` with `experiment` and `data_run_id` inputs, plus a `schedule` for retraining. It assumes the OIDC role and runs `run_training_job.py --mode sagemaker`. `--kind hpo` for experiments. `--kind production` reads `config/production/model.yaml` and never reruns AMT.
- Winner selection runs `select_winner.py`, then `evaluate.py` scores the winner on the test split and applies the gate, and the comparison table is posted the comparison table to the job summary.
- Hyperparameters and search bounds live only in YAML, not in workflow files. Only the OIDC role ARN and region are configured in GitHub.

## 7. Dependencies
Add `xgboost` and `scikit-learn` to `pyproject.toml` (refresh `uv.lock`) and `ruff` to the dev group. Pin the container's XGBoost version in config so it matches local.

## 8. Tests
- `tests/unit/training/`: `build_params`, config merge and the tuner-config builder in `hpo_tuner.py` (boto stubbed).
- `tests/unit/evaluation/`: `metrics.py`, `evaluate.py` report building, and `select_winner` ranking on fake MLflow runs.
- Integration: train locally on a small synthetic split, and check that a local run and a SageMaker-style run give the same metrics for the same seed.

## 9. Verification order
1. `uv run pytest` passes.
2. Local: generate splits with the existing local Spark job on `dataset/raw/data-v0`, then run `run_training_job.py --mode local --kind manual` against `experiment-001.yaml`. Confirm a run in DagsHub with params, metrics, model and tags.
3. `terraform plan` reviewed, then applied with approval.
4. SageMaker: one training job, then the three AMT experiments. Confirm parent and child runs in DagsHub, comparable on shared metric names.
5. `select_winner.py` prints the comparison and picks the winner, `evaluate.py` scores it on the test split and applies the gate, and the winning candidate is registered. Promotion stays a manual approval.

## 10. Open items
- MLflow registry vs SageMaker Model Package Group (defaulting to MLflow).
- Prod bucket and role ARNs are placeholders, so CI targets `dev` only for now.
- Stale `tests/preprocessing/*` entries are in the git index; they are left alone.

---

> Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
