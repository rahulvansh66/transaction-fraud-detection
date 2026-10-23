# Model training: step-by-step implementation guide

Companion to [model-training-plan.md](model-training-plan.md). The plan says *what* we build and *why*; this guide says *in what order* and *how each piece works*. Snippets are minimal sketches to show the idea, not final code. Real code must follow the repo rules in `CLAUDE.md` (Google docstrings on everything, type hints, `logging` with key=value fields, no secrets or PII in logs, Terraform-only infrastructure).

Each step has: **Goal**, **Concept**, **Files**, **Snippet**, **Done when**.

---

## Step 0. Primer: the vocabulary

| Term | Plain-English meaning |
|---|---|
| **Training job** | SageMaker starts a container on a fresh instance, runs your script, saves the model, then shuts the instance down. You pay per second. |
| **Script mode** | You give SageMaker a Python file (`train.py`) and it runs it inside AWS's ready-made XGBoost container. No Dockerfile needed. |
| **Channels** | Named input folders. If you pass `{"train": s3://.../train/, "val": s3://.../val/}`, SageMaker downloads them and sets `SM_CHANNEL_TRAIN` and `SM_CHANNEL_VAL` to the local paths. |
| **`SM_MODEL_DIR`** | The folder where `train.py` must write the model. SageMaker uploads it to S3 as `model.tar.gz`. |
| **Hyperparameters** | Key/value pairs SageMaker passes to `train.py` as command-line args. |
| **AMT (Automatic Model Tuning)** | SageMaker runs many training jobs ("trials") with different hyperparameters chosen by Bayesian search, and reports which trial scored best on an objective metric. |
| **MLflow** | An experiment tracker. Each training run logs params, metrics, tags and the model. We host it on DagsHub. |
| **Parent / child run** | For AMT: one parent run represents the whole search, and each trial is a child run nested under it. |
| **Model registry** | A list of model versions with a promotion state (`None`, `Staging`, `Production`). |
| **CI/CD (GitHub Actions)** | A workflow file that runs automatically (on a PR, on a button click, on a schedule) and executes commands, here: tests, and starting SageMaker jobs. |
| **OIDC** | Lets GitHub Actions assume an AWS role without storing long-lived AWS keys in GitHub. |

### The whole flow

```
preprocessing output (immutable processed/<run_id>/{train,val,test})
        |
        v
train.py  (single run  OR  AMT trials)      <- logs to MLflow, uses train + val only
        |
        v
select_winner.py   ranks trials on VALIDATION metrics, tags the winner
        |
        v
evaluate.py        scores the winner ONCE on TEST, applies the gate
        |
   pass |  fail -> reject (run a new experiment, never pick another trial)
        v
register candidate in MLflow Model Registry -> manual approval -> production
```

### How a run is chosen (from the experimentation-workflow skill)

The `params` block of an experiment file decides, and `--kind` is only `experiment` or `production`:

- all scalars: one training job (`mode=single`), no AMT.
- lists with `tuning.strategy: grid`: AMT runs every combination once (`mode=grid`).
- ranges (and/or lists) with `strategy: bayesian` or `random`: AMT search limited by `max_jobs`.
- `production`: retrain the *already approved* config on new data. Never reruns AMT.

Every MLflow run is tagged `mode=single|grid|random|bayesian|production`.

---

## Step 1. Dependencies

**Goal:** Local and container environments use the same library versions.

**Concept:** SageMaker's XGBoost container ships one specific XGBoost version. If your laptop uses a different one, "same seed, same metrics" can fail. So we pin one version and use it in both places.

**Files:** `pyproject.toml`, `uv.lock`

```toml
[project]
dependencies = [
  "xgboost==1.7.4",        # must equal the container framework_version
  "scikit-learn",
  "mlflow",
]
[dependency-groups]
dev = ["ruff", "pytest"]
```

Run `uv sync` to refresh the lock file.

**Done when:** `uv run python -c "import xgboost, sklearn, mlflow"` works.

---

## Step 2. Configs (Git holds the definitions)

**Goal:** All hyperparameters and search ranges live in YAML, never in code or workflow files.

**Concept:** Git stores the *search-space definition*. AMT generates the *trial values*. You never write trial values by hand. The data location is an exact `run_id` path, never "latest", so a re-run reads the same bytes.

**Files:**
- `config/experiments/experiment-001.yaml`: tree structure (`max_depth`, `min_child_weight`, `gamma`)
- `config/experiments/experiment-002.yaml`: regularisation and imbalance (`reg_alpha`, `reg_lambda`, `scale_pos_weight` factor, `subsample`, `colsample_bytree`)
- `config/experiments/experiment-003.yaml`: boosting speed (`eta`, `num_round`)
- `config/production/model.yaml`: template, filled only after a winner is approved
- `config/env/dev.yaml`, `prod.yaml`: add a `training:` block (instance type, `max_run`, spot off). No model settings here.

```yaml
# config/experiments/experiment-001.yaml
experiment:
  name: exp-001-tree-structure
data:
  run_id: 2026-09-01T10-00-00_ab12cd     # immutable, explicit
params:
  objective: binary:logistic             # scalar = fixed
  eval_metric: aucpr
  eta: 0.1
  seed: 42
  max_depth:        {type: integer,     min: 3,   max: 10}   # range
  min_child_weight: {type: continuous,  min: 1,   max: 10}
  gamma:            {type: continuous,  min: 0,   max: 5}
  # a list such as max_depth: [2, 5, 7] means discrete values (use strategy: grid)
tuning:
  strategy: bayesian     # always explicit: grid (lists only) | random | bayesian
  objective_metric: validation:aucpr
  objective_type: Maximize
  max_jobs: 3            # small on purpose while we validate 001 end to end
  max_parallel_jobs: 2
```

**Current scope:** For now we run only `experiment-001.yaml`. Experiments 002 (all-scalar baseline), 003 (Bayesian ranges) and 004 (grid over `max_depth: [2, 5, 7]`) now exist in the new schema. Raise `max_jobs` once the 001 plumbing is confirmed.

**Note on ranges:** one baseline point gives no direction, so prefer ranges around sensible defaults over ranges narrowed from a single run, and say so in a comment at the top of each file. Once a winner is approved, copy its values into `config/production/model.yaml`. 

**Done when:** The three files load with `load_yaml` and contain no environment values (bucket names, ARNs, tracking URIs).

---

## Step 3. Shared helpers

**Goal:** Small, dependency-light building blocks that every later step uses.

**Concept:** Keep MLflow and config plumbing out of the ML code, so the same model code runs locally, in an AMT trial and in production.

**Files:** `src/config_loader`, `src/mlflow_tracking`, `src/training/lineage.py`

```python
# src/config_loader.py
def merge_configs(base: dict, override: dict) -> dict:
    """Recursively merge override into base and return a new dict."""
    out = dict(base)
    for k, v in override.items():
        out[k] = merge_configs(out[k], v) if isinstance(v, dict) and k in out else v
    return out
```

```python
# src/mlflow_tracking.py  (thin wrappers)
def set_lineage_tags(git_sha: str, config_hash: str, data_run_id: str, mode: str) -> None:
    """Tag the active run with the identifiers needed to reproduce it."""
    mlflow.set_tags({"git_sha": git_sha, "config_hash": config_hash,
                     "data_run_id": data_run_id, "mode": mode})
```

- `fetch_dagshub_secret(secret_id)` reads `fraud-detection/<env>/dagshub-mlflow` from Secrets Manager on SageMaker. Locally it reads `.env`. Never log the value.
- `lineage.py` reuses `config_hash` and `build_run_id` from preprocessing.

**Done when:** Unit tests pass for `merge_configs` and `config_hash` (same input gives the same hash).

---

## Step 4. Pure logic: metrics, data, model

**Goal:** The ML logic, with no I/O, no S3 and no MLflow.

**Concept:** Pure functions are easy to test on tiny synthetic data. `metrics.py` is shared by training and evaluation, so "good" has one definition (avoids train/eval skew). It imports only numpy and scikit-learn to keep the training image light.

**Files:** `src/evaluation/metrics.py`, `src/training/data.py`, `src/training/model.py`

```python
# metrics.py
def compute_metrics(y_true, y_score, threshold: float) -> dict[str, float]:
    """AUPRC, ROC-AUC, and precision/recall/F1 at the given threshold."""
    y_pred = y_score >= threshold
    return {
        "aucpr": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

def select_threshold(y_true, y_score, min_precision: float) -> float:
    """Pick the threshold on VALIDATION data that keeps precision >= min_precision with best recall."""
```

```python
# model.py
def build_params(cfg: dict, hp_overrides: dict, scale_pos_weight: float) -> dict:
    """Merge static params, AMT-provided hyperparameters and class weight."""

def fit(dtrain, dval, params: dict, num_round: int, seed: int) -> xgb.Booster:
    """Train with early stopping on validation AUPRC; seed is set explicitly."""
    return xgb.train({**params, "seed": seed}, dtrain, num_round,
                     evals=[(dval, "validation")], early_stopping_rounds=20)
```

`data.py`: `load_split(path, features, label)` reads Parquet and returns X and y. `read_metadata(path)` reads `metadata.json` (features, label, `scale_pos_weight`, thresholds).

**Done when:** `tests/unit` pass on a 200-row synthetic dataset.

---

## Step 5. `train.py`, the SageMaker entry script

**Goal:** One script that runs identically locally and as a SageMaker job, and logs the full lineage.

**Concept:** This is the *only* file that knows about SageMaker (`SM_CHANNEL_*`, `SM_MODEL_DIR`). It parses args, calls the pure functions, logs to MLflow, and writes the model. It never touches the test split.

```python
def main() -> None:
    logging.basicConfig(level=logging.INFO)          # only entry points configure logging
    args = parse_args()                              # hyperparameters arrive as CLI args
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", args.train_dir)
    val_dir   = os.environ.get("SM_CHANNEL_VAL",   args.val_dir)
    model_dir = os.environ.get("SM_MODEL_DIR",     args.model_dir)

    meta = read_metadata(train_dir)
    params = build_params(cfg, hp_overrides, meta["scale_pos_weight"])
    booster = fit(dtrain, dval, params, args.num_round, args.seed)

    thr = select_threshold(y_val, booster.predict(dval), cfg["min_precision"])
    with mlflow.start_run(run_name=job_name, nested=True):   # child run under the AMT parent
        set_lineage_tags(git_sha, config_hash, data_run_id, mode)
        mlflow.log_params(params)
        mlflow.log_metrics(compute_metrics(y_val, booster.predict(dval), thr))
        mlflow.xgboost.log_model(booster, "model", signature=sig, input_example=ex)
    booster.save_model(os.path.join(model_dir, "xgboost-model"))
```

Also print the objective metric in the form AMT can parse from logs, for example `[5]#011validation-aucpr:0.8123` (XGBoost's own eval output already does this).

### Lineage checklist (from ml-lineage-reproducibility)

| Lineage input | How `train.py` records it |
|---|---|
| Code version | `git_sha` tag (passed in as an env var or hyperparameter by the launcher) |
| Config version | `config_hash` tag plus the resolved config as an artifact |
| Data version | `data_run_id` tag and `mlflow.log_input` on the exact S3 path |
| Pipeline version | `PipelineExecutionArn` tag when run from a pipeline (empty for manual runs) |
| Environment | image URI **with digest** and `pip freeze` artifact |
| Hyperparameters | the full set via `log_params`, not a subset |
| Seeds | `seed` logged |
| Metrics | train and val, threshold-free and at the chosen threshold. **Test metrics come from `evaluate.py`.** |
| Artifacts | model, feature list, signature, input example |

Log at the start of the run: job name, git SHA, data run_id, e.g. `logger.info("step=train status=start job=%s git_sha=%s data_run_id=%s", ...)`.

**Done when:** Running `train.py` directly on local Parquet produces a model file and a run in MLflow with every row of the table filled.

---

## Step 6. Unit tests

**Goal:** Catch logic bugs before paying for AWS.

**Files:** `tests/unit/training/`, `tests/unit/evaluation/`

- `build_params`: overrides win over static params, `scale_pos_weight` applied.
- `merge_configs`, and the tuner-config builder (with boto stubbed).
- `metrics.py`: known inputs give known AUPRC and threshold.
- `select_winner`: ranks fake MLflow runs, picks the max on `validation:aucpr`.
- Determinism: the same seed gives identical metrics on a tiny synthetic split.

**Done when:** `uv run pytest` passes and `uv run ruff check .` is clean.

---

## Step 7. Run locally first

**Goal:** Prove the end-to-end path before touching SageMaker.

**Concept:** `run_training_job.py` is the client-side launcher. `--mode local` runs `train.py` as a normal subprocess. It contains no ML logic.

```bash
# 1. make small splits with the existing local Spark job
# 2. train
uv run python -m src.training.run_training_job \
  --mode local --kind experiment --experiment config/experiments/experiment-002-manual-params.yaml
```

Check in DagsHub: the run exists, `mode=single`, git SHA and config hash tags are set, params and metrics are present, the model has a signature.

**Done when:** A run appears in DagsHub with all lineage fields, and re-running with the same seed reproduces the metrics.

---

## Step 8. Terraform (no console clicks)

**Goal:** All AWS pieces exist through code.

**Concept:** Terraform is the source of truth. You write `.tf`, run `plan` to see what would change, and `apply` only after review and approval.

**Files under `infrastructure/`:**
- `cloudwatch.tf`: log group `/aws/sagemaker/TrainingJobs` with explicit `retention_in_days` (short for dev). Extend the execution role's log permissions.
- `training.tf`: execution-role permissions for `sagemaker:CreateTrainingJob`, `*HyperParameterTuningJob*`, scoped `iam:PassRole`, and S3 access to `models/*`.
- `github_oidc.tf`: OIDC provider plus a role that only this repo can assume.

```hcl
resource "aws_cloudwatch_log_group" "training" {
  name              = "/aws/sagemaker/TrainingJobs"
  retention_in_days = 14
}
```

```bash
terraform fmt -check && terraform validate && terraform plan
```

**Done when:** `plan` looks right and, after your approval, `apply` succeeds.

---

## Step 9. One SageMaker training job

**Goal:** Run `train.py` on SageMaker once, before any tuning.

**Concept:** The `Estimator` says which container, instance and script to use. `fit()` starts the job with the data channels. Hyperparameters become CLI args.

```python
from sagemaker.xgboost.estimator import XGBoost

est = XGBoost(
    entry_point="train.py", source_dir="src/training",
    framework_version="1.7-1",               # matches the pinned xgboost
    role=env["execution_role_arn"], instance_type=env["training"]["instance_type"],
    instance_count=1, max_run=env["training"]["max_run"], use_spot_instances=False,
    hyperparameters={**static_params, "git_sha": git_sha, "data_run_id": run_id},
    environment={"DAGSHUB_SECRET_ID": f"fraud-detection/{env_name}/dagshub-mlflow"},
)
est.fit({"train": f"s3://{bucket}/processed/{run_id}/train/",
         "val":   f"s3://{bucket}/processed/{run_id}/val/"})
```

Look at the logs in CloudWatch (`/aws/sagemaker/TrainingJobs`) and the run in DagsHub. Record the image URI with digest as a tag.

**Done when:** The SageMaker run's metrics match the local run for the same seed and data.

---

## Step 10. AMT (hyperparameter tuning)

**Goal:** Run the three experiments as tuning jobs, tracked as parent and child runs.

**Concept:** A `HyperparameterTuner` wraps the estimator. You give it ranges and an objective metric. It launches trials (2 at a time here), learns from the finished ones and picks the next values. AMT finds the objective by regex over the container's log lines.

**Files:** `src/training/hpo_tuner.py` (used only when `params` has lists or ranges)

```python
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter

tuner = HyperparameterTuner(
    estimator=est,
    objective_metric_name="validation:aucpr",
    objective_type="Maximize",
    metric_definitions=[{"Name": "validation:aucpr",
                         "Regex": r"validation-aucpr:([0-9\.]+)"}],
    hyperparameter_ranges=build_ranges(plan.lists, plan.ranges),  # from the YAML params
    strategy=cfg["tuning"]["strategy"],      # grid | random | bayesian, explicit
    max_jobs=cfg["tuning"]["max_jobs"],      # omitted for grid: AMT derives it
    max_parallel_jobs=cfg["tuning"]["max_parallel_jobs"],
)
tuner.fit({"train": train_uri, "val": val_uri}, wait=False)
```

MLflow structure:
- The launcher opens the **parent run** (tag `mode` = resolved strategy, tuning job name).
- Each trial's `train.py` opens a **child run** carrying that trial's generated hyperparameters. The parent run id reaches the trial as an environment variable.

Run experiments 001, 002, 003 in turn. Start with a small `max_jobs` (for example 4) to confirm plumbing, then the real value.

**Done when:** DagsHub shows three parent runs with child runs, all with the same metric names so they can be compared.

---

## Step 11. `select_winner.py`

**Goal:** Choose the best trial without touching test data.

**Concept:** Ranking uses validation metrics only. If you ranked on test, you would tune to the test set and its score would be optimistic.

```python
def select_winner(parent_run_id: str, metric: str = "validation_aucpr") -> str:
    """Return the run_id of the best child run and tag it winner=true."""
    runs = mlflow.search_runs(filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
                              order_by=[f"metrics.{metric} DESC"], max_results=1)
    best = runs.iloc[0].run_id
    MlflowClient().set_tag(best, "winner", "true")
    return best
```

It also prints a comparison table (top trials and their params). It does not gate and does not register.

**Done when:** A run is tagged `winner=true` and the table is printed.

---

## Step 12. `evaluate.py`: test scoring, gate, registration

**Goal:** Get an honest estimate for the winner, and decide if it may become a candidate.

**Concept:** This is the only place the test split is scored, once, for the chosen winner. It reuses the validation-chosen threshold. The gate has two parts: absolute minimums (for example test AUPRC and recall@precision) and "no worse than the current production model on the same test data". A pass leads to registration, never to auto-promotion.

```python
report = compute_metrics(y_test, model.predict(dtest), threshold_from_training_run)
passed = (report["aucpr"] >= gate["min_aucpr"]
          and report["aucpr"] >= prod_report["aucpr"])
write_json("evaluation.json", {"metrics": report, "passed": passed})
mlflow.log_metrics({f"test_{k}": v for k, v in report.items()})
if passed:
    mlflow.register_model(f"runs:/{winner_run_id}/model", "fraud-detector")  # stage: None
```

Rules to remember:
- If the winner fails, do **not** pick another trial. That turns the test set into a second validation set. Run a new experiment; if it keeps happening, get a fresh holdout.
- Promotion (`Staging` then `Production`) stays a manual approval.

**Open decision (flagged in the lineage skill):** the plan defaults to the MLflow Model Registry. The alternative is a SageMaker Model Package Group. Confirm before implementing registration.

Run it as a SageMaker Processing job when it becomes a pipeline step.

**Done when:** `evaluation.json` exists, test metrics are in MLflow, and a passing model appears as a registered version awaiting approval.

---

## Step 13. CI/CD with GitHub Actions

**Goal:** Tests on every PR, and training started from Git rather than from a laptop.

**Concept:** A workflow is a YAML file in `.github/workflows/`. GitHub gives it a short-lived token, which AWS trusts through the OIDC role from step 8, so there are no stored AWS keys. Hyperparameters stay in the experiment YAML. GitHub only holds the role ARN and region.

**`ci.yml`** (on pull request):

```yaml
on: pull_request
jobs:
  checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run pytest tests/unit
      - run: terraform -chdir=infrastructure fmt -check && terraform -chdir=infrastructure validate
```

**`train.yml`** (manual button plus schedule):

```yaml
on:
  workflow_dispatch:
    inputs:
      experiment: {description: "config/experiments file", required: true}
      data_run_id: {description: "immutable processed/<run_id>", required: true}
      kind: {type: choice, options: [experiment, production], default: experiment}
  schedule:
    - cron: "0 3 * * 1"          # weekly retrain, production kind only
permissions: {id-token: write, contents: read}
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.AWS_OIDC_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION }}
      - run: uv sync
      - run: |
          uv run python -m src.training.run_training_job --mode sagemaker \
            --kind ${{ inputs.kind || 'production' }} \
            --experiment ${{ inputs.experiment }} --data-run-id ${{ inputs.data_run_id }}
      - run: uv run python -m src.evaluation.select_winner >> $GITHUB_STEP_SUMMARY
      - run: uv run python -m src.evaluation.evaluate
```

Notes:
- `--kind production` reads `config/production/model.yaml` and never reruns AMT.
- The scheduled run trains the approved config on new data. It does not search.
- CI targets `dev` only until prod bucket and role ARNs exist.
- The `select_winner` step only applies to searches (`run.json` has `parent_run_id`); it is skipped for single runs and `production`.

**Done when:** A PR runs `ci.yml` green, and a manual `train.yml` dispatch starts a SageMaker job and posts the comparison table to the job summary.

---

## Step 14. Verification and wrap-up

### Order of checks
1. `uv run pytest` and `ruff` pass.
2. Local single run appears in DagsHub with full lineage.
3. `terraform plan` reviewed, `apply` after approval.
4. One SageMaker training job matches the local metrics.
5. Three AMT experiments show parent and child runs.
6. `select_winner.py`, then `evaluate.py`, then gate, then registered candidate. Promotion is manual.

### Pitfalls
- **Different XGBoost versions** locally and in the container.
- **Reading "latest"** instead of an explicit `run_id`.
- **Scoring test during training or ranking.**
- **Hyperparameters typed into workflow files** instead of YAML.
- **Logging secrets** (DagsHub token) or PII to CloudWatch.
- **Log group with no retention.** Set it in Terraform.
- **Forgetting `max_run`** on training jobs, which can leave a job running and costing money.
- **Per-row logging** in loops. Log aggregates.

### Open items
- Registry decided: MLflow Model Registry (`config/evaluation/gate.yaml`, model `fraud-detector`, promotion via the `production` alias, manual).
- Implementation notes: SageMaker SDK v3 has no `XGBoost` estimator, so `sagemaker_jobs.py`/`hpo_tuner.py` call boto3 (`create_training_job`, `create_hyper_parameter_tuning_job`). Gate thresholds are placeholders until real data exists.
- Unverified on AWS: container Python version vs. the code's 3.10+ syntax, `metadata.json` as an S3Prefix channel, and image digest capture (only the tag URI is recorded).
- Prod bucket and role ARNs are placeholders.
- Stale `tests/preprocessing/*` entries in the git index are left alone.

---

> Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
