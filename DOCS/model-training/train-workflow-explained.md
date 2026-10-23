# `train.yml` explained (for people new to GitHub Actions)

File: [.github/workflows/train.yml](../../.github/workflows/train.yml). This guide explains what it does, how to run it, and what can go wrong. Companion to [model-training-implementation-steps.md](model-training-implementation-steps.md) (step 13).

---

## 1. What is a workflow?

A **workflow** is a YAML file in `.github/workflows/`. GitHub reads it and, when a **trigger** happens, starts a temporary Linux machine (a **runner**), runs your listed **steps** on it in order, then throws the machine away.

Vocabulary:

| Term | Meaning |
|---|---|
| **Trigger (`on:`)** | The event that starts the workflow: a button click, a schedule, a pull request. |
| **Job** | A group of steps that run on one runner. `train.yml` has one job, `train`. |
| **Step** | One command (`run:`) or one ready-made action (`uses:`). |
| **Runner** | The temporary machine (`ubuntu-latest`) GitHub gives the job. |
| **Repository variable** | A non-secret setting stored in GitHub (Settings, Secrets and variables, Actions, Variables). Read as `${{ vars.NAME }}`. |
| **OIDC** | Lets the runner prove "I am a workflow from this repo" to AWS and receive short-lived credentials, so no AWS keys are stored in GitHub. |

**Important:** the runner is only a remote control. It does *not* train the model. It tells AWS SageMaker to start training jobs, waits, and then scores the result.

---

## 2. The big picture

```
You click "Run workflow"  (or the weekly timer fires)
        |
        v
GitHub runner starts, checks out the code
        |
        v
Runner assumes the AWS role (OIDC)  ->  gets temporary AWS credentials
        |
        v
Runner loads DagsHub (MLflow) credentials from AWS Secrets Manager
        |
        v
Train:   run_training_job.py  ->  SageMaker training job or tuning job (AMT)
        |
        v
Select winner (hpo only):  best trial by VALIDATION score, tagged winner=true
        |
        v
Evaluate:  score winner ONCE on TEST, apply gate, register candidate if it passes
```

Nothing is promoted to production automatically. A pass only creates a *registered candidate* in the MLflow Model Registry; you approve it manually.

---

## 3. When does it run?

**Pushing or committing code does not start training.** Only these triggers do (`on:` block; the second is currently disabled):

### 3.1 `workflow_dispatch`: the manual button

In GitHub open **Actions**, pick **train**, click **Run workflow**. It asks for three inputs:

| Input | What to enter |
|---|---|
| `experiment` | Path of an experiment YAML, e.g. `config/experiments/experiment-001.yaml`. |
| `data_run_id` | The exact folder name under `processed/` in S3 (never `latest`). |
| `kind` | `experiment` or `production` (see section 4). |

### 3.2 `schedule`: the weekly timer (currently commented out)

**Status: disabled.** The `schedule:` lines in `train.yml` are commented out because we do not need automatic retraining yet. Nothing runs on a timer today.

What it is: a **cron timer**. `cron: "0 3 * * 1"` means "03:00 UTC every Monday". GitHub starts the workflow by itself and nobody types any inputs. The idea is a weekly retrain of the model you already approved, on fresh data.

**How it would work when enabled.** With no human to fill in the inputs, the workflow falls back to the defaults in the `env:` block at the top of the file (see 5.2):

- `KIND` becomes `production`, so it retrains `config/production/model.yaml` and never runs AMT.
- `EXPERIMENT` is ignored for `production`.
- `DATA_RUN_ID` comes from the repository variable `PRODUCTION_DATA_RUN_ID`.

**Why it would fail today** if you turned it on:

1. `PRODUCTION_DATA_RUN_ID` is not created, so `DATA_RUN_ID` is empty and the launcher rejects it. This is the "never use `latest`/empty" guard: training must always read one exact, immutable data folder so a re-run gives the same result.
2. `config/production/model.yaml` is still an empty template. It is only filled in after you approve a tuning winner.

**To enable it later:**

1. Approve a winner and copy its hyperparameters into `config/production/model.yaml`.
2. Create the repository variable `PRODUCTION_DATA_RUN_ID` (Settings, Secrets and variables, Actions, Variables) with the exact `processed/<run_id>` name to retrain on. Update it whenever new approved data should be used, otherwise every week retrains on the same data.
3. In `train.yml`, remove the leading `#` from these three lines:
   ```yaml
   schedule:
     - cron: "0 3 * * 1"
   ```
4. Merge to `main`. Scheduled runs only fire from the default branch, and a successful one launches paid SageMaker jobs.

### 3.3 Two gotchas

- The **Run workflow** button (and the schedule, if enabled) only work once the file exists on the repo's **default branch** (`main`). While it only lives on a feature branch, neither appears or fires.
- Scheduled runs happen without you watching, and a successful one launches paid SageMaker jobs.

---

## 4. The two `kind` values

| kind | Question it answers | What runs |
|---|---|---|
| `experiment` | The experiment file's `params` decide. | All scalars: one SageMaker training job, no AMT. Lists with `tuning.strategy: grid`: one AMT job that runs every combination once. Ranges with `strategy: bayesian` (or `random`): one AMT job with `max_jobs` trials, then the best on validation is picked. |
| `production` | "Retrain the model we already approved on new data." | One training job using `config/production/model.yaml`. The `experiment` input is ignored. Never runs AMT. |

Rule of thumb: start with a baseline (all-scalar params), use a grid to check a few chosen values, use a Bayesian search over wide ranges once you know which parameters matter, and use `production` only after a winner is approved.

---

## 5. Walk through the file

### 5.1 Header and permissions

```yaml
permissions:
  id-token: write
  contents: read
```

`id-token: write` lets the runner request the OIDC token needed to log in to AWS. `contents: read` lets it download the repo. It gets nothing else.

### 5.2 `env:` block

```yaml
env:
  KIND: ${{ inputs.kind || 'production' }}
  EXPERIMENT: ${{ inputs.experiment || 'config/experiments/experiment-001.yaml' }}
  DATA_RUN_ID: ${{ inputs.data_run_id || vars.PRODUCTION_DATA_RUN_ID }}
```

`a || b` means "use `a` if it exists, otherwise `b`". When you click the button, your inputs win. When the timer fires, there are no inputs, so the fallbacks apply. These become environment variables (`$KIND` and so on) for every step.

Design rule: **hyperparameters never appear in this file.** It only says *which* config and *which* data. The values live in `config/experiments/*.yaml`, so every run is reproducible from Git.

### 5.3 Steps

1. **`actions/checkout@v4`**: downloads the repo onto the runner.
2. **`astral-sh/setup-uv@v5`** and **`uv sync`**: installs `uv` and the exact locked Python dependencies.
3. **`aws-actions/configure-aws-credentials@v4`**: uses OIDC to assume the role in `vars.AWS_OIDC_ROLE_ARN` in region `vars.AWS_REGION`. This is why those two variables must be set. The role can only be assumed by this repository and is limited to SageMaker jobs, the data bucket and the DagsHub secret.
4. **Load MLflow credentials**: reads the DagsHub username and token from Secrets Manager (`fraud-detection/dev/dagshub-mlflow`) so the later steps can log to MLflow. `::add-mask::` makes GitHub hide the token if it ever appears in the logs.
5. **Train**: runs `python -m src.training.run_training_job --mode sagemaker ...`. This submits the SageMaker job (or the tuning job), waits for it to finish, and writes the MLflow run id to `run.json`. The step fails if the job does not end as `Completed`.
6. **Select winner** (only when `KIND == 'hpo'`): reads the parent run id from `run.json`, ranks the trials by `validation_aucpr` (validation data only), tags the best as `winner=true`, prints a comparison table into the job summary, and writes the winner's id back to `run.json`.
7. **Evaluate**: takes `run_id` from `run.json` and scores that model **once** on the test split, applies the gate from `config/evaluation/gate.yaml`, logs `test_*` metrics, and registers the model as a candidate if it passes. If the gate fails the step exits with an error, so the workflow turns red.

`run.json` is the small hand-off file between steps: the Train step writes it, Select winner and Evaluate read it.

---

## 6. Before your first run: checklist

- [ ] Terraform applied (log group, training policy, GitHub role). Done.
- [ ] Repository variables `AWS_OIDC_ROLE_ARN` and `AWS_REGION` set. Done.
- [ ] `train.yml` merged to `main` (the button will not appear before that).
- [ ] Processed data exists in S3: `processed/<run_id>/{train,val,test}` plus `metadata.json`. Local `synthetic-v0` does **not** count; it is only on your laptop.
- [ ] `config/evaluation/gate.yaml` thresholds reviewed (currently placeholders).
- [ ] Schedule stays commented out until `PRODUCTION_DATA_RUN_ID` and `config/production/model.yaml` are ready (see 3.2).

Recommended first run: try one job from your laptop first, because errors there are quicker to debug than inside Actions:

```bash
uv run python -m src.training.run_training_job --mode sagemaker --kind experiment \
  --experiment config/experiments/experiment-001.yaml --data-run-id <run_id>
```

---

## 7. Reading results

- **GitHub:** Actions, click the run, click the `train` job to see each step's log. For `hpo` the winner table appears under **Summary**.
- **AWS:** SageMaker console, Training jobs / Hyperparameter tuning jobs. Logs are in CloudWatch under `/aws/sagemaker/TrainingJobs` (14-day retention).
- **DagsHub MLflow:** runs with tags `mode`, `git_sha`, `config_hash`, `data_run_id`; the winner has `winner=true`; a passing model appears under the registered model `fraud-detector`.

---

## 8. Common failures

| Symptom | Likely cause |
|---|---|
| Run workflow button missing | File not on `main` yet. |
| "Could not assume role" / credentials error | `AWS_OIDC_ROLE_ARN` or `AWS_REGION` variable wrong, or the workflow ran from a repo other than `rahulvansh66/transaction-fraud-detection`. |
| "data run id must not be empty or latest" | `data_run_id` blank, `latest`, or `PRODUCTION_DATA_RUN_ID` unset on a scheduled run. |
| Train step fails on a dirty tree | Not an issue in Actions (fresh checkout); only when launching locally with uncommitted changes. |
| Training job fails on SageMaker | Read its CloudWatch log. Unverified so far: container Python version and the `meta` channel for `metadata.json`. |
| Evaluate exits with an error | The gate failed. This is expected behaviour, not a bug: the winner is not registered. Run a new experiment; do not pick a different trial. |
| "already scored on test" | The single-use test rule. Only override with `--allow-rescore` if you understand the trade-off. |
| Weekly run red on Monday | Only if you enabled the schedule without an approved production config or `PRODUCTION_DATA_RUN_ID` (see 3.2). |

---

## 9. Related files

- [.github/workflows/ci.yml](../../.github/workflows/ci.yml): runs lint, unit tests and `terraform validate` on every pull request. It never touches AWS.
- [infrastructure/github_oidc.tf](../../infrastructure/github_oidc.tf): the AWS role and its trust rules.
- [config/evaluation/gate.yaml](../../config/evaluation/gate.yaml): pass/fail thresholds.
- [config/experiments/](../../config/experiments/) and [config/production/model.yaml](../../config/production/model.yaml): the hyperparameters and search spaces.

---

> Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
