# Understanding this pipeline

You know how to train a model locally. This doc maps everything we have built on top of that into one picture, tells you where to look to see it running, and — because the point is to grow as an ML engineer, not just get a model out — explains *why* each piece exists, i.e. what production-grade practice it demonstrates. It is the entry point into [DOCS/](.); the other files go deeper on one stage each and are linked from the matching section below.

---

## 1. Orientation

```
S3 raw/<data_version>/                         (immutable source data)
        |
        v
PySpark Processing job  ->  S3 processed/<run_id>/{train,val,test} + metadata.json
        |                                       (immutable snapshot, never overwritten)
        v
Training  (local, or SageMaker: manual | hpo)  ->  one MLflow run per attempt
        |
        v
[hpo only] select_winner.py  ->  best trial by VALIDATION score, tagged winner=true
        |
        v
evaluate.py  ->  score ONCE on TEST, apply gate (config/evaluation/gate.yaml)
        |
        v
Pass  ->  register a CANDIDATE in the MLflow Model Registry  (not production yet)
        |
        v
A human looks at it and promotes it to the "production" alias.  Nothing does this automatically.
```

Five words you'll see everywhere in this doc:

| Term | Meaning |
|---|---|
| **Job** | One unit of work AWS runs for you on a temporary machine it manages — a *Processing job* (runs `spark_job.py`) or a *Training job* (runs `train.py`). You submit it, AWS runs it, the machine disappears when it's done. |
| **Run** | One entry in MLflow — one attempt at training, with its params, metrics and artifacts attached. |
| **Lineage** | Being able to answer "what exact code, config and data produced this model?" for any run, after the fact. |
| **Immutability** | Once data is written to a versioned path (`processed/<run_id>/...`), it is never edited or overwritten. Need different data? You get a new `run_id`, not an edit. |
| **Gate** | A pass/fail check a model must clear before it's even allowed to become a registry candidate. |

---

## 2. Raw data — S3

- **Where:** S3 console → bucket `fraud-detection-dev-data-use1` → `raw/<data_version>/`, Hive-partitioned parquet.
- **What to check:** nothing changes here day to day. That's the point — raw data is read-only input, never reprocessed in place.
- **Practice point:** every later stage reads from an explicit `<data_version>`, never a "latest" pointer. If raw data changes, it gets a new version folder, so every downstream run can still say exactly which raw snapshot it came from.
- Background on the dataset: [dataset-info.md](dataset-info.md).

---

## 3. Preprocessing — SageMaker PySpark Processing job

- **Code:** [src/preprocessing/spark_job.py](../src/preprocessing/spark_job.py) (the job itself), [features.py](../src/preprocessing/features.py) (pure transform functions — select, clean, feature engineering, leakage gate, time-based split), [lineage.py](../src/preprocessing/lineage.py) (`config_hash`, `build_run_id`), [run_preprocessing_job.py](../src/preprocessing/run_preprocessing_job.py) (the launcher you run from your machine or CI).
- **Config:** [config/preprocessing/preprocessing.yaml](../config/preprocessing/preprocessing.yaml) — label column, kept transaction types, split fractions, night-hour definition, Spark job sizing. No logic decisions live outside this file.
- **Where to look:**
  - SageMaker console → **Processing jobs**.
  - CloudWatch → log group `/aws/sagemaker/ProcessingJobs`.
  - S3 → `processed/<run_id>/{train,val,test}` parquet + `metadata.json` (git commit, feature list, label, split thresholds, `scale_pos_weight`, row/fraud counts, full resolved config, `config_hash`).
- **Practice points worth understanding:**
  - The job writes with `mode("errorifexists")` and the launcher uploads its launch marker with S3's `IfNoneMatch: "*"` — re-running the same `run_id` is physically rejected, not just discouraged. That's how immutability is enforced, not just promised.
  - `apply_leakage_gate` keeps the genuinely leaky columns (`errorBalanceOrig`/`errorBalanceDest`) out unless explicitly turned on, and `validate_outputs` re-reads the written splits afterward to assert no leaky columns and no overlapping time ranges between train/val/test.
  - The split is **time-based** (by `step`), not random — fraud detection data is sequential, so random splitting would leak future information into training.
  - `run_id = <git_sha>-<config_hash>-<data_version>` — the folder name itself encodes exactly which code and config produced it. A dirty working tree gets a `-dirty-<hash>` suffix so you can never mistake an uncommitted experiment for a reproducible one.
  - Deep dive: [preprocessing/modularization.md](preprocessing/modularization.md), [preprocessing/preprocessing-pyspark-processing-plan.md](preprocessing/preprocessing-pyspark-processing-plan.md), [preprocessing/preprocessing-pyspark-implementation-steps.md](preprocessing/preprocessing-pyspark-implementation-steps.md), [feature-engineering.md](feature-engineering.md).

---

## 4. Training — local, SageMaker manual, or SageMaker HPO

- **Code:** [src/training/train.py](../src/training/train.py) (the job entry point — the only module that reads SageMaker's `SM_CHANNEL_*` env vars; never touches the test split), [data.py](../src/training/data.py), [model.py](../src/training/model.py) (pure XGBoost param building/fitting, no I/O), [sagemaker_jobs.py](../src/training/sagemaker_jobs.py) (builds the boto3 `CreateTrainingJob` request), [hpo_tuner.py](../src/training/hpo_tuner.py) (builds the AMT tuning request), [run_training_job.py](../src/training/run_training_job.py) (the CLI launcher: `--mode {local,sagemaker} --kind {manual,hpo,production}`).
- **Config:** [config/experiments/experiment-001.yaml](../config/experiments/experiment-001.yaml) — `data.run_id`, `runtime.xgboost_version` (must match the SageMaker container), `static_params`, `manual_params`, `search_space`, `tuning`.
- **Where to look:**
  - SageMaker console → **Training jobs** (for `manual`/`production`) or **Hyperparameter tuning jobs** (for `hpo`).
  - CloudWatch → log group `/aws/sagemaker/TrainingJobs`, one stream per job — this is where you read the actual Python traceback if a job fails.
  - Example from this week: job `fraud-manual-e55a206e-0927094015` failed twice first (XGBoost 1.7.6 vs the container's actual 1.7.4; then a `3.9`-incompatible type hint) before completing and matching the local run exactly.
- **Practice points worth understanding:**
  - **No hyperparameter is ever hardcoded** in `src/` — it all comes from the experiment YAML. The CLI/CI layer only ever passes *which config file* and *which immutable data run*, never values.
  - `check_xgboost_version` in `train.py` asserts the installed XGBoost matches `runtime.xgboost_version` and fails fast rather than silently training with a different version than you tested locally — this is exactly what caught the real container-version mismatch this week.
  - `validate_data_run_id` rejects `""`/`latest`/`current` outright — you cannot accidentally train against a moving target.
  - `build_source_tar` builds the code bundle **deterministically** (fixed timestamps) and uploads it content-addressed (hash in the S3 key) — identical code always produces the identical bundle, and re-uploading is a no-op.
  - In `hpo` mode, the search space (ranges) lives in Git; AMT generates the actual trial values. You never hand-write "trial 7 used eta=0.083" into a config — if you did, you'd be defeating the point of a search.
  - Local launcher has the same dirty-tree guard as preprocessing (`--allow-dirty` to override) — the `git_sha` tag on a run is only trustworthy because of this guard.

---

## 5. MLflow tracking — DagsHub

MLflow isn't AWS; it's a separate service here (DagsHub-hosted). This is where you actually *read* what a training run did.

- **Code:** [src/mlflow_tracking/mlflow_tracking.py](../src/mlflow_tracking/mlflow_tracking.py) (`configure_mlflow_tracking`, `set_lineage_tags`, `fetch_dagshub_secret`); the logging itself happens in `train.py`'s `log_run`.
- **Where to look:** DagsHub → your repo → **Experiments** tab.
  - **Tags**: `git_sha`, `config_hash`, `data_run_id`, `mode` (`manual`/`hpo`/`production`) — this is the answer to "what exactly produced this model," all in one place.
  - **Params**: the full resolved hyperparameter set, seed, `num_round`, `best_iteration`.
  - **Metrics**: `validation_*` and `train_*` (aucpr, roc_auc, accuracy, precision, recall, f1), plus `threshold`.
  - **Artifacts**: `resolved_config.json`, `features.json`, `pip_freeze.txt` (full installed dependency list — environment lineage), the model itself with its input signature.
  - **Compare**: select two runs and diff them side by side — this is how this week's SageMaker-vs-local comparison was actually verified (run `82022260` on SageMaker vs `a51d39e3` locally: every metric matched to 6 decimal places).
- **Practice point:** this is the concrete implementation of the lineage principle — *every run must be traceable back to the exact code, config, data and environment that produced it.* If you ever can't answer "why did this run score what it scored," something is missing from this list, not acceptable to shrug off.

---

## 6. Picking an HPO winner

- **Code:** [src/evaluation/select_winner.py](../src/evaluation/select_winner.py) — `pick_best` ranks AMT child runs by `validation_aucpr` only, tags the winner `winner=true` on the run and `winner_run_id` on the parent, and (in CI) prints a comparison table into the GitHub Actions job summary.
- **Practice point:** this step only ever looks at validation metrics. It never reads the test split and never registers anything — picking a winner and deciding whether it's good enough are kept as two separate steps on purpose, so you can't accidentally tune your "final" evaluation by re-running this.

---

## 7. Evaluation and the gate

- **Code:** [src/evaluation/evaluate.py](../src/evaluation/evaluate.py), [metrics.py](../src/evaluation/metrics.py).
- **Config:** [config/evaluation/gate.yaml](../config/evaluation/gate.yaml) — `min_aucpr`, `min_recall`, `no_worse_than_production` (currently placeholder thresholds for synthetic data).
- **What happens:** the winning run's model is scored on the **test split, exactly once** (an `evaluated` tag prevents a second scoring unless you explicitly pass `--allow-rescore`), using the threshold already chosen on validation — never re-tuned against test. The gate then checks absolute minimums and, if a `production`-aliased model already exists, that the candidate isn't worse than it.
- **Where to look:** MLflow run → `test_*` metrics and the `evaluation.json` artifact. In CI, a failing gate turns the `evaluate` step red — that is the workflow working correctly, not a bug.
- **Practice point:** a pass only **registers a candidate** — it does not deploy anything. A failed gate means "run a new experiment," not "pick a different trial from the same search" (that would just be test-set p-hacking).

---

## 8. MLflow Model Registry

- **Where to look:** DagsHub → **Models** tab → `fraud-detector`.
- A passing evaluation adds a new version here with no alias. The `production` alias only moves when a human decides to move it.
- **Practice point:** promotion is a deliberate, auditable, manual act — never something a script does because a number crossed a threshold.

---

## 9. Infrastructure — Terraform (`infrastructure/`)

Everything in AWS that this pipeline touches is defined here. **If it's not in Terraform, it doesn't officially exist** — don't create or edit AWS resources by hand in the console, even to "just try something."

| File | What it owns |
|---|---|
| [providers.tf](../infrastructure/providers.tf) | AWS provider, remote state backend (S3 + DynamoDB lock) |
| [s3.tf](../infrastructure/s3.tf) | `raw/`/`processed/`/`models/` prefixes, and the IAM policy granting the SageMaker role read/write access to them |
| [cloudwatch.tf](../infrastructure/cloudwatch.tf) | the two log groups (`ProcessingJobs`, `TrainingJobs`), each with an explicit **14-day retention** — not "never expire" |
| [training.tf](../infrastructure/training.tf) | log-write permissions for the training job |
| [github_oidc.tf](../infrastructure/github_oidc.tf) | the GitHub Actions role — trusted only by `rahulvansh66/transaction-fraud-detection`, scoped to `fraud-*`/`hpo-*` job names only |
| [sagemaker_domain.tf](../infrastructure/sagemaker_domain.tf) | the SageMaker Studio domain |
| [dagshub_secret.tf](../infrastructure/dagshub_secret.tf) | the Secrets Manager secret holding DagsHub credentials |

- **Where to look:** AWS console → IAM, to *read* what these roles can do, not to change them. Any change goes through `terraform plan`/`apply` from your machine (or eventually CI), reviewed first.
- **Practice point:** this is what makes the GitHub Actions role trustworthy — you can read, in one file, the exact and complete list of what the CI pipeline is allowed to do in your AWS account.

---

## 10. CI/CD — GitHub Actions

Covered in depth in [train-workflow-explained.md](model-training/train-workflow-explained.md) — read that for the full walkthrough (triggers, OIDC, steps, failure table). The short version:

- [.github/workflows/ci.yml](../.github/workflows/ci.yml) runs on every pull request (lint, unit tests, `terraform validate`) and **never touches AWS**.
- [.github/workflows/train.yml](../.github/workflows/train.yml) only runs when you click **Run workflow**, and only passes *which config* and *which data run id* — never a hyperparameter. It authenticates to AWS via OIDC (no stored AWS keys in GitHub at all).
- **Practice point:** separating "can this code even run" (CI, cheap, every PR) from "actually spend money training a model" (train.yml, manual button) is deliberate — you don't want a typo in a PR accidentally launching a paid SageMaker job.

---

## 11. If you only check five things regularly

1. **MLflow → Experiments → Compare** the last couple of runs — is the metric trend sane, did a change help or hurt?
2. **CloudWatch** `/aws/sagemaker/TrainingJobs` the moment a job shows Failed — the real error is always in the last ~20 lines of the stream.
3. **S3 `processed/` and `models/`** — spot-check that nothing has an unexpectedly recent "last modified" on a folder you expect to be old; immutability should hold.
4. **The gate result** on any `evaluate` run — pass/fail and which threshold it tripped, in `evaluation.json` or the MLflow `test_*` metrics.
5. **`terraform plan`** output before ever running `apply` — read the diff, don't just trust it.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Channel** | A named input a SageMaker training job receives (here: `train`, `val`, `meta`), each mapped to an S3 path. |
| **AMT (Automatic Model Tuning)** | SageMaker's hyperparameter search feature; we call this "HPO" in this repo's `kind: hpo`. |
| **OIDC** | A way for GitHub Actions to get short-lived AWS credentials by proving "I am a workflow from this exact repo," without any AWS access key ever being stored in GitHub. |
| **IaC (Infrastructure as Code)** | All AWS resources are declared in Terraform files, not clicked together in the console. |
| **Registry alias** | A movable label (e.g. `production`) on a specific Model Registry version — moving the alias is how you promote a model, without deleting or duplicating anything. |
| **Lineage** | See section 1 — being able to trace a model back to its exact code/config/data/environment. |

---

## 13. All the docs in this repo, by topic

- **This file** — the map.
- **Dataset & features:** [dataset-info.md](dataset-info.md), [feature-engineering.md](feature-engineering.md).
- **Preprocessing:** [preprocessing/modularization.md](preprocessing/modularization.md), [preprocessing/preprocessing-pyspark-processing-plan.md](preprocessing/preprocessing-pyspark-processing-plan.md), [preprocessing/preprocessing-pyspark-implementation-steps.md](preprocessing/preprocessing-pyspark-implementation-steps.md).
- **Training pipeline build log & decisions:** [model-training/model-training-plan.md](model-training/model-training-plan.md), [model-training/model-training-implementation-steps.md](model-training/model-training-implementation-steps.md).
- **GitHub Actions deep dive:** [model-training/train-workflow-explained.md](model-training/train-workflow-explained.md).
- **Background/planning notes** (less polished, some superseded): `my-notes/` — the MLOps target operating model, HPO architecture comparison, and a future real-time serving design not yet built.

---

# How to do training on new hyperpara on same preprocessed data?

Preprocessing doesn't rerun. You point a new training run at the same immutable `processed/<run_id>/` and change only the hyperparameters.

**What you do**
1. Copy [experiment-001.yaml](config/experiments/experiment-001.yaml) to a new file, for example `experiment-002.yaml`.
2. Change only the hyperparameters:
   - `manual_params` for a single deliberate run (`--kind manual`).
   - `search_space` and `tuning` for an AMT search (`--kind hpo`).
3. Keep `data.run_id` the same, or pass `data_run_id` when you trigger the [train workflow](.github/workflows/train.yml).
4. Commit the config, then dispatch the workflow with `experiment=config/experiments/experiment-002.yaml`, `data_run_id=<same run_id>` and the `kind` you want.

**What happens next**
- Training reads `processed/<run_id>/{train,val,test}` and `metadata.json` unchanged, so the features, label, `scale_pos_weight` and splits are identical.
- That makes the runs comparable. Any difference in metrics comes from the hyperparameters and not from the data.
- Each run is a separate MLflow run with its own config, git commit and hyperparameters. All of them record the same `data_run_id`.
- For `hpo`, `select_winner` picks the best trial. Then `evaluate` scores it on the test split, applies the gate and registers the model.

**Why this is safe**
- The data folder is never overwritten (`errorifexists`) and `latest` is rejected ([sagemaker_jobs.py:48](src/training/sagemaker_jobs.py#L48)). The same `run_id` therefore always means the same bytes.
- Hyperparameters live only in `config/experiments/*.yaml`, so the experiment file plus the data `run_id` fully define a run.

**When you must re-preprocess instead:** if you change anything in `metadata.json`, such as features, split fractions, `keep_types` or `night_hours`. That produces a new `run_id` (new commit or config hash). The old one stays untouched.

**Current data:** `synthetic-v0` and `3c5bc15-…-dirty` have only a handful of rows, and the dirty run's val split has no fraud rows. New hyperparameters will run on them, but the metrics won't mean anything until you have real preprocessed data.

---

> Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
