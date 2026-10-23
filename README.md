# transaction-fraud-detection

Production-grade, reproducible fraud-detection pipeline on AWS: PySpark preprocessing (SageMaker Processing) -> XGBoost training / AMT hyperparameter search (SageMaker) -> validation-based winner selection -> one-shot test evaluation with a gate -> MLflow Model Registry (DagsHub). All AWS resources are defined in Terraform; every run is traceable to its git sha, config hash and immutable data run.

## Repository layout

| Path | Purpose |
|---|---|
| `src/config_loader/` | Shared YAML config loading (`config/env/{env}.yaml`, secrets from `.env`) |
| `src/mlflow_tracking/` | Points MLflow at the DagsHub server |
| `src/preprocessing/` | `features.py` (PySpark feature logic), `spark_job.py` (job entry point), `run_preprocessing_job.py` (SageMaker launcher), `lineage.py` (run id / config hash) |
| `src/training/` | `experiment_config.py` (validation and plan: single / grid / random / bayesian), `train.py` (script-mode entry point), `model.py`, `data.py`, `sagemaker_jobs.py` (boto3 request builders), `hpo_tuner.py` (AMT), `run_training_job.py` (launcher), `lineage.py` |
| `src/evaluation/` | `metrics.py` (shared metrics and threshold choice), `select_winner.py` (validation only), `evaluate.py` (test once, gate, register) |
| `src/inference/`, `src/monitoring/` | Reserved for serving and monitoring (not implemented yet) |
| `config/` | `env/` (dev/prod values), `preprocessing/`, `experiments/` (numbered, immutable once run), `production/model.yaml`, `evaluation/gate.yaml`, `inference/feature_contract.yaml` |
| `infrastructure/` | Terraform: S3, IAM, GitHub OIDC, SageMaker domain, training resources, CloudWatch, DagsHub secret |
| `scripts/` | Data helpers: CSV to Hive parquet, synthetic splits, S3 upload |
| `tests/` | `unit/` (config, preprocessing, training, evaluation) and `integration/` (Spark feature parity) |
| `DOCS/` | Design notes and implementation plans |

## Running the notebooks

Dependencies are managed with [uv](https://docs.astral.sh/uv/); the environment lives in `.venv/` at the repo root.

1. Install everything (including dev tools such as `ipykernel`):
   ```powershell
   uv sync
   ```
2. Register the kernel once (optional, for Jupyter/JupyterLab outside VS Code):
   ```powershell
   uv run python -m ipykernel install --user --name transaction-fraud-detection --display-name "Python (transaction-fraud-detection)"
   ```
3. Select the kernel in VS Code: open a notebook, click **Select Kernel** (top-right) →
   **Python Environments…** → **.venv**. If it is not listed, run
   **Developer: Reload Window**, or **Python: Select Interpreter** →
   `.venv\Scripts\python.exe`.
4. Verify: a cell running `import sys; sys.executable` should print a path inside `.venv`.

Headless run (e.g. to check a notebook end to end):
```powershell
uv run jupyter nbconvert --to notebook --execute notebooks/<name>.ipynb --output-dir <tmp-dir>
```

## How to run preprocessing

`src/preprocessing/spark_job.py` reads raw Hive-partitioned parquet, builds the features in `features.py` (filter to `keep_types`, time-based split on `step`, class weight) and writes `processed/<run_id>/{train,val,test}` plus `metadata.json`. The run id is `<git_sha>-<config_hash>-<data_version>`, so output is immutable and never overwritten. Settings live in `config/preprocessing/preprocessing.yaml`; feature semantics shared with serving are in `config/inference/feature_contract.yaml`.

```bash
uv run python -m src.preprocessing.run_preprocessing_job --env dev --data-version data-v0
```
Requires a clean git tree (`--allow-dirty` for smoke tests) and AWS credentials. Add `--no-wait` to submit and return. Logs go to CloudWatch.

## How to run a training job

Hyperparameters live in `config/experiments/*.yaml`, never in code or workflows. Each file has its own number, is self-contained and asks one question through its `params:` block: a scalar is fixed, a list is a set of discrete values, and a `{type, min, max}` dict is a range. All scalars is one training job (`experiment-002-manual-params.yaml`); lists with `tuning.strategy: grid` try every value once (`experiment-004-depth-sensitivity.yaml`); ranges with `strategy: bayesian` run an AMT search (`experiment-003-search-space.yaml`). The strategy is always explicit for anything tunable (`grid`, `random` or `bayesian`), and grid supports lists only. The file is validated before any job is submitted. Never edit a file after it has produced runs; start the next number instead. Training reads an immutable processed data run (`processed/<run_id>/{train,val,test}` plus `metadata.json`), never `latest`.

### 1. Prerequisites
- `uv sync`, and a `.env` with the DagsHub credentials (see `.env.example`) for local runs.
- Infrastructure applied with Terraform (`infrastructure/`); SageMaker jobs also need AWS credentials.
- Preprocessing has produced the data run you want to train on.

### 2. Run locally (no AWS)
```bash
uv run python scripts/make_synthetic_splits.py   # optional: synthetic-v0 test data
uv run python -m src.training.run_training_job --mode local --kind experiment \
  --experiment config/experiments/experiment-002-manual-params.yaml
```
Runs `train.py` as a subprocess and logs a run to DagsHub MLflow. The same seed reproduces the same metrics.

### 3. Run on SageMaker
The git tree must be clean: commit first (`--allow-dirty` is for smoke tests only). Local mode runs single (all-scalar) experiments only.
```bash
# One training job (all-scalar params)
uv run python -m src.training.run_training_job --mode sagemaker --kind experiment \
  --experiment config/experiments/experiment-002-manual-params.yaml --data-run-id <run_id>

# Grid over lists (3 trials)
uv run python -m src.training.run_training_job --mode sagemaker --kind experiment \
  --experiment config/experiments/experiment-004-depth-sensitivity.yaml --data-run-id <run_id>

# Bayesian search over ranges (AMT); uses the tuning block of the YAML
uv run python -m src.training.run_training_job --mode sagemaker --kind experiment \
  --experiment config/experiments/experiment-003-search-space.yaml --data-run-id <run_id>
```
Add `--no-wait` to submit and return, and `--output-json run.json` to save the MLflow run id for the next steps. Logs go to CloudWatch `/aws/sagemaker/TrainingJobs`.

### 4. Pick a winner, evaluate, register (after a grid/random/bayesian search)
```bash
uv run python -m src.evaluation.select_winner --parent-run-id <parent_run_id>   # validation metrics only
uv run python -m src.evaluation.evaluate --run-id <winner_run_id>               # test scored once, gate in config/evaluation/gate.yaml
```
A passing model is registered in the MLflow Model Registry as a candidate. Promotion to production is manual. A failed winner is final: run a new experiment instead of picking another trial.

### 5. Tests and lint
```bash
uv run pytest tests/unit          # fast, no AWS
uv run pytest tests/integration   # Spark feature parity, needs Java
uv run ruff check .
```

### 6. Run from GitHub Actions
- `ci.yml` runs on pull requests (ruff, unit tests, `terraform validate`). It never starts a job.
- `train.yml` starts only on **Actions → train → Run workflow** (inputs: experiment file, data run id, kind) or its weekly schedule (`production` kind). It authenticates via OIDC using the repository variables `AWS_OIDC_ROLE_ARN` and `AWS_REGION`.
- Pushing or committing code does not start a training job.

## Author

Built by Rahul Vansh · [LinkedIn](https://www.linkedin.com/in/rahul-vansh/)

---

> Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
