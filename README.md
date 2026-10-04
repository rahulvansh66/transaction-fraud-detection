# transaction-fraud-detection
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

## How to run a training job

Hyperparameters and search ranges live in `config/experiments/*.yaml`, never in code or workflows. Training reads an immutable processed data run (`processed/<run_id>/{train,val,test}` plus `metadata.json`), never `latest`.

### 1. Prerequisites
- `uv sync`, and a `.env` with the DagsHub credentials (see `.env.example`) for local runs.
- Infrastructure applied with Terraform (`infrastructure/`); SageMaker jobs also need AWS credentials.
- Preprocessing has produced the data run you want to train on.

### 2. Run locally (no AWS)
```bash
uv run python scripts/make_synthetic_splits.py   # optional: synthetic-v0 test data
uv run python -m src.training.run_training_job --mode local --kind manual \
  --experiment config/experiments/experiment-001.yaml
```
Runs `train.py` as a subprocess and logs a run to DagsHub MLflow. The same seed reproduces the same metrics.

### 3. Run on SageMaker
The git tree must be clean (or pass `--allow-dirty`).
```bash
# One training job
uv run python -m src.training.run_training_job --mode sagemaker --kind manual \
  --experiment config/experiments/experiment-001.yaml --data-run-id <run_id>

# Hyperparameter search (AMT); uses tuning.max_jobs / max_parallel_jobs from the YAML
uv run python -m src.training.run_training_job --mode sagemaker --kind hpo \
  --experiment config/experiments/experiment-001.yaml --data-run-id <run_id>
```
Add `--no-wait` to submit and return, and `--output-json run.json` to save the MLflow run id for the next steps. Logs go to CloudWatch `/aws/sagemaker/TrainingJobs`.

### 4. Pick a winner, evaluate, register (after an hpo run)
```bash
uv run python -m src.evaluation.select_winner --parent-run-id <parent_run_id>   # validation metrics only
uv run python -m src.evaluation.evaluate --run-id <winner_run_id>               # test scored once, gate in config/evaluation/gate.yaml
```
A passing model is registered in the MLflow Model Registry as a candidate. Promotion to production is manual. A failed winner is final: run a new experiment instead of picking another trial.

### 5. Run from GitHub Actions
- `ci.yml` runs on pull requests (ruff, unit tests, `terraform validate`). It never starts a job.
- `train.yml` starts only on **Actions → train → Run workflow** (inputs: experiment file, data run id, kind) or its weekly schedule (`production` kind). It authenticates via OIDC using the repository variables `AWS_OIDC_ROLE_ARN` and `AWS_REGION`.
- Pushing or committing code does not start a training job.

## Author

Built by Rahul Vansh · [LinkedIn](https://www.linkedin.com/in/rahul-vansh/)

---

> Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
