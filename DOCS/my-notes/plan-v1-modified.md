  Here's the plan as plain markdown you can copy straight into a file:

# Fraud Detection Pipeline — Build Plan

## v1 — Training + Inference Pipeline (SageMaker, DagsHub MLflow, S3)

### Step 1: Set up AWS foundations
- Create/access an AWS account (sandbox is fine for learning)
- Create an IAM role for SageMaker with S3 and CloudWatch permissions
- Create one S3 bucket with folders:
  - `raw/` — incoming transaction data
  - `processed/` — cleaned/feature-engineered data
  - `models/` — trained model artifacts
- Setup SageMaker Studio domain if doesnt have.

### Step 2: Set up MLflow tracking on DagsHub
- Create a DagsHub repo (connect it to the GitHub repo) — DagsHub hosts a managed MLflow tracking server + model registry, so there is nothing to run on AWS
- Tracking URI: `https://dagshub.com/<user>/<repo>.mlflow`; artifacts are stored on DagsHub's storage
- Authenticate with `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` (DagsHub access token); keep them in `.env` locally and in AWS Secrets Manager (`fraud-detection/<env>/dagshub-mlflow`, defined in Terraform at `infrastructure/dagshub_secret.tf`; token passed via `TF_VAR_dagshub_token`) for SageMaker jobs — never in code or logs
- Grant the SageMaker execution role `secretsmanager:GetSecretValue` on that secret (policy document in the same Terraform file)
- Allow outbound internet (NAT/no VPC-isolation) for SageMaker jobs so they can reach DagsHub
- This is where every experiment run, metric, and model version gets logged going forward

### Step 3: Build the preprocessing step
- Start by prototyping your preprocessing/feature engineering logic on a small sample in pandas/sklearn in a notebook locally (fast iteration)
: Apply appropriate preprocessing steps, engineer features, split train/test. 

- Port the finalized logic to PySpark for the actual Processing job. That way you're not debugging Spark's distributed quirks while you're still figuring out what features you want.
- Run it as a SageMaker Spark Processing job (decouples data prep from training — this is how production pipelines do it, not inline in a notebook)
- Output goes to `processed/` in S3

### Step 4: Train with experiment tracking
- Write training scripts:
  - `train_xgboost.py` — XGBoost (fraud detection)
- Use SageMaker's built-in algorithm containers or your own script in a SageMaker Training job
- Wrap training calls with MLflow logging: `mlflow.log_param`, `log_metric`, `log_model`
- Every run becomes comparable in the DagsHub MLflow UI

### Step 5: Register the best model
- Compare runs in MLflow, pick the winner for each model type
- Register it in the DagsHub-hosted MLflow Model Registry (or promote to SageMaker Model Registry)
- This is your first real "promotion gate" — only explicitly registered models get deployed
- CI/CD for model trianing/retraining 

### Step 6: Deploy a real-time inference endpoint
- Deploy the registered model(s) as SageMaker real-time endpoint(s)
- One endpoint per model to start (multi-model endpoint later in future once comfortable)
- Test by sending a sample transaction payload via the SageMaker SDK or boto3 — no API Gateway yet, just endpoint-to-endpoint validation
- CI/CD for model deployment

### Step 7: Validate end to end
- Run known-fraud and known-good transactions through both endpoints
- Confirm outputs make sense
- Confirm traceability: run → model version → registry entry → deployed endpoint, all linked through MLflow

**Tips:**
- Do steps 1–3 interactively in a notebook first before wrapping in "real" Processing/Training jobs — easier to debug pandas locally than in CloudWatch logs
- Keep scripts separate and clearly named from the start; it keeps MLflow run names and S3 structure readable as things grow

---

## v1.2 — Add API Gateway + Lambda

- Wrap the v1 SageMaker endpoint(s) behind API Gateway → Lambda so they're callable over HTTP instead of via the SDK
- Learning focus: IAM permissions between API Gateway, Lambda, and SageMaker; request/response shaping; basic error handling
- Lambda invokes the SageMaker endpoint (`invoke_endpoint`), formats the response, returns it through API Gateway

---

## v1.3 — Add model monitoring with Evidently

- Compute reference vs. current data reports using Evidently: feature drift, prediction drift, data quality
- Decide where reports get stored/surfaced (e.g. S3 + a simple dashboard, or CloudWatch alarms on drift thresholds)
- Learning focus: what drift looks like in practice, and how monitoring plugs into a deployed endpoint without disrupting it

---

## Later (not yet planned in detail)
- Streaming ingestion (Kafka/MSK) in front of the pipeline
- Multi-model endpoints / autoscaling
- Full production hardening (VPC, encryption, alerting, IaC)