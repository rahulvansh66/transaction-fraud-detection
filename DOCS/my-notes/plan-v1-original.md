Here's the plan as plain markdown you can copy straight into a file:

```markdown
# Fraud Detection Pipeline — Build Plan

## v1 — Training + Inference Pipeline (SageMaker, MLflow, S3)

### Step 1: Set up AWS foundations
- Create/access an AWS account (sandbox is fine for learning)
- Create an IAM role for SageMaker with S3 and CloudWatch permissions
- Create one S3 bucket with folders:
  - `raw/` — incoming transaction data
  - `processed/` — cleaned/feature-engineered data
  - `models/` — trained model artifacts
  - `mlflow-artifacts/` — MLflow experiment artifacts
- Launch a SageMaker Studio domain (your notebook environment for everything below)

### Step 2: Stand up MLflow tracking
- Use SageMaker's managed MLflow capability (launched from Studio — no EC2/server to manage)
- Point its artifact store at your S3 bucket (`mlflow-artifacts/`)
- This is where every experiment run, metric, and model version gets logged going forward

### Step 3: Build the preprocessing step
- Write a preprocessing script (pandas/sklearn): clean raw transaction data, engineer features, split train/test
- Run it as a SageMaker Processing job (decouples data prep from training — this is how production pipelines do it, not inline in a notebook)
- Output goes to `processed/` in S3

### Step 4: Train with experiment tracking
- Write two training scripts:
  - `train_rcf.py` — Random Cut Forest (anomaly detection)
  - `train_xgboost.py` — XGBoost (fraud detection)
- Use SageMaker's built-in algorithm containers or your own script in a SageMaker Training job
- Wrap training calls with MLflow logging: `mlflow.log_param`, `log_metric`, `log_model`
- Every run becomes comparable in the MLflow UI

### Step 5: Register the best model
- Compare runs in MLflow, pick the winner for each model type
- Register it in the MLflow Model Registry (or promote to SageMaker Model Registry)
- This is your first real "promotion gate" — only explicitly registered models get deployed

### Step 6: Deploy a real-time inference endpoint
- Deploy the registered model(s) as SageMaker real-time endpoint(s)
- One endpoint per model to start (multi-model endpoint later once comfortable)
- Test by sending a sample transaction payload via the SageMaker SDK or boto3 — no API Gateway yet, just endpoint-to-endpoint validation

### Step 7: Validate end to end
- Run known-fraud and known-good transactions through both endpoints
- Confirm outputs make sense
- Confirm traceability: run → model version → registry entry → deployed endpoint, all linked through MLflow

**Suggested file layout:**
```
project/
├── preprocess.py
├── train_rcf.py
├── train_xgboost.py
├── deploy.py
└── notebooks/
    └── exploration.ipynb
```

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
- CI/CD for model retraining and redeployment
- Full production hardening (VPC, encryption, alerting, IaC)
```

That last section ties back to the Kafka architecture from earlier — once v1 is solid, its endpoint is exactly the "ML inference" box in those diagrams.