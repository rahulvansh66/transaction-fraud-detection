# Production MLOps Architecture: SageMaker, AMT, MLflow, Git, and GitHub Actions

## Executive Recommendation

After comparing the existing production experimentation architecture (DOCS\production_ml_experimentation_git_sagemaker_amt_mlflow.md) with Sofian Hamiti's `amazon-sagemaker-github-actions-mlflow` repository (git repo: https://github.com/sofianhamiti/amazon-sagemaker-github-actions-mlflow , blog: https://medium.com/data-science/5-simple-steps-to-mlops-with-github-actions-mlflow-and-sagemaker-pipelines-19abf951a70) , the recommended approach is a **hybrid**:

- Keep the existing architecture as the **production ML experimentation and HPO operating model**.
- Use the GitHub repository as an **implementation reference** for CI/CD, Docker, ECR, infrastructure, SageMaker execution, and deployment.
- Do not copy the repository literally; it is primarily a practical MLOps implementation example and dates from 2022.
- Modernize the implementation where appropriate using current SageMaker and MLflow capabilities.

**Overall recommendation: Hybrid approach — approximately 9.5/10.**

---

## 1. What Each Architecture Is Optimized For

### Existing architecture

The existing architecture answers:

> **How should an ML engineer experiment, perform HPO, approve a model, and move it into production?**

Its central lifecycle is:

```text
Manual experimentation
        ↓
Learn promising region
        ↓
AMT
        ↓
Best configuration
        ↓
Final evaluation
        ↓
MLflow Model Registry
        ↓
Approval
        ↓
Production
        ↓
Retraining with approved configuration
```

This is particularly strong because it explicitly separates experimentation from production retraining.

### GitHub repository

Sofian Hamiti's repository answers more:

> **How do I automate an ML project from GitHub through SageMaker, MLflow, CI/CD, and deployment?**

It is useful as an implementation template covering areas such as:

- GitHub Actions
- SageMaker Pipelines
- MLflow
- Docker
- ECR
- deployment
- CDK/infrastructure
- project structure

Therefore, the two approaches are complementary rather than competing.

---

# 2. Side-by-Side Comparison

| Area | Existing Architecture | GitHub Repository | Preferred |
|---|---|---|---|
| Manual experimentation | Explicitly supported | Less central | **Existing architecture** |
| Pipeline Parameters | Core concept | Less central | **Existing architecture** |
| SageMaker AMT/HPO | Explicitly designed | Not the main focus | **Existing architecture** |
| Narrowing HPO using experiments | Explicitly supported | Not emphasized | **Existing architecture** |
| Separating HPO from production retraining | Strongly defined | Less explicit | **Existing architecture** |
| MLflow experiment tracking | Yes | Yes | Tie |
| MLflow Model Registry | Yes | Yes | Tie |
| Git/config lineage | Yes | Yes | Tie |
| SageMaker Pipeline orchestration | Yes | Yes | Tie |
| GitHub Actions CI/CD | Conceptual | Implemented | **GitHub repository** |
| Docker/containerization | Conceptual | Implemented | **GitHub repository** |
| ECR | Conceptual | Implemented | **GitHub repository** |
| Deployment | Conceptual | Implemented | **GitHub repository** |
| Infrastructure/CDK | Limited | Implemented | **GitHub repository** |
| Generic across ML problem types | Strong | Example-oriented | **Existing architecture** |
| Experimentation mental model | Strong | Less detailed | **Existing architecture** |
| Practical implementation reference | Limited | Strong | **GitHub repository** |

---

# 3. The Most Important Design Decision: Manual Experiments vs AMT

A production ML system should **not** assume that every experiment needs to go through SageMaker AMT.

The recommended model is:

```text
                    ML DEVELOPMENT
                           |
             +-------------+-------------+
             |             |             |
             ↓             ↓             ↓
          MANUAL          HPO        PRODUCTION
       EXPERIMENTS         |         RETRAINING
             |             AMT             |
             |             |               |
             +-------------+---------------+
                           |
                           ↓
                   Best configuration
                           |
                           ↓
                    Final evaluation
                           |
                           ↓
                  MLflow Model Registry
                           |
                           ↓
                       Production
```

### Manual experimentation

Manual experiments answer:

> **What should I try?**

For example:

```yaml
hyperparameters:
  learning_rate: 0.0005
  batch_size: 64
  dropout: 0.1
```

The same SageMaker Pipeline can be executed with different values:

```text
Pipeline v7

Run 101 → lr=.001,  batch=64
Run 102 → lr=.0005, batch=64
Run 103 → lr=.0001, batch=128
```

No AMT is required.

This is important because ML engineers often want to deliberately compare:

```text
optimizer = Adam
optimizer = AdamW
```

or:

```text
feature_set = A
feature_set = B
```

or:

```text
architecture = A
architecture = B
```

These are engineering hypotheses, not necessarily problems that should be handed to an automated optimizer.

---

# 4. Why Pipeline Parameters Matter

Pipeline Parameters provide **execution-level runtime inputs**.

For example:

```text
Git configuration
       ↓
GitHub Actions
       ↓
SageMaker Pipeline
       ↓
Pipeline Parameters
       ↓
Training Job
       ↓
MLflow
```

Possible parameters include:

```text
DatasetVersion
ExperimentName
LearningRate
BatchSize
MaxTrainingJobs
MaxParallelJobs
```

However:

> **Do not turn every YAML field into a Pipeline Parameter.**

Only expose values that genuinely need runtime variation.

Pipeline Parameters are execution inputs; they are not a replacement for experiment tracking.

---

# 5. Manual Experiments Should Feed HPO

Manual experimentation and AMT are complementary.

Recommended workflow:

```text
Human-guided experimentation
           ↓
Understand the problem
           ↓
Identify promising region
           ↓
Define focused search space
           ↓
AMT
           ↓
Systematic search
```

Example:

### Stage 1 — Manual experiments

```text
lr=.001   → metric=.72
lr=.0005  → metric=.76
lr=.0001  → metric=.71
```

This suggests that the useful region may be around `0.0005`.

### Stage 2 — AMT

```yaml
hpo:
  learning_rate:
    min: 0.0003
    max: 0.002

  batch_size:
    values: [32, 64, 128]

  dropout:
    min: 0.0
    max: 0.2
```

AMT can then systematically search the focused region.

This is preferable to immediately giving AMT a huge, uninformed search space.

---

# 6. AMT's Responsibility

A useful separation is:

```text
Git
 ↓
Defines search-space intent

Pipeline Parameters
 ↓
Runtime configuration

AMT
 ↓
Generates and evaluates trials

MLflow
 ↓
Records experiment evidence
```

For example:

```text
AMT search space:

learning_rate = [0.0003, 0.002]
batch_size = [32, 64, 128]
dropout = [0, 0.2]
```

AMT generates individual trials:

```text
Trial 1 → lr=.0007, batch=32,  dropout=.05
Trial 2 → lr=.0014, batch=64,  dropout=.10
Trial 3 → lr=.0004, batch=128, dropout=.15
...
Trial 17 → lr=.0009, batch=64, dropout=.08  ← best
```

The key rule:

> **Git stores the search-space definition. AMT generates the individual trials.**

---

# 7. Do Not Run AMT for Every Production Retraining

This is one of the strongest parts of the recommended architecture.

AMT answers:

> **What configuration should we use?**

Production retraining answers:

> **Train the approved configuration on the latest approved data.**

These are different workflows.

After AMT finds an acceptable configuration:

```text
AMT winner
    ↓
Final evaluation
    ↓
Acceptance gates
    ↓
MLflow Model Registry
    ↓
Approval
    ↓
Production
```

Then production retraining can use the approved configuration:

```text
Latest approved data
        ↓
Approved fixed configuration
        ↓
Training
        ↓
Evaluation
        ↓
MLflow
        ↓
Approval
        ↓
Production
```

There is normally no reason to rerun HPO every time the model is retrained.

---

# 8. Production Configuration Should Be Explicit

Once a configuration is selected and approved, record it explicitly.

Example:

```yaml
# production/model.yaml

model_version: v27
data_version: v10

hyperparameters:
  learning_rate: 0.0009
  batch_size: 64
  dropout: 0.08
```

Production then becomes:

```text
production config
       ↓
SageMaker Pipeline
       ↓
Training
       ↓
Evaluation
       ↓
MLflow
       ↓
Model Registry
       ↓
Approval
       ↓
Production
```

This makes production retraining reproducible.

---

# 9. MLflow's Role

MLflow should provide the **experiment and model lineage layer**.

For a manual run, capture:

```text
Experiment
Git commit
Dataset version
Hyperparameters
Metrics
Artifacts
Environment
```

For an AMT trial, additionally capture:

```text
Trial ID
Hyperparameters generated by AMT
Metrics
Artifacts
Git commit
Dataset version
```

Then register the selected model in the MLflow Model Registry.

The responsibility separation becomes:

```text
Git
 ↓
Code + configuration lineage

SageMaker
 ↓
Pipeline/job orchestration

AMT
 ↓
HPO trials

MLflow
 ↓
Experiment tracking + model registry

S3
 ↓
Datasets + model artifacts
```

---

# 10. Production Evaluation and Approval

The AMT winner should **not automatically be deployed**.

Use an evaluation and gating stage:

```text
AMT winner
    ↓
Final evaluation
    |
    +---- FAIL → Reject
    |
    +---- PASS
            ↓
      MLflow Registry
            ↓
         Approval
            ↓
        Production
```

Possible gates include:

- offline model metrics
- regression tests
- data-quality checks
- robustness checks
- business KPI checks
- latency/resource checks
- safety/fairness checks where relevant

---

# 11. What to Borrow from the GitHub Repository

The repository is particularly useful for the **implementation layer**.

## 11.1 GitHub Actions

Use GitHub Actions for:

```text
Code change
    ↓
Tests / validation
    ↓
Build
    ↓
Deploy/update SageMaker Pipeline
    ↓
Execute pipeline when appropriate
```

The existing architecture already has GitHub Actions conceptually; the repository provides a concrete implementation reference.

---

## 11.2 Docker

For production, package the training environment into a reproducible container:

```text
Training code
      ↓
Docker image
      ↓
ECR
      ↓
SageMaker Training Job
```

This makes dependencies and runtime environments more reproducible.

---

## 11.3 ECR

Use ECR as the container registry:

```text
GitHub Actions
      ↓
Build Docker image
      ↓
Push to ECR
      ↓
SageMaker
```

Avoid tying production training to an uncontrolled local environment.

---

## 11.4 Infrastructure as Code

The repository also provides CDK/infrastructure examples.

For a production system, infrastructure should ideally be reproducible:

```text
Infrastructure as Code
        ↓
AWS resources
        ↓
SageMaker
S3
ECR
IAM
Networking
Endpoints
```

Do not rely on manually configured AWS resources for a serious production system.

---

# 12. What NOT to Copy Literally from the Repository

The repository is a useful reference, but it should not be treated as the final production blueprint.

For example, its configuration contains problem-specific names such as:

```yaml
model:
  name: housing-random-forest
```

For a reusable production architecture, use generic/problem-specific configuration instead:

```yaml
model:
  name: <problem-specific-model>
  version: <version>
```

Similarly, deployment configuration should be separated cleanly from experimentation configuration.

The architecture should remain applicable to:

- classification
- regression
- ranking
- recommendation
- NLP
- computer vision
- other ML problems

---

# 13. Recommended Hybrid Production Architecture

The strongest architecture combines the two approaches:

```text
                              Git
                               |
                      code + experiment config
                               |
                               v
                        GitHub Actions
                         /                                   /                               CI / Tests       Build Image
                                      |
                                      v
                                     ECR
                                      |
                                      v
                           SageMaker Pipeline
                                      |
                    +-----------------+----------------+
                    |                                  |
                    v                                  v
              MANUAL MODE                         HPO MODE
                    |                                  |
          Pipeline Parameters                          AMT
                    |                                  |
                    |                            multiple trials
                    |                                  |
                    +------------------+---------------+
                                       |
                                       v
                                    MLflow
                                       |
                                       v
                              Evaluation / Gates
                                       |
                              +--------+--------+
                              |                 |
                             FAIL              PASS
                              |                 |
                           Reject                v
                                         MLflow Registry
                                                |
                                             Approval
                                                |
                                                v
                                           Production
                                                |
                                                v
                                      Scheduled Retraining
                                                |
                                                v
                                     Approved fixed config
```

This is the recommended target architecture.

---

# 14. Repository Structure

A production-oriented repository can combine the configuration-driven experimentation model with the implementation patterns from the GitHub project:

```text
ml-project/
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── pipeline.py
│   └── inference.py
│
├── configs/
│   ├── experiments/
│   │   ├── experiment-001.yaml
│   │   └── experiment-002.yaml
│   │
│   └── production/
│       └── model.yaml
│
├── tests/
│
├── docker/
│   └── Dockerfile
│
├── infrastructure/
│   └── ...
│
├── requirements.txt
│
└── .github/
    └── workflows/
        ├── ci.yml
        ├── build.yml
        └── ml-pipeline.yml
```

The key principle remains:

> **Training code should not need to change just because a hyperparameter changes.**

---

# 15. Configuration Responsibilities

Keep responsibilities separated.

| Configuration | Purpose |
|---|---|
| Experiment config | Manual experiment or HPO definition |
| Production config | Approved production configuration |
| Pipeline parameters | Runtime overrides |
| Docker configuration | Runtime/dependency environment |
| Infrastructure config | AWS resources |
| Deployment config | Serving infrastructure |

Avoid putting everything into one YAML file.

---

# 16. Reproducibility Requirements

A production model should be traceable to:

```text
Code version
+
Experiment/config version
+
Dataset version
+
Pipeline version
+
Execution parameters
+
Environment/container version
+
Hyperparameters
+
Metrics
+
Model artifact
```

For example:

```text
MLflow Model v27
|
+-- Git commit: abc123
+-- Config: experiment-002
+-- Dataset: v10
+-- Pipeline: v7
+-- AMT trial: 17
+-- learning_rate: .0009
+-- batch_size: 64
+-- dropout: .08
+-- validation_metric: .80
+-- container/image version
```

This is the lineage required to reproduce and audit a production model.

---

# 17. Production Best-Practice Rules

## Rule 1 — Keep training code configuration-driven

Prefer:

```python
train(
    learning_rate=config["learning_rate"],
    batch_size=config["batch_size"],
)
```

instead of:

```python
learning_rate = 0.0009
```

hard-coded inside the training implementation.

---

## Rule 2 — Version experiment configurations

Every meaningful experiment should be traceable to a Git commit.

---

## Rule 3 — Use Pipeline Parameters for runtime variation

Use them when values genuinely vary between executions.

---

## Rule 4 — Keep AMT search spaces intentional

Use:

- domain knowledge
- baseline experiments
- previous experiments
- literature
- previous production models

to construct a reasonable search space.

---

## Rule 5 — Track important runs in MLflow

At minimum record:

```text
parameters
metrics
artifacts
code version
data version
environment
experiment name
```

---

## Rule 6 — Keep HPO separate from production retraining

```text
HPO:
"What configuration should we use?"

Production:
"Train the approved configuration."
```

---

## Rule 7 — Keep production configuration explicit

Once a configuration is approved, record it explicitly.

---

## Rule 8 — Use immutable/versioned datasets and artifacts

Avoid relying on moving paths such as:

```text
s3://bucket/latest/
```

for reproducibility.

Prefer versioned datasets and artifacts.

---

## Rule 9 — Do not automatically promote the AMT winner

Always perform evaluation and approval.

---

## Rule 10 — Do not change code for every hyperparameter experiment

Configuration should change; training implementation should remain stable.

---

# 18. Final Responsibility Map

| Component | Responsibility |
|---|---|
| **Git** | Code + versioned experiment/config definitions |
| **GitHub Actions** | CI/CD and triggering pipeline executions |
| **Docker/ECR** | Reproducible training/runtime environment |
| **SageMaker Pipeline** | End-to-end workflow orchestration |
| **Pipeline Parameters** | Runtime inputs/overrides |
| **SageMaker AMT** | Systematic hyperparameter search |
| **MLflow Tracking** | Parameters, metrics, artifacts, experiment history |
| **MLflow Model Registry** | Model versions and promotion state |
| **S3** | Data and model artifacts |
| **Infrastructure as Code** | Reproducible AWS infrastructure |
| **Production pipeline** | Train/evaluate approved configuration |

---

# 19. Interview-Ready Architecture

A concise way to explain the architecture:

```text
                         GIT
                          |
                     code + config
                          |
                          v
                    GitHub Actions
                          |
                          v
                 SageMaker Pipeline
                          |
              +-----------+-----------+
              |                       |
           MANUAL                    HPO
              |                       |
      Pipeline Parameters             AMT
              |                       |
              |                multiple trials
              |                       |
              +-----------+-----------+
                          |
                          v
                       MLflow
                          |
                       Evaluation
                          |
                          v
                 MLflow Registry
                          |
                       Approval
                          |
                          v
                      Production
                          |
                          v
                Periodic Retraining
                          |
                          v
                Approved configuration
```

### Strong interview answer

> "I separate experimentation from production retraining. During development, I can manually test specific hyperparameter values by passing them as SageMaker Pipeline Parameters without changing the training code. Once I understand the promising region, I define a Git-versioned search space and use SageMaker AMT to systematically search it. I track both manual runs and AMT trials in MLflow, evaluate the selected configuration, and register the approved model in MLflow Model Registry. For production retraining, I use the approved fixed configuration rather than rerunning HPO every time. Git provides configuration and code lineage, Pipeline Parameters provide runtime flexibility, AMT performs HPO, SageMaker orchestrates the workflow, and MLflow provides experiment and model lineage. GitHub Actions, Docker/ECR, and infrastructure as code provide the production CI/CD and deployment layer."

---

# 20. Final Mental Model

Remember the system as seven steps:

```text
1. MANUALLY EXPERIMENT
   "What should I try?"

          ↓

2. LEARN / NARROW
   "What region looks promising?"

          ↓

3. AMT
   "Find the best configuration systematically."

          ↓

4. CONFIRM
   "Does it meet acceptance criteria?"

          ↓

5. REGISTER
   "Store the model and lineage in MLflow."

          ↓

6. PRODUCTION
   "Use the approved configuration."

          ↓

7. RETRAIN
   "Train again on new approved data
    without automatically doing HPO."
```

## One-line rule

> **Use humans to decide what is worth trying, Pipeline Parameters to run deliberate experiments, AMT to search efficiently, Git to version the intent/configuration, SageMaker to orchestrate execution, MLflow to record the evidence and approved model, and CI/CD + containers + IaC to make the system production-grade.**

---

## Reference

Sofian Hamiti — `amazon-sagemaker-github-actions-mlflow`

GitHub:
https://github.com/sofianhamiti/amazon-sagemaker-github-actions-mlflow

Medium:
https://medium.com/data-science/5-simple-steps-to-mlops-with-github-actions-mlflow-and-sagemaker-pipelines-19abf951a70
