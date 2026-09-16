# Production ML Experimentation with Git, SageMaker Pipelines, Pipeline Parameters, AMT, and MLflow

> **Implementation status:** [mlflow-and-reproducibility.md](mlflow-and-reproducibility.md)
> tracks how much of this operating model is wired today and the concrete gaps
> (tracking server, data/code/env lineage, evaluation gate, registry promotion).

## Context

This architecture assumes:
- AWS SageMaker for training and pipeline orchestration
- MLflow for experiment tracking and model registration
- Git for source control and versioned experiment configuration
- SageMaker Automatic Model Tuning (AMT) for systematic hyperparameter optimization

The terminology is intentionally generic so the architecture applies to classification, regression, ranking, recommendation, NLP, computer vision, and other ML problems.

---

## 1. Core mental model

Do not think:

> Production ML means every experiment must go through AMT.

Instead, use three modes:

```text
                    ML DEVELOPMENT
                          |
              +-----------+-----------+
              |           |           |
              v           v           v
           MANUAL       HPO       PRODUCTION
         EXPERIMENTS     |        RETRAINING
              |         AMT           |
              |           |           |
              +-----------+-----------+
                          |
                          v
                  Best configuration
                          |
                          v
                    Final evaluation
                          |
                          v
                    MLflow Registry
                          |
                          v
                     Production
```

**Manual experimentation** tests specific hypotheses.

**HPO** systematically searches a promising space.

**Production retraining** uses an approved configuration and normally does not rerun HPO every time.

---

## 2. What each component does

```text
Git
= What code/configuration did we define and approve?

SageMaker Pipeline Parameters
= What values should this particular execution use?

SageMaker AMT
= Which configuration is best within the defined search space?

MLflow
= What happened in each run, and which model was registered?

S3
= Where are datasets and model artifacts stored?
```

The key principle is:

> Each component should have a clear responsibility rather than becoming a replacement for the others.

---

## 3. Recommended high-level architecture

```text
                         Git
                          |
                code + experiment config
                          |
                          v
                    GitHub Actions
                          |
                          v
                 SageMaker Pipeline
                          |
              +-----------+-----------+
              |                       |
              v                       v
           MANUAL                    HPO
              |                       |
     Pipeline Parameters             AMT
              |                       |
              |                +------+------+
              |                |      |      |
              |               T1     T2     T3 ...
              |                |      |      |
              +----------------+------+------+
                               |
                               v
                         MLflow Tracking
                               |
                               v
                          Evaluation
                               |
                               v
                       MLflow Model Registry
                               |
                               v
                           Approval
                               |
                               v
                          Production
```

---

## 4. Repository structure

```text
ml-project/
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── pipeline.py
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
├── requirements.txt
│
└── .github/
    └── workflows/
        └── ml-pipeline.yml
```

The exact layout is flexible. The important principle is:

> Training code should not need to change just because a hyperparameter changes.

---

# 5. Manual experimentation

Manual experiments are important.

Example:

```yaml
experiment_name: experiment-001

data_version: v10

mode: manual

hyperparameters:
  learning_rate: 0.0005
  batch_size: 64
  dropout: 0.1
```

The execution can be:

```text
Git configuration
       |
       v
CI/CD
       |
       v
SageMaker Pipeline
       |
       | Pipeline Parameters
       | learning_rate = 0.0005
       | batch_size = 64
       | dropout = 0.1
       |
       v
Training Job
       |
       v
MLflow
```

No AMT is required.

Pipeline Parameters let multiple executions of the same pipeline use different values:

```text
Pipeline v7

Run 101 → lr=.001,  batch=64
Run 102 → lr=.0005, batch=64
Run 103 → lr=.0001, batch=128
```

---

# 6. Manual experiments lead into HPO

Suppose manual experiments produce:

```text
lr=.001   → metric=.72
lr=.0005  → metric=.76
lr=.0001  → metric=.71
```

You have learned that the useful region may be around `.0005`.

Now AMT can search a narrower space:

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

This is better than blindly searching an enormous range.

Human-guided experimentation and automated HPO are therefore complementary.

---

# 7. HPO with SageMaker AMT

An HPO experiment configuration might be:

```yaml
experiment_name: experiment-002

data_version: v10

mode: hpo

hpo:
  strategy: bayesian

  max_training_jobs: 30
  max_parallel_jobs: 5

  learning_rate:
    min: 0.0003
    max: 0.002

  batch_size:
    values: [32, 64, 128]

  dropout:
    min: 0.0
    max: 0.2
```

Important distinction:

> Git stores the search-space definition. AMT generates the individual trials.

For example:

```text
Trial 1 → lr=.0007, batch=32, dropout=.05
Trial 2 → lr=.0014, batch=64, dropout=.10
Trial 3 → lr=.0004, batch=128, dropout=.15
...
Trial 17 → lr=.0009, batch=64, dropout=.08  ← best
```

AMT owns the trial generation.

---

# 8. Pipeline Parameters + AMT

Pipeline Parameters can provide execution-level configuration to the pipeline.

Examples:

```text
ExperimentName
DatasetVersion
MaxTrainingJobs
MaxParallelJobs
LearningRateMin
LearningRateMax
```

Conceptually:

```text
Git configuration
       |
       v
GitHub Actions
       |
       | runtime values
       v
SageMaker Pipeline
       |
       v
TuningStep
       |
       v
SageMaker AMT
       |
       +---- Trial 1
       +---- Trial 2
       +---- Trial 3
       +---- ...
       +---- Trial 30
```

Do **not** make every YAML field a Pipeline Parameter automatically.

Pipeline Parameters should expose values that genuinely need runtime variation.

AMT should own the generation of individual trial values.

---

# 9. Manual experiments and AMT are complementary

They are not competing approaches.

```text
                  Human-guided
                 experimentation
                       |
                       v
              Learn the problem
                       |
                       v
              Narrow search space
                       |
                       v
                       AMT
                       |
                       v
               Systematic search
```

Example:

### Stage 1 — Manual

```text
lr=.001   → .72
lr=.0005  → .76
lr=.0001  → .71
```

### Stage 2 — AMT

```text
lr=[.0003, .002]
batch=[32,64,128]
dropout=[0,.2]

30 trials
       ↓
best metric = .80
```

This is often more efficient than immediately giving AMT a huge uninformed search space.

---

# 10. Do not use AMT for everything

AMT is not a replacement for ML engineering judgment.

Some experiments are better expressed as deliberate comparisons:

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

The engineer should decide what is worth tuning.

---

# 11. What happens after AMT finds the best configuration?

Suppose AMT finds:

```text
learning_rate = 0.0009
batch_size = 64
dropout = 0.08

metric = 0.80
```

Do not automatically deploy it.

Use a validation/gating stage:

```text
AMT winner
    |
    v
Final evaluation
    |
    +---- FAIL → reject
    |
    +---- PASS
            |
            v
      MLflow Model Registry
            |
            v
          Approval
            |
            v
        Production
```

Depending on the problem, gates can include:
- offline model metrics
- regression tests
- data-quality checks
- robustness checks
- business KPI checks
- latency/resource checks
- safety/fairness checks where relevant

---

# 12. MLflow's role

MLflow should be the experiment and model lineage layer.

For a manual run, record:

```text
Experiment: experiment-001

Git commit:
abc123

Dataset:
v10

Hyperparameters:
learning_rate = .0005
batch_size = 64
dropout = .1

Metrics:
validation_metric = .76

Artifacts:
model
evaluation results
```

For an AMT trial:

```text
Experiment: experiment-002

Trial: 17

Git commit:
def456

Dataset:
v10

Hyperparameters:
learning_rate = .0009
batch_size = 64
dropout = .08

Metrics:
validation_metric = .80

Artifacts:
model
evaluation results
```

Then register the selected model in **MLflow Model Registry**.

This gives a clean separation:

```text
Git
  ↓
code/config lineage

SageMaker
  ↓
pipeline/job orchestration

AMT
  ↓
HPO trials

MLflow
  ↓
experiment tracking + model registry
```

---

# 13. Production configuration is different from HPO configuration

Once a configuration is selected and approved, create an explicit production configuration:

```yaml
# production/model.yaml

model_version: v27

data_version: v10

hyperparameters:
  learning_rate: 0.0009
  batch_size: 64
  dropout: 0.08
```

Production retraining then becomes:

```text
production config
       |
       v
SageMaker Pipeline
       |
       | Pipeline Parameters
       v
Training
       |
       v
Evaluation
       |
       v
MLflow
       |
       v
MLflow Model Registry
```

There is normally no reason to run AMT every time the model is retrained.

---

# 14. Complete lifecycle

```text
                 ┌─────────────────────┐
                 │       Git           │
                 │                     │
                 │ Code + Config       │
                 └──────────┬──────────┘
                            |
                            v
                     GitHub Actions
                            |
                            v
                 SageMaker Pipeline
                            |
                +-----------+-----------+
                |                       |
                v                       v
             MANUAL                    HPO
          experiment                experiment
                |                       |
         Pipeline Params              AMT
                |                       |
                |                multiple trials
                |                       |
                +-----------+-----------+
                            |
                            v
                       MLflow
                            |
                            v
                       Evaluation
                            |
                            v
                 MLflow Model Registry
                            |
                         Approval
                            |
                            v
                       Production
                            |
                            v
                    Periodic retraining
                            |
                            v
                  Approved configuration
```

---

# 15. What should be versioned?

For reproducibility, capture:

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

A model should be traceable to these inputs.

Example:

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
```

---

# 16. Best practices

## 16.1 Keep training code configuration-driven

Prefer:

```python
train(
    learning_rate=config["learning_rate"],
    batch_size=config["batch_size"],
)
```

rather than hard-coded values.

## 16.2 Version experiment configurations in Git

Every meaningful experiment should be traceable to a Git commit.

## 16.3 Use Pipeline Parameters for runtime overrides

Use them when a value genuinely varies between executions.

Examples:

```text
dataset version
experiment name
selected hyperparameters
HPO limits
search-space boundaries
```

## 16.4 Keep the AMT search space intentional

Use:
- domain knowledge
- baseline experiments
- previous experiments
- literature
- prior production models

to construct a reasonable search space.

## 16.5 Track important runs in MLflow

Record at least:

```text
parameters
metrics
artifacts
code version
data version
environment
experiment name
```

## 16.6 Keep HPO separate from production retraining

HPO asks:

> What configuration should we use?

Production retraining asks:

> Train the approved configuration on the latest approved data.

These are different workflows.

## 16.7 Keep production configuration explicit

Once a configuration is approved, record it explicitly so production can be reproduced.

## 16.8 Use immutable/versioned datasets and artifacts

Avoid relying on moving paths such as:

```text
s3://bucket/latest/
```

for reproducibility. Prefer versioned data/artifacts.

---

# 17. Things to avoid

### Hard-code hyperparameters in training code

```python
learning_rate = 0.0009
```

inside the training implementation.

### Run AMT for every training run

HPO is expensive and unnecessary when an approved configuration already exists.

### Give AMT an unnecessarily huge search space

A search space should be informed by experimentation and domain knowledge.

### Make every YAML field a Pipeline Parameter

Only expose values that genuinely need runtime variation.

### Treat Pipeline Parameters as experiment tracking

Pipeline Parameters are execution inputs, not a replacement for MLflow.

### Store secrets in configuration or Pipeline Parameters

Use an appropriate secrets-management mechanism.

### Automatically promote the AMT winner

Use evaluation and approval gates.

### Change code for every hyperparameter experiment

The training implementation should remain stable while configuration changes.

---

# 18. Recommended separation of responsibilities

| Component | Responsibility |
|---|---|
| **Git** | Code + versioned experiment/config definitions |
| **GitHub Actions** | CI/CD and triggering pipeline executions |
| **SageMaker Pipeline** | End-to-end workflow orchestration |
| **Pipeline Parameters** | Runtime inputs/overrides |
| **SageMaker AMT** | Systematic hyperparameter search |
| **MLflow Tracking** | Parameters, metrics, artifacts, experiment history |
| **MLflow Model Registry** | Model versions and promotion state |
| **S3** | Data and model artifacts |
| **Production pipeline** | Reproduce/train/evaluate approved configuration |

---

# 19. Interview-ready architecture

```text
                     GIT
                      |
               code + config
                      |
                      v
                 CI/CD
                      |
                      v
            SAGEMAKER PIPELINE
                      |
            +---------+---------+
            |                   |
         MANUAL                HPO
            |                   |
    Pipeline Parameters         AMT
            |                   |
            |            multiple trials
            |                   |
            +---------+---------+
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
```

A strong interview answer:

> "I separate experimentation from production retraining. During development, I can manually test specific hyperparameter values by passing them as SageMaker Pipeline Parameters, without changing the training code. Once I understand the promising region, I can define a search space in a Git-versioned experiment configuration and use SageMaker AMT to systematically search it. I track manual runs and AMT trials in MLflow, evaluate the selected configuration, and register the approved model in MLflow Model Registry. For production retraining, I use the approved fixed configuration rather than rerunning HPO every time. Git gives me configuration and code lineage, Pipeline Parameters provide runtime flexibility, AMT performs HPO, and MLflow provides experiment and model lineage."

---

# 20. Final mental model

```text
1. MANUALLY EXPERIMENT
   "What should I try?"

             ↓

2. LEARN / NARROW
   "What region looks promising?"

             ↓

3. AMT
   "Find the best configuration
    systematically."

             ↓

4. CONFIRM
   "Does the selected configuration
    meet our acceptance criteria?"

             ↓

5. REGISTER
   "Store the model in MLflow."

             ↓

6. PRODUCTION
   "Use the approved configuration."

             ↓

7. RETRAIN
   "Train again with new approved data,
    without automatically doing HPO."
```

## One-line rule

> **Use humans to guide what is worth trying, AMT to search efficiently, Pipeline Parameters to inject runtime values, Git to version the intent/configuration, and MLflow to record the evidence and approved model.**
