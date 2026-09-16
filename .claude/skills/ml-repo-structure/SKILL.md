---
name: ml-repo-structure
description: Defines where code, configs, docker, infra, and CI files belong in this ML repo, and the rule that training code stays config-driven with no hardcoded hyperparameters. Use whenever scaffolding the repo, adding a new src/ module or configs/ file, deciding where a new experiment/production/environment config or workflow file should live, or reviewing whether a value should be a hyperparameter in code vs. a config field. Also use if code review turns up a hardcoded hyperparameter, a dev/prod environment value (bucket name, tracking URI, role ARN) leaking into an experiment or production model config, or a YAML file mixing unrelated concerns.
---

# ML repo structure

Source: [DOCS/ok/production_ml_experimentation_git_sagemaker_amt_mlflow.md](../../../DOCS/ok/production_ml_experimentation_git_sagemaker_amt_mlflow.md)
(§4, §16.1) and [DOCS/ok/best of his repo and our hpo approach.md](../../../DOCS/ok/best%20of%20his%20repo%20and%20our%20hpo%20approach.md)
(§13–15). Read those for the full architecture; this skill just enforces the
structural rules day to day.

## Target layout

```
ml-project/
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── pipeline.py
│   └── inference.py
├── configs/
│   ├── experiments/        # one file per manual or HPO experiment
│   │   └── experiment-NNN.yaml
│   ├── production/
│   │   └── model.yaml      # the single approved, versioned config
│   └── env/                # per-AWS-environment values, not model config
│       ├── dev.yaml
│       └── prod.yaml
├── tests/
├── docker/
│   └── Dockerfile
├── infrastructure/         # IaC (CDK/Terraform/etc.)
├── requirements.txt
└── .github/
    └── workflows/
        ├── ci.yml
        ├── build.yml
        └── ml-pipeline.yml
```

The exact folder names are flexible — the two things that aren't are: `src/`
never contains a hyperparameter value, and `configs/experiments/` vs
`configs/production/` are never merged into one file or one folder. Those two
splits are what let the rest of the operating model ([[ml-experimentation-workflow]],
[[ml-lineage-reproducibility]]) work.

## Rule 1: training code is config-driven

> Training code should not need to change just because a hyperparameter changes.

Prefer:

```python
train(learning_rate=config["learning_rate"], batch_size=config["batch_size"])
```

Never:

```python
learning_rate = 0.0009  # inside train.py
```

If you're about to add a new tunable value, ask "does this belong in a config
file, or is it truly structural?" — almost everything belongs in config. This
is what lets the same `src/` support manual experiments, AMT trials, and
production retraining without a code change between them.

## Rule 2: one config file, one responsibility

Don't put everything into one YAML. Keep these separate:

| Config | Purpose | Lives in |
| --- | --- | --- |
| Experiment config | One manual run or one HPO search-space definition | `configs/experiments/*.yaml` |
| Production config | The single approved, versioned config actually deployed | `configs/production/model.yaml` |
| Environment config | Per-AWS-environment values (bucket names, tracking URIs, role ARNs) — never model hyperparameters | `configs/env/{dev,prod}.yaml` |
| Pipeline parameters | Runtime overrides passed into a SageMaker Pipeline execution | not a file — passed at execution time (see [[ml-experimentation-workflow]]) |
| Docker config | Training/runtime environment | `docker/Dockerfile` |
| Infra config | AWS resources (SageMaker, S3, ECR, IAM, networking, endpoints) | `infrastructure/` |
| Deployment config | Serving infra, separate from experimentation config | wherever inference/serving lives, not `configs/experiments/` |

If a new file would mix two of these rows, split it instead. This separation
is what makes each component auditable on its own — you can tell whether
something changed because someone tried a new hyperparameter, approved a new
production config, or reconfigured infrastructure, without diffing one giant
file.

## Two different axes: experiment-vs-production, and dev-vs-prod environment

Don't conflate these — they answer different questions and belong in
different files:

- **`configs/experiments/` vs `configs/production/`** answers "is this
  model configuration a candidate I'm trying, or the one we've approved to
  run?" It changes only when a new config is proposed or approved.
- **`configs/env/{dev,prod}.yaml`** answers "which AWS account/bucket/
  tracking server should this execution talk to?" It has nothing to do with
  model hyperparameters — the same `configs/production/model.yaml` should
  produce the same model whether it's executed against the dev or prod
  environment config.

Concretely: an HPO run in the dev AWS account still reads
`configs/experiments/experiment-014.yaml` + `configs/env/dev.yaml`. A
scheduled prod retrain reads `configs/production/model.yaml` +
`configs/env/prod.yaml`. Never bake environment values (bucket names,
tracking URIs, role ARNs) into an experiment or production config, and never
let an environment file carry a hyperparameter — that's the same
mixed-responsibility mistake Rule 2 warns about, just on the other axis.

## Where to look next

- Deciding manual vs. HPO vs. production for a new experiment → [[ml-experimentation-workflow]]
- Wiring MLflow logging / lineage for a new pipeline step → [[ml-lineage-reproducibility]]
