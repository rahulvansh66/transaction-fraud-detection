---
name: ml-engineer
description: Generic senior-ML-engineer philosophy for building scalable, reproducible, production-grade ML pipelines (training or fine-tuning, any domain). Covers the preprocessing philosophy (prototype in pandas/sklearn on a sample, then port to PySpark for the Processing job) and how to modularise pipeline code. Use whenever designing or writing a preprocessing / feature-engineering step, deciding pandas vs. PySpark, splitting pipeline code into modules, or moving a notebook prototype into a scalable job. Also use when asked "how should I structure this step" for any new ML project.
---

# ML engineer philosophy

Project-agnostic principles. Project-specific rules live in the other skills
(`ml-repo-structure`, `ml-lineage-reproducibility`, `ml-experimentation-workflow`);
this one covers the mindset that applies to any training or fine-tuning project.

## Core stance

- Build for the production data size (assume it will be 10-100x the dev sample), but
  develop on a small sample so end-to-end iteration stays fast.
- Every run must be reproducible: pinned dependencies, seeded splits, config-driven
  values, immutable data snapshots.
- Infrastructure is code. Nothing is created by hand.

## Preprocessing philosophy: prototype small, then scale

Preprocessing is done in two deliberate phases. Do not skip phase 1, and do not
ship phase 1 as the production job.

### Phase 1. Prototype in pandas/sklearn on a small sample

Work locally in a notebook on a small, representative sample, where iteration is fast:

1. Explore the data: schema, types, nulls, cardinality, class balance, leakage risks.
2. Apply the preprocessing steps: cleaning, imputation, encoding, scaling, and so on.
3. Engineer features.
4. Split train/validation/test (time-aware if the data is temporal; stratify if the
   target is imbalanced; fix the seed).
5. Record the *decisions* (which columns dropped, which encodings, which thresholds)
   in the notebook markdown and move the tunable values into config.

Exit criterion: the logic is finalized, i.e. you can state it as a deterministic
sequence of transformations, and the notebook output is the reference for phase 2.

### Phase 2. Port to PySpark for the real Processing job

Port the *finalized* logic to PySpark so it runs as a SageMaker/EMR/Glue Processing
job on the full dataset:

- Reimplement transformations with Spark DataFrame APIs. Avoid `toPandas()`, Python
  UDFs, and anything that collects data to the driver.
- Fit statistics (means, category vocabularies, scaler params) on the train split only,
  persist them as artifacts, and reuse them at inference time to avoid train/serve skew.
- Write partitioned, columnar output (Parquet) to an immutable, versioned S3 path.
- **Parity-check** the port: run both implementations on the same sample and assert the
  outputs match (row counts, schema, per-feature summary stats, or exact values within
  tolerance). The pandas prototype is the test oracle.

### Rules of thumb

- Only prototype in pandas what is going to be ported; do not add a pandas-only trick
  that has no Spark equivalent.
- Any change to feature logic goes through the loop again: change the prototype, re-run
  the parity check, then update the Spark job.
- If the data fits comfortably in memory and will always do so, say so explicitly in the
  docs and skip phase 2. Otherwise default to porting.

## Modularising pipeline code

Split code by why it changes and where it runs. Dependencies point one way.

| Layer | Responsibility | Notes |
|---|---|---|
| Pure logic (`features.py`) | DataFrame in, DataFrame out | No I/O or cloud SDKs. Unit-testable and shared between notebook and job. |
| Job entry point (`spark_job.py`) | Parse args, read input, call logic, write output | The only place that touches paths and configures logging; logs run IDs and the git commit. |
| Launcher (`scripts/run_*_job.py`) | Builds and submits the Processing/Training job | Runs locally or in CI, not in the cluster. Changes when infrastructure changes. |
| Config (`config/**.yaml`) | Column lists, thresholds, paths, instance sizes | No hardcoded hyperparameters or environment values in code. |
| Infra (`infrastructure/*.tf`) | Buckets, roles, log groups | Terraform only. |

The same layering applies to training, evaluation, and inference steps.

## Related skills: load these when relevant

This skill is the entry point. Pull in the others as the task requires:

- **`ml-repo-structure`**: when deciding where a file belongs (`src/`, `config/`,
  `docker/`, `infrastructure/`, CI), adding a new module or config file, or reviewing
  for hardcoded hyperparameters and environment values leaking into code or configs.
- **`ml-lineage-reproducibility`**: when touching `train.py`, `pipeline.py`, or any
  processing/evaluation step or MLflow logging, to check what lineage to record
  (code, config, data snapshot, hyperparameters) and which S3 paths must be immutable.
- **`ml-experimentation-workflow`**: when writing experiment configs, SageMaker
  pipeline/AMT tuning steps, or GitHub Actions that trigger pipelines, and when deciding
  between a manual run, an HPO search, or production retraining.

## Checklist before calling a step done

- [ ] Prototype notebook documents the decisions, and the values are in config.
- [ ] Spark (or other scalable) job implements the same logic, with a parity test.
- [ ] Pure logic is separate from I/O and has unit tests.
- [ ] Split is seeded, leakage-free, and reproducible.
- [ ] Fitted preprocessing artifacts are saved and versioned for inference.
- [ ] Logs are structured (key=value), carry run IDs, and contain no PII or secrets.
- [ ] Output data path is versioned and immutable; lineage is recorded.
- [ ] Files are placed per `ml-repo-structure`; lineage per `ml-lineage-reproducibility`.
