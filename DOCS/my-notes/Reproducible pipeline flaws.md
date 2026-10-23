what are best practices followed to make ml pipeline reproducable, and check if any practice is missing? and if missing can we cange code, withouth affecting current lineage?

Short answer: most of the checklist is already covered. Six things are missing or weak, and none of them need to change how run IDs or config hashes are computed. I read the code but haven't run it or changed anything. I did not review the GitHub workflows beyond seeing that they run `uv sync`.

## Already in place

| Practice | Where |
|---|---|
| Immutable processed data: `processed/<run_id>/`, run ID built from git SHA, config hash and data version | [lineage.py](src/preprocessing/lineage.py), [run_preprocessing_job.py](src/preprocessing/run_preprocessing_job.py) |
| Overwrite protection: S3 `IfNoneMatch="*"` and Spark `errorifexists` | [run_preprocessing_job.py:117](src/preprocessing/run_preprocessing_job.py#L117), [spark_job.py:200](src/preprocessing/spark_job.py#L200) |
| `latest` and empty data IDs rejected | [sagemaker_jobs.py:39](src/training/sagemaker_jobs.py#L39) |
| Dirty-tree refusal on SageMaker, plus a dirty-diff hash in the SHA | [lineage.py](src/training/lineage.py) |
| Deterministic source tarball with a content-addressed S3 key | [sagemaker_jobs.py:72](src/training/sagemaker_jobs.py#L72) |
| Seed set and logged; XGBoost version fail-fast | [model.py](src/training/model.py), [train.py:90](src/training/train.py#L90) |
| Full booster params, `pip freeze`, resolved config, features, model signature and input example logged | [train.py:118](src/training/train.py#L118) |
| Train, val and test metrics, validation-chosen threshold, test set single-use, gate | [evaluate.py](src/evaluation/evaluate.py) |
| Run tags (`mode`, SHA, config hash, data run ID), HPO trials as child runs | [mlflow_tracking.py](src/mlflow_tracking/mlflow_tracking.py), [hpo_tuner.py](src/training/hpo_tuner.py) |

## Gaps

| # | Gap | Why it matters |
|---|---|---|
| 1 | The training container deps are unpinned. [requirements.txt](src/training/requirements.txt) has `mlflow>=2.14`, and the CI `uv sync` isn't `--locked`. | The same code can pull a different MLflow next month. |
| 2 | The image is recorded as a tag (`1.7-1`), not a digest. | The tag can be re-pushed. |
| 3 | Training logs only a 1000-row sample of val as the MLflow input ([train.py:167](src/training/train.py#L167)). There is no file manifest or checksum for the raw input, and `metadata.json` isn't attached to the run. | The data fingerprint is weak. |
| 4 | Raw and processed data aren't protected against overwrite at the bucket level. [s3.tf](infrastructure/s3.tf) only references the bucket, which a legacy Terraform project owns, and it grants write on `processed/*` and `models/*`. | Nothing stops an in-place overwrite of a path we call immutable. |
| 5 | The source-bundle S3 URI, the full git SHA (the SHA is short) and the env name aren't tagged on the run. | You can't jump from a run to the exact code bundle. |
| 6 | There is no determinism test, such as training twice with the same seed and asserting identical metrics. | Nothing catches reproducibility regressions. |

**Not a gap yet:** `PipelineExecutionArn` is always empty because there is no SageMaker Pipeline step. That belongs to the pipeline stage.

**Possible bug, not a gap:** [evaluate.py:126](src/evaluation/evaluate.py#L126) calls `predict` without `iteration_range`, but training picked the threshold using the best iteration. Test scores may use all trees. I haven't verified how XGBoost 1.7.4 behaves here, so check it before trusting test metrics.

**Docs:** [mlflow-and-reproducibility.md](DOCS/my-notes/mlflow-and-reproducibility.md) is stale. It describes 5 params and 1 metric, and it points at a `src/model_build` path that no longer exists. The lineage skill also links to `DOCS/ok/`, which doesn't exist; the files are in `DOCS/my-notes/`.

## Can we fix these without affecting lineage?

Lineage identity comes from two things:

- `config_hash(cfg)`, which hashes the experiment YAML or the preprocessing YAML.
- The run ID, which is `<sha>-<config_hash>-<data_version>`.

**Safe.** These only add information and leave every existing hash and `processed/<run_id>` path untouched:

- Gap 5 (extra tags), gap 3 (log `metadata.json`, a data manifest and a full-data digest), gap 6 (the test), and the docs refresh.
- Gap 4: this is Terraform-only, and needs to go into the legacy bucket's project. It adds versioning or a deny-overwrite policy and changes no existing data.
- Gap 2: resolve and tag the image digest as a best-effort lookup. I'd need to check that the role can read the AWS-owned ECR repo before relying on it.
- Gap 1: pin the deps and run CI with `--locked`.
  - Pin to the versions in an earlier run's `pip_freeze.txt` artifact, so behaviour doesn't shift.
  - Editing `requirements.txt` changes the source-tarball hash, which gives a new `models/code/<hash>/` key. Old keys stay valid and no run ID changes.

**Would change IDs.** Avoid these unless you want a clean break:

- Editing `preprocessing.yaml` (even adding a key) or `config_hash()` / `build_run_id()`. That produces a new `processed/<run_id>` and the old data won't dedupe against it.
- Adding fields to an experiment YAML changes its training `config_hash`. Old runs keep their old hash, but new runs won't match them.
- Changing which keys feed the hash. Add new lineage as separate tags instead.

Do you want me to implement the safe set, 1 to 3 and 5 to 6? Gap 4 needs a separate Terraform change in the legacy project.