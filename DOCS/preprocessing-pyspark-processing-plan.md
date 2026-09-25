# Scalable Preprocessing: SageMaker PySpark Processing Job

Plan for step 3 of [plan-v1-modified.md](plan-v1-modified.md): port the pandas prototype in
[01_preprocessing_feature_engineering.ipynb](../notebooks/01_preprocessing_feature_engineering.ipynb)
to PySpark and run it as a SageMaker Spark Processing job that writes to `processed/` in S3.

Related: [feature-engineering.md](feature-engineering.md), [dataset-info.md](dataset-info.md).

---

## 1. Goal and scope

**In scope**
- One PySpark job that reproduces the notebook logic: load → clean → past-only account counts → type filter → features → leakage gate → time-based split → class weight → persist.
- Runs as a SageMaker Processing job, launched with the SageMaker Python SDK v3 `PySparkProcessor` (`sagemaker.core.spark.processing`).
- Built and verified on `data-v0` only (local parity plus one SageMaker smoke run). The code is written so the full 15GB+ dataset needs only config changes (instance type and count), but running at that scale is a **future step** (section 12).
- Output is an immutable, versioned prefix under `processed/`.

**Out of scope (later steps)**
- Running and tuning on the full 15GB+ dataset (future step, section 12).
- Training (step 4), registry (step 5), serving (step 6), the `ProcessingStep` wrapper inside a SageMaker Pipeline (step 5 will reuse this job unchanged).
- Glue/scheduled account-aggregate jobs described in `my-notes/preprocessing-glue-and-sagemaker.md`. The v1 features are computed inside this one job.

**Principle:** the notebook is the spec. Every rule is already decided there, so the port must not invent new feature logic. It only changes the execution engine.

---

## 2. Architecture

```text
s3://<data_bucket>/raw/data-v0/day=*/...        (Hive-partitioned parquet)
        │  read directly via s3:// URI (--input-uri); config via ProcessingInput
        ▼
┌──────────────────────────────────────────────────────────┐
│ SageMaker Processing job (PySparkProcessor, Spark 3.5)    │
│  spark_job.py (entry) -> features.py (pure functions)     │
│  load -> clean -> past counts -> filter -> features       │
│       -> leakage gate -> time split -> scale_pos_weight   │
└──────────────────────────────────────────────────────────┘
        │  written directly via s3:// URI (--output-uri)
        ▼
s3://<data_bucket>/processed/<run_id>/train/part-*.parquet
                                     /val/part-*.parquet
                                     /test/part-*.parquet
                                     /metadata.json
```

- **Launcher** (`src/preprocessing/run_sagemaker_preprocessing_job.py`, runs on a laptop / CI / later the pipeline): builds the `PySparkProcessor` and calls `run()`. It reads the role ARN and bucket from `config/env/dev.yaml`.
- **Entry script** (`src/preprocessing/spark_job.py`): arg parsing, `logging.basicConfig`, SparkSession creation, orchestration. This is the only place that configures logging.
- **Feature logic** (`src/preprocessing/features.py`): pure `DataFrame -> DataFrame` functions, testable with a local SparkSession.
- **Data I/O via `s3://` URIs** passed as arguments (`--input-uri`, `--output-uri`). `ProcessingInput` copies data to each node's local disk, which breaks multi-node Spark, so it carries only the small config YAML (uploaded to S3 by the launcher first). The input URI is recorded in `metadata.json` for lineage.
- **SDK: SageMaker Python SDK v3 throughout.** `sagemaker-core` (`sagemaker.core.spark.processing.PySparkProcessor`, `sagemaker.core.shapes.ProcessingInput`/`ProcessingS3Input`) and `sagemaker-mlops` (`sagemaker.mlops.workflow` for Pipelines). Do not use v2 paths (`sagemaker.spark.processing`, `sagemaker.workflow`) or v2 idioms (`ProcessingInput(source=, destination=)`). Check field names in the installed package, since most online examples are v2.
- **Image:** the AWS-managed SageMaker Spark container (no custom ECR image, nothing to maintain). Pin `framework_version` (e.g. `3.5`) for reproducibility.

---

## 3. Notebook to PySpark porting map

| Notebook function | PySpark equivalent | Parity risk |
|---|---|---|
| `load_raw` | `spark.read.option("basePath", ...).parquet(path)`; select `RAW_COLUMNS`; raise `ValueError` if any column is missing; drop the `day` partition column | Types: enforce the notebook dtypes (`int` for `step`, `double` for amounts, `string` for names) with explicit casts |
| `clean` | `dropna(subset=RAW_COLUMNS)` then `dropDuplicates()`; log null and zero-amount counts as aggregates | Low. `dropDuplicates` keeps an arbitrary copy of each duplicate group, which is harmless because the rows are identical |
| `add_past_counts` | `F.count("*").over(Window.partitionBy("nameOrig").orderBy("step").rangeBetween(unboundedPreceding, currentRow))`, and the same for `nameDest` | Resolved, see 4.1 (order-free `RANGE` frame). Must run before the type filter |
| `filter_types` | `df.filter(F.col("type").isin(keep_types))`, with fraud counts before and after logged | Low |
| `add_basic_features` | `is_transfer = (type == "TRANSFER").cast("byte")`, `log_amount = F.log1p("amount")`, `hour_of_day = step % 24`, `is_night = hour_of_day.between(a, b)`, `day_of_month = floor(step / 24)` | `step // 24` in pandas is floor division; use `F.floor` on a double or `(step - step % 24) / 24` cast to int. Non-negative `step`, so no sign edge case |
| `apply_leakage_gate` | Same `BASE_FEATURES` / `ERROR_FEATURES` lists; `select("step", *features, label)`; log a `WARNING` when error features are on | Low. Move the constants into `features.py` so notebook and job share them |
| `time_split` | One `agg(F.max("step"))`, `t1 = floor(train_frac * max)`, `t2 = floor(val_frac * max)`, then three `filter`s | `np.floor` vs Python `math.floor`: same result. Compute `max` once on the driver so all splits use identical thresholds |
| `compute_scale_pos_weight` | One `agg(F.sum(label), F.count("*"))` on train only; `None` when there are no positives | Low |
| `persist` | `df.write.mode("errorifexists").parquet(.../<split>)` per split; `metadata.json` written by the driver | Spark names files `part-00000-<uuid>.parquet`. Consumers read the folder, never a filename (already true in the notebook) |
| `validate_outputs` | Re-read the written splits and assert columns, no nulls, no leaky columns and ordered step ranges | Runs at the end of the job, so a bad output fails the job |
| `find_repo_root` / `load_config` | Config path comes from `--config`; `yaml.safe_load` | None. `find_repo_root` is notebook-only |

---

## 4. Key design decisions and risks

### 4.1 Order-free account counts (resolved)
`step` is one simulated hour and the only time axis, so rows in the same step are simultaneous. The original notebook broke ties by file position, which Spark cannot reproduce (`monotonically_increasing_id()` depends on how files are split), and file order within a step is not guaranteed to be chronological.

**Decision:** count every row with `step` earlier than or equal to the current row's, using a `RANGE` frame on `step` (`orderBy("step").rangeBetween(unboundedPreceding, currentRow)`). Tied rows share one count and no order between them is invented. An alphabetical tie-break on `type` was tried and rejected: it reverses the real TRANSFER-then-CASH_OUT pattern and biases counts by type. This is also exactly the notebook's stated definition ("earlier-or-equal `step`").

**Consequence:** a row's count includes other transactions of the same account in the same hour, some of which may be later in real time. At hour resolution this is a small effect. It is deterministic and identical in pandas and Spark, so a row-for-row parity test is possible (and passes).

**Alternative if ever needed:** carry a `row_id` (CSV line number) from `csv_to_hive_parquet.py` and order by `(step, row_id)`. Only worth it if the full file's within-step order is shown to be chronological.

### 4.1a Account IDs and train/serve consistency
Raw `nameOrig` / `nameDest` never reach the model (the leakage gate keeps only `step`, the 8 features and the label). Account IDs do influence the model through `orig_txn_count` / `dest_txn_count`, which are computed from them before they are dropped. Those two counts are the main train/serve skew risk: at serving time they must be computed over the account's full prior history (all transaction types, all earlier steps, current transaction included), or the model sees a distribution it wasn't trained on. The rules are recorded in [config/inference/feature_contract.yaml](../config/inference/feature_contract.yaml) for the inference step to use; keep it in sync with `features.py`.

### 4.2 Skew and window cost at 15GB+ (future step; design note only)
- Each account count is a window over a full shuffle of the dataset. Mule `nameDest` accounts receive many transactions, so a few partitions can be very large. Enable AQE and skew handling (`spark.sql.adaptive.enabled`, `spark.sql.adaptive.skewJoin.enabled`) and set `spark.sql.shuffle.partitions` from data size (about 128-200MB per partition).
- A window's rows for one key must sit on one executor. If a single key ever exceeds executor memory, the fallback is an incremental approach (per-key `step`-bucketed counts + cumulative sum of buckets), documented here but not built until the scale test shows it is needed.
- The two windows (`nameOrig`, `nameDest`) are two shuffles. Keep them in one job and select only the columns needed before shuffling (drop unused columns first) to cut shuffle bytes.

### 4.3 Reproducibility (per the ml-lineage-reproducibility skill)
- **Immutable output prefix:** `processed/<run_id>/`, where `run_id = <git_sha>-<config_hash>` (or the processing job name). The job writes with `errorifexists`, so an existing run is never overwritten.
- **Pinned everything:** Spark framework version, SageMaker SDK v3 versions (`sagemaker-core`, `sagemaker-mlops`, via `uv.lock`), `seed` from config (there is no random op today, but it is passed through for future sampling).
- **Deterministic output layout:** fixed `repartition` per split (by count or target size), so re-runs produce comparable file layouts. Row *content* is deterministic; part file names contain a UUID and are not.
- **Input versioning:** record the input S3 prefix, the object count and total bytes in `metadata.json`. For prod, land raw data under a dated or versioned prefix, so the snapshot cannot change under a run.

### 4.4 No train-only fitting issue
Past-only counts use only earlier rows, so they are computed on the full data before the split without leakage. The only fitted quantity is `scale_pos_weight`, computed on train only. Nothing is fitted on val or test.

### 4.5 Output file size
Spark writes one file per task. Repartition each split before writing to target 128-256MB files (few files for the tiny `data-v0` sample; hundreds for the full run). Avoid `coalesce(1)` on train at scale.

### 4.6 Config delivery
- `config/preprocessing/preprocessing.yaml` is uploaded to S3 by the launcher, mounted via a v3 `ProcessingInput(s3_input=ProcessingS3Input(...))` and passed as `--config`. It is the same file the notebook uses, so notebook and job cannot drift.
- Environment values (bucket, role ARN, region) come from `config/env/dev.yaml` in the launcher only. The Spark job itself never sees account-specific values, and only gets S3 locations through its input/output mounts.
- The config snapshot is written into `metadata.json`.

---

## 5. Code layout

Follows the `ml-repo-structure` skill and the docstring / typing / logging rules in `CLAUDE.md`.

```text
src/preprocessing/
    features.py          # pure DataFrame -> DataFrame functions + feature constants
    spark_job.py         # entry point: args, logging, SparkSession, orchestration, metadata
scripts/
    run_sagemaker_preprocessing_job.py   # launcher: builds PySparkProcessor, calls run()
tests/preprocessing/
    test_features.py     # local[2] SparkSession unit tests per function
    test_parity.py       # Spark output vs notebook/pandas output on data-v0
config/preprocessing/preprocessing.yaml   # reused as-is; add processing-job sizing keys (see below)
```

- Every module, class and function gets a Google-style docstring with all args documented, and type hints on every signature.
- `getLogger(__name__)` per module; only `spark_job.main` calls `basicConfig`.
- **Dependencies (`pyproject.toml`, via uv):** `pyspark` (dev, for local tests), `sagemaker-core` (launcher) and `sagemaker-mlops` (Pipelines, step 5), `pytest`. The Spark container already ships PySpark; only `pyyaml` may need to be shipped (the container includes it, to be verified in the first run).
- **Shipping `features.py`:** pass `submit_py_files=["src/preprocessing/features.py"]` (or a zipped `src/preprocessing`) to `PySparkProcessor.run`, so the entry script can import it.
- **Config additions** (`preprocessing.yaml`, a `processing:` block, not hyperparameters): `instance_type`, `instance_count`, `volume_size_gb`, `max_runtime_s`, `spark_conf` (shuffle partitions, AQE flags), and `target_file_mb`. Dev and prod override these through `config/env/`.

---

## 6. Cluster sizing and Spark configuration

| Run | Data | Instances | Notes |
|---|---|---|---|
| Local dev | `data-v0` | none (`local[*]`) | Unit and parity tests |
| SageMaker smoke | `data-v0` in S3 | 1 x `ml.m5.xlarge` | Proves IAM, S3 paths, packaging and logging. About minutes of runtime, mostly container start-up |
| Full scale (**future step**, not in current scope) | 15GB+ | start with 3-4 x `ml.m5.4xlarge`, then tune from CloudWatch metrics | Record runtime, shuffle spill and skew; adjust instance count and shuffle partitions |

Settings to define explicitly: `spark.sql.shuffle.partitions`, AQE on, executor / driver memory (`configuration` argument of `run()`), `max_runtime_in_seconds` as a cost guard, and `volume_size_in_gb` large enough for shuffle spill.

---

## 7. Terraform work items

Terraform is the only way to change AWS (see `CLAUDE.md`). Nothing below is done by hand.

1. **S3 permissions** (`infrastructure/s3.tf`, `sagemaker_data_access` policy): Spark's output committer writes to a temporary path and renames (copy + delete), which needs `s3:DeleteObject` on `processed/*`. The policy currently grants only `GetObject`/`PutObject` there. Add `s3:DeleteObject` and `s3:AbortMultipartUpload` and `s3:ListBucketMultipartUploads` (bucket-level) for large writes.
2. **CloudWatch log group:** an explicit `aws_cloudwatch_log_group` for `/aws/sagemaker/ProcessingJobs` with `retention_in_days` (e.g. 14 for dev, longer for prod). This prevents the default "never expire".
3. **Role trust:** confirm `fraud-detection-dev-sagemaker-execution` trusts `sagemaker.amazonaws.com` for processing jobs (it already serves the Studio domain). It also needs `logs:*` write and `ecr:GetAuthorizationToken` / pull for the AWS-managed Spark image; verify against the role's current policies and add only what is missing.
4. **Optional:** an S3 prefix for the Spark event logs/history (`spark-history/`), for debugging skew at scale.
5. **Networking:** the job needs no internet (no DagsHub call in preprocessing), so it can later run in a VPC without NAT. Keep this in mind when the VPC hardening (v-later) lands.

---

## 8. Logging and lineage

Log lines are `key=value` and lazy-formatted so CloudWatch Logs Insights can query them, e.g. `step=split part=train rows=%d fraud=%d`.

At job start, log: processing job name (`TrainingEnv` / `/opt/ml/config/processingjobconfig.json`), git commit (passed as a job argument or environment variable by the launcher), input URI, output URI, config hash and Spark version.

Only aggregates are logged: counts, fractions and thresholds. There are no rows, account IDs or DataFrame dumps (PII / volume rule in `CLAUDE.md`).

`metadata.json` (same structure as the notebook, extended for lineage):

- `run_id`, `processing_job_name`, `git_commit`, `spark_version`, `sagemaker_sdk_version`
- `source_path` (S3 URI), input object count / bytes
- `features`, `label`, `thresholds`, `scale_pos_weight`
- `row_counts`, `fraud_counts` per split
- `config` snapshot and `config_hash`

Step 4 (training) will read this file and log it to MLflow (dataset URI, `run_id`, `git_commit`), so a model can be traced back to the exact data snapshot.

---

## 9. Validation plan

1. **Unit tests** (`test_features.py`): each function on a tiny hand-built DataFrame, with edge cases: duplicate rows, ties at the same `step`, an account appearing in non-kept types, no positives in train, and invalid split fractions.
2. **Parity test** (`test_parity.py`): run the Spark job locally (`local[*]`) on `dataset/raw/data-v0` and compare with the notebook output (features, split membership, `scale_pos_weight`, thresholds). Exact match is expected because both use order-free counts (4.1).
3. **Output contract:** the ported `validate_outputs` runs inside the job.
4. **SageMaker smoke run** on `data-v0` in S3 (1 instance): check the CloudWatch logs, the `processed/<run_id>/` layout and `metadata.json`.
5. **Scale test** (future step, section 12): not run now.

---

## 10. Milestones

- [x] Order-free account counts in notebook and Spark (4.1), notebook re-run
- [ ] `features.py`: port each function, with docstrings and types
- [ ] `spark_job.py`: entry point, config, metadata, validation
- [ ] Local unit and parity tests green
- [ ] Terraform: S3 policy fix and log group with retention (section 7); `terraform plan` reviewed before apply
- [ ] `run_sagemaker_preprocessing_job.py` launcher; SageMaker smoke run on `data-v0`
- [ ] Document the final `processed/<run_id>/` contract for the training step
- [ ] Future (out of scope now): upload the full dataset under `raw/`, scale test, tune sizing (section 12)

## 11. Open questions (defaults in bold)

- Read input directly from **`s3://`** in code (resolved: `ProcessingInput` copies data to each node's local disk, which breaks multi-node Spark; use it only for the config file, and record the input URI in `metadata.json`). See [preprocessing-pyspark-implementation-steps.md](preprocessing-pyspark-implementation-steps.md).
- **One job emitting all three splits**, or separate jobs per split? (One: thresholds need a single global `max(step)`.)
- Is `data-v0` already under `raw/` in S3? `scripts/upload_to_s3.py` covers uploading it if not.
- `run_id` naming: **`<git_sha>-<config_hash>`** (idempotent, deduplicates identical runs) or the processing job name (unique per execution)?

## 12. Future step: scale to the full dataset (15GB+)

Deferred. Do this only after the `data-v0` pipeline is complete end to end.

- Upload the full dataset under a new versioned prefix (e.g. `raw/data-v1/`).
- Raise `processing:` sizing (3-4 x `ml.m5.4xlarge`, larger volume, shuffle partitions for about 128-200MB each) via config only.
- Run the scale test: record duration, cost, spill and the longest task (skew indicator). Decide final sizing and whether the 4.2 fallback is needed.
- Optionally enable Spark event logs (`spark-history/`, section 7 item 4).
- Commit tuned settings to `preprocessing.yaml` / `config/env/`.
