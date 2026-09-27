# Step-by-Step Implementation: PySpark Preprocessing on SageMaker

Hands-on build order for the design in
[preprocessing-pyspark-processing-plan.md](preprocessing-pyspark-processing-plan.md).
Each step says **what** you do, **why**,
and shows a minimal snippet. The snippets are sketches to convey the idea, not final code.
Follow the repo rules in `CLAUDE.md` (docstrings on everything, type hints, `logging` not
`print`, Terraform for all AWS changes).

Scope: **`data-v0` only.** Scaling to the full dataset is a documented future step (end of this file).

Build order: **local first, cloud last.** Debug Spark on your laptop against `data-v0`, and only
then pay for a SageMaker run.

---

## 0. Mental model (read once)

**PySpark** is Python driving Spark. You describe transformations on a DataFrame (like pandas,
but lazy and distributed). Nothing runs until an *action* (`count()`, `write`, `collect()`).

| Pandas (notebook) | PySpark (job) |
|---|---|
| `df["a"] = ...` | `df = df.withColumn("a", ...)` (DataFrames are immutable, so you re-assign) |
| `df[df.x > 1]` | `df.filter(F.col("x") > 1)` |
| `groupby().cumcount()` | window function: `F.count("*").over(Window...)` |
| runs in memory, eagerly | builds a plan, runs on `write`/`count`, spread over many cores/machines |

**SageMaker Processing job** = "run this script on a temporary container/cluster, then shut it
down". For Spark, SageMaker gives you an AWS-managed Spark image; you supply your `.py` script.
You are billed only while the job runs. Logs go to CloudWatch.

**Two pieces of code, two places they run:**
- *Launcher* (`src/preprocessing/run_preprocessing_job.py`) runs on **your machine**. It only tells
  SageMaker "start a job with this script and these inputs".
- *Job script* (`src/preprocessing/spark_job.py`) runs **inside SageMaker**. This is the real work.

---

## 1. Prerequisites and dependencies

**Goal:** be able to run Spark locally.

1. Install **Java 17** (Spark needs a JVM). Check with `java -version`.
   - Windows only: local Spark can also need `winutils.exe` / `HADOOP_HOME`. If you hit
     `HADOOP_HOME and hadoop.home.dir are unset`, either install winutils or run the local
     tests in WSL/Linux. It is only a local quirk; SageMaker runs on Linux.
   - Reading parquet locally fails with `UnsatisfiedLinkError ... NativeIO$Windows.access0`
     without them. Setup done on this machine: `winutils.exe` + `hadoop.dll` (hadoop-3.3.6 from
     `cdarlint/winutils`) in `F:\hadoop\bin`, with user env var `HADOOP_HOME=F:\hadoop`
     (open a new terminal to pick it up). `F:\hadoop\bin` must also be on `PATH` for
     `hadoop.dll` to load; `tests/preprocessing/conftest.py` adds it automatically for tests, but
     add it to your user `PATH` if you run `spark_job.py` directly on Windows.
2. Add dependencies with uv:
   ```bash
   uv add --dev pyspark==3.5.* pytest
   uv add sagemaker-core sagemaker-mlops
   ```
   **This project uses SageMaker Python SDK v3 everywhere** (imports under `sagemaker.core.*` and
   `sagemaker.mlops.*`). Most online examples are v2 (`sagemaker.spark.processing`,
   `ProcessingInput(source=, destination=)`), so check field names in the installed package.
   `sagemaker-core` provides `PySparkProcessor`; `sagemaker-mlops` provides Pipelines (used in
   step 10). `sagemaker-mlops` pulls in `torch` transitively (its `sagemaker-serve` dependency
   imports it), so `.venv` is about 1.5 GB. Our code never uses `torch`.
   Match the `pyspark` minor version to the SageMaker container `framework_version` you will use
   (`3.5`), so local behaviour matches the cloud.
3. Confirm AWS access works (`aws sts get-caller-identity` with your profile) and that `data-v0`
   is in S3 under `raw/` (`scripts/upload_to_s3.py` does this).

**Done when:** `uv run python -c "from pyspark.sql import SparkSession; SparkSession.builder.master('local[1]').getOrCreate().range(3).show()"` prints a table.

---

## 2. Extend the config

**Goal:** no hardcoded values (`ml-repo-structure` skill). Add a `processing:` block to
[config/preprocessing/preprocessing.yaml](../config/preprocessing/preprocessing.yaml). These are
job-sizing settings, not model hyperparameters.

```yaml
processing:
  instance_type: "ml.m5.xlarge"
  instance_count: 1
  volume_size_gb: 20                            # sized for data-v0; raise when scaling (future step)
  max_runtime_s: 3600
  target_file_mb: 128
  spark_conf:
    spark.sql.shuffle.partitions: "8"          # sized for data-v0; raise when scaling (future step)
    spark.sql.adaptive.enabled: "true"
    spark.sql.adaptive.skewJoin.enabled: "true"
```

The env-specific values (bucket, role ARN) stay in `config/env/dev.yaml`; the job never sees them.

---

## 3. Write the feature functions (`src/preprocessing/features.py`)

**Goal:** port each notebook function as a pure `DataFrame -> DataFrame` function. Do them one at a
time and test each (step 5) before moving on. Every function needs a Google-style docstring.

### 3.1 Constants and schema check (from `load_raw`)

```python
RAW_COLUMNS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
               "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]

def select_raw(df: DataFrame) -> DataFrame:
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df.select(*RAW_COLUMNS)            # also drops the `day` partition column
```

### 3.2 Clean (from `clean`)

```python
def clean(df: DataFrame) -> DataFrame:
    return df.dropna(subset=RAW_COLUMNS).dropDuplicates()
```
Row counts for logging come from `df.count()`. Each `count()` triggers a full computation, so log
counts sparingly (start, after clean, per split) and `cache()` the cleaned frame if you reuse it.

### 3.3 Account counts up to the current step (from `add_past_counts`), the hard one

A **window** is Spark's version of a grouped running count. Partition by account, order by `step`,
count all rows from the start up to the current step.

```python
from pyspark.sql import Window, functions as F

def add_past_counts(df: DataFrame) -> DataFrame:
    def running_count(account_col: str) -> Column:
        w = (Window.partitionBy(account_col).orderBy("step")
                   .rangeBetween(Window.unboundedPreceding, Window.currentRow))   # RANGE, not ROWS
        return F.count(F.lit(1)).over(w)
    return (df.withColumn("orig_txn_count", running_count("nameOrig"))
              .withColumn("dest_txn_count", running_count("nameDest")))
```
Why **order-free**: `step` is one simulated hour and the only time axis, so rows in the same step
are simultaneous. A `RANGE` frame includes every row with the same `step`, so tied rows share one
count and no order between them is invented. (An earlier version broke ties alphabetically by
`type`, which reverses the real TRANSFER-then-CASH_OUT pattern, and file order is neither
guaranteed meaningful nor reproducible in Spark.) The notebook's `add_past_counts` computes the
same thing in pandas, so both engines agree exactly. Must run **before** the type filter.

**Keep in mind for serving:** the raw account IDs are never model inputs, but these two counts are
derived from them, so they are the main train/serve skew risk. Serving must compute them over the
account's full prior history (all types, all earlier steps, current transaction included). The
rules live in [config/inference/feature_contract.yaml](../config/inference/feature_contract.yaml);
update it whenever `features.py` changes.

### 3.4 Filter and basic features (from `filter_types`, `add_basic_features`)

```python
def filter_types(df: DataFrame, keep_types: list[str]) -> DataFrame:
    return df.filter(F.col("type").isin(keep_types))

def add_basic_features(df: DataFrame, night_hours: list[int]) -> DataFrame:
    hour = F.col("step") % 24
    return (df.withColumn("is_transfer", (F.col("type") == "TRANSFER").cast("byte"))
              .withColumn("log_amount", F.log1p("amount"))
              .withColumn("hour_of_day", hour.cast("short"))
              .withColumn("is_night", hour.between(night_hours[0], night_hours[1]).cast("byte"))
              .withColumn("day_of_month", F.floor(F.col("step") / 24).cast("short")))
```

### 3.5 Leakage gate (from `apply_leakage_gate`)

Reuse the same `BASE_FEATURES` / `ERROR_FEATURES` lists and end with
`df.select("step", *features, label)`. The column order matters, because the training step relies
on it. Log a `WARNING` if error features are enabled.

### 3.6 Time split and class weight (from `time_split`, `compute_scale_pos_weight`)

```python
def split_thresholds(df: DataFrame, train_frac: float, val_frac: float) -> tuple[int, int]:
    if not 0 < train_frac < val_frac < 1:
        raise ValueError("require 0 < train_frac < val_frac < 1")
    max_step = df.agg(F.max("step")).first()[0]          # one small job, result on the driver
    return math.floor(train_frac * max_step), math.floor(val_frac * max_step)

def time_split(df, t1, t2) -> dict[str, DataFrame]:
    return {"train": df.filter(F.col("step") <= t1),
            "val":   df.filter((F.col("step") > t1) & (F.col("step") <= t2)),
            "test":  df.filter(F.col("step") > t2)}

def compute_scale_pos_weight(train: DataFrame, label: str) -> float | None:
    n_pos, n = train.agg(F.sum(label), F.count("*")).first()
    return None if not n_pos else (n - n_pos) / n_pos
```

**Done when:** all functions exist with docstrings and type hints, and import without errors.

---

## 4. Write the job entry point (`src/preprocessing/spark_job.py`)

**Goal:** glue: parse arguments, create the Spark session, call the functions in order, write output.

Key points:
- `main()` is the only place that calls `logging.basicConfig(...)`.
- Everything variable arrives as **command-line arguments** (`--input-uri`, `--output-uri`, `--config`, `--git-commit`, `--run-id`), so the same script runs locally and on SageMaker.
- **Read and write S3 directly with `s3://` URIs.** In a multi-machine Spark cluster each node must reach the whole dataset; SageMaker's `ProcessingInput` copies data onto each node's *local disk*, which does not work for a distributed job. Use `ProcessingInput` only for the small config file.

```python
def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = yaml.safe_load(open(args.config))
    spark = SparkSession.builder.appName("fraud-preprocessing").getOrCreate()

    logger.info("step=start run_id=%s git_commit=%s input=%s", args.run_id, args.git_commit, args.input_uri)
    df = spark.read.option("basePath", args.input_uri).parquet(args.input_uri)

    df = features.clean(features.select_raw(df))
    df = features.add_past_counts(df)
    df = features.filter_types(df, cfg["data"]["keep_types"])
    df = features.add_basic_features(df, cfg["features"]["night_hours"])
    df, feature_names = features.apply_leakage_gate(df, cfg["features"]["include_balance_error_features"])

    t1, t2 = features.split_thresholds(df, cfg["split"]["train_frac_of_max_step"], cfg["split"]["val_frac_of_max_step"])
    splits = features.time_split(df, t1, t2)
    weight = features.compute_scale_pos_weight(splits["train"], cfg["data"]["label_col"])

    for name, part in splits.items():
        part.write.mode("errorifexists").parquet(f"{args.output_uri}/{name}")   # never overwrite a run
    write_metadata(...)      # JSON to <output_uri>/metadata.json (see 4.1)
    validate_outputs(spark, args.output_uri, feature_names)
```

### 4.1 `metadata.json`
Build a dict with `run_id`, `git_commit`, `spark_version`, `source_path`, `features`, `label`,
`thresholds`, `scale_pos_weight`, `row_counts`, `fraud_counts`, `config`, `config_hash`. Write it
with `boto3.client("s3").put_object(...)` when the URI starts with `s3://`, otherwise to a local
file (this keeps local tests working).

### 4.2 Output validation
Re-read the written splits and assert the notebook's contract (columns, no nulls, no leaky
columns, ordered step ranges). A violation raises, so the job fails instead of producing a bad dataset.

### 4.3 Log hygiene
Log only aggregates (counts, thresholds, fractions), never rows or account IDs. Use `key=value`
lazy format: `logger.info("step=split part=%s rows=%d fraud=%d", name, n, f)`.

**Done when:** `spark_job.py --help` works and `main()` reads top to bottom like the notebook.

---

## 5. Test locally

**Goal:** prove the port is correct before touching AWS.

### 5.1 Unit tests (`tests/preprocessing/test_features.py`)
Use one shared local session:

```python
@pytest.fixture(scope="session")
def spark() -> SparkSession:
    return SparkSession.builder.master("local[2]").config("spark.sql.shuffle.partitions", "2").getOrCreate()

def test_past_counts_same_step_rows_share_a_count(spark):
    df = spark.createDataFrame([(11, "PAYMENT", 1.0, "C1", "M1"), (12, "TRANSFER", 5.0, "C1", "C2"),
                                (12, "CASH_OUT", 5.0, "C1", "C3")],
                               ["step", "type", "amount", "nameOrig", "nameDest"])
    out = {(r.step, r.type): r.orig_txn_count for r in features.add_past_counts(df).collect()}
    assert out == {(11, "PAYMENT"): 1, (12, "TRANSFER"): 3, (12, "CASH_OUT"): 3}   # tied rows share a count
```
Cover: duplicates, ties at one `step`, an account seen in a non-kept type (its count must still
include those rows), train with no positives, invalid split fractions.

### 5.2 Parity test (`test_parity.py`)
Run the whole job locally on `dataset/raw/data-v0`:
```bash
uv run python src/preprocessing/spark_job.py --input-uri dataset/raw/data-v0 \
  --output-uri dataset/processed/spark-v0 --config config/preprocessing/preprocessing.yaml \
  --git-commit local --run-id local
```
Compare with the notebook's `dataset/processed/data-v0`: same columns, same rows per split (sort
both before comparing), same `thresholds` and `scale_pos_weight`. Fix any differences here, where
debugging is cheap.

Windows note: Spark starts `python` for its workers, which resolves to the Microsoft Store shortcut
and fails with `Python was not found`. `tests/preprocessing/conftest.py` pins `PYSPARK_PYTHON` to
the running interpreter; set the same variable if you run the job script directly on Windows.

**Done when:** `uv run pytest tests/preprocessing` is green and the parity test matches.
(Status: 14 tests pass locally, including the parity check against the re-executed notebook.)

---

## 6. Terraform changes

**Goal:** give the SageMaker role what Spark needs. Edit `infrastructure/`, never the console.

1. In [infrastructure/s3.tf](../infrastructure/s3.tf), extend the `sagemaker_data_access` policy:
   Spark writes to a temp path and renames it (copy + delete), so `processed/*` also needs
   `s3:DeleteObject` and `s3:AbortMultipartUpload`, and the bucket needs `s3:ListBucketMultipartUploads`.
   ```hcl
   statement {
     sid     = "ReadWriteProcessedAndModels"
     actions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"]
     resources = ["${data.aws_s3_bucket.data.arn}/processed/*", "${data.aws_s3_bucket.data.arn}/models/*"]
   }
   ```
2. Add a log group with explicit retention (new file, e.g. `infrastructure/cloudwatch.tf`):
   ```hcl
   resource "aws_cloudwatch_log_group" "processing_jobs" {
     name              = "/aws/sagemaker/ProcessingJobs"
     retention_in_days = 14   # dev; longer in prod
   }
   ```
   If the log group already exists (SageMaker auto-creates it on the first job), import it into
   state first (`terraform import`) instead of creating a duplicate.
3. Check that the execution role can write logs and pull the SageMaker Spark image from ECR.
   Add only what is missing.
4. Run `terraform plan`, **read the diff**, then `terraform apply`.

**Done when:** `terraform plan` shows no drift and the policy contains the new actions.

---

## 7. Write the launcher (`src/preprocessing/run_preprocessing_job.py`)

**Goal:** start the SageMaker job from your machine. This is where SageMaker-specific code lives.

```python
from sagemaker.core.spark.processing import PySparkProcessor
from sagemaker.core.shapes import ProcessingInput, ProcessingS3Input

processor = PySparkProcessor(
    base_job_name="fraud-preprocess",
    framework_version="3.5",                      # pinned Spark version
    role=env["aws"]["iam_roles"]["sagemaker_execution_role_arn"],
    instance_type=cfg["processing"]["instance_type"],
    instance_count=cfg["processing"]["instance_count"],
    volume_size_in_gb=cfg["processing"]["volume_size_gb"],
    max_runtime_in_seconds=cfg["processing"]["max_runtime_s"],   # cost guard
)

run_id = f"{git_sha}-{config_hash}"
processor.run(
    submit_app="src/preprocessing/spark_job.py",
    submit_py_files=["src/preprocessing/features.py"],           # so `import features` works
    # v3: ProcessingInput takes an S3 location (no local `source=`), so the launcher first uploads the
    # config to S3, e.g. s3://{bucket}/processed/{run_id}/config/ via boto3, then points at it:
    inputs=[ProcessingInput(input_name="config",
                            s3_input=ProcessingS3Input(
                                s3_uri=f"s3://{bucket}/processed/{run_id}/config/preprocessing.yaml",
                                local_path="/opt/ml/processing/input/config",
                                s3_data_type="S3Prefix", s3_input_mode="File"))],
    arguments=["--input-uri", f"s3://{bucket}/raw/data-v0/",
               "--output-uri", f"s3://{bucket}/processed/{run_id}",
               "--config", "/opt/ml/processing/input/config/preprocessing.yaml",
               "--git-commit", git_sha, "--run-id", run_id],
    configuration=[{"Classification": "spark-defaults",
                    "Properties": cfg["processing"]["spark_conf"]}],
    wait=True, logs=True,
)
```

Notes for a beginner:
- `submit_app` is uploaded to S3 by the SDK and executed with `spark-submit` inside the container.
- `logs=True` streams CloudWatch logs to your terminal, so you can watch the job.
- Refuse to launch with a dirty git tree or a run whose `processed/<run_id>/` already exists
  (reproducibility rule).
- Use your own AWS profile/credentials to launch; the *role* is what the job itself runs as.

---

### 7.1 Concept: how code reaches the container (the deps zip)

A SageMaker Spark container receives only what the launcher hands it, not the repo. `PySparkProcessor.run(submit_app=...)`
uploads and runs the entry script (`spark_job.py`); every module it imports must be shipped separately.

| File | In `preprocessing_deps.zip`? | Why |
| --- | --- | --- |
| `spark_job.py` | No | Entry script, shipped by `submit_app`. A second copy would have to be kept in sync. |
| `features.py` | Yes | Imported by `spark_job.py`. |
| `lineage.py` | Yes | Imported by `spark_job.py` for `config_hash`; the launcher imports the same file, so both sides always compute the same hash. |
| `run_preprocessing_job.py` | No | Client-side (laptop/CI); needs `boto3` and the SageMaker SDK, which the container never uses. |

Rule: the zip holds the library modules the entry script imports, not the entry script and not the launcher. The
container mounts it at `/opt/ml/processing/input/deps/`, and `spark_job.py` puts the zip on `sys.path` (Python imports
straight from a zip), so `import features` works there. Nothing on your laptop reads the zip.

**Why the zip lives under `processed/<run_id>/deps/` (and not a shared `deps/` path):**
- **Reproducibility:** the run prefix then holds the exact code (`deps/`), parameters (`config/`) and output of that run,
  so "which `features.py` produced this model's data?" is answered by looking in one folder.
- **No overwrites:** a shared, mutable path would be replaced by the next run, silently destroying the code of every earlier
  run and breaking audit and rollback. Run-scoped, immutable keys cannot be clobbered.
- **No races:** two runs launched at once (different commits or data versions) each read their own zip, never the other's.
- **Cleanup and access:** retention or deletion of a run, and IAM scoping, apply to the whole prefix in one rule.
- The zip is built deterministically (fixed timestamps, sorted entries), so identical sources give identical bytes and a
  stable checksum (`deps_sha256` in the launcher log).

### 7.2 Concept: one job script, local or SageMaker

The launcher (`run_sagemaker_preprocessing_job.py`) is SageMaker-only and never runs Spark itself. Local vs cloud is
decided by how `spark_job.py` is invoked, and the script has no environment switch:

- **Paths are arguments.** `--input-uri` / `--output-uri` take local paths or `s3://` URIs. Locally you pass local paths;
  the launcher passes S3 URIs.
- **Import fallback.** `spark_job.py` tries `from src.preprocessing import features` (local, package import) and on failure
  adds the mounted `preprocessing_deps.zip` to `sys.path` and imports `features` / `lineage` as top-level modules (SageMaker
  ships only the one script, not the `src` package; see 7.1).
- **Spark master.** Local runs use `local[*]`; on SageMaker the Spark container provides the cluster.

| | Local | SageMaker |
| --- | --- | --- |
| Entry | `python src/preprocessing/spark_job.py ...` | `python -m src.preprocessing.run_sagemaker_preprocessing_job` |
| Data | local paths | `s3://` URIs |
| Code | repo imports | deps zip mounted at `/opt/ml/processing/input/deps` |

## 8. Smoke run on SageMaker (`data-v0`)

1. `uv run python -m src.preprocessing.run_preprocessing_job`
2. Watch the streamed logs. First-time failures are usually one of: `AccessDenied` (step 6 not
   applied), `ModuleNotFoundError` (a dependency missing in the container, so add it or use
   `submit_py_files`), or a wrong S3 path.
3. Check S3: `processed/<run_id>/train|val|test/part-*.parquet` and `metadata.json`.
4. Download the output and compare with the local run from step 5.2. It should match.
5. Look at the job in the SageMaker console (Processing) and the log group in CloudWatch, and
   confirm the retention from step 6 is set.

**Done when:** the job status is `Completed`, the output matches the local run, and logs show the
`run_id`, `git_commit` and split counts, with no PII.

**Status (done).** The smoke run completed; splits, `thresholds`, `scale_pos_weight`, row and fraud counts match
`dataset/processed/data-v0`. Launch with `uv run python -m src.preprocessing.run_preprocessing_job [--allow-dirty]`
(`--allow-dirty` adds a `-dirty-<diff hash>` suffix to the run id; use only for smoke runs).
Run id = `<git_sha>-<config_hash>-<data_version>`, so the same code and config on a new dataset version gets its own prefix.

SDK v3 quirks hit while building the launcher (workarounds are in the code):
- `submit_py_files` and `configuration` on `PySparkProcessor.run()` fail with a pydantic `ValidationError`
  (the SDK builds v2-style `ProcessingInput(source=, destination=)` internally). Workaround (temporary until the SDK is fixed, or a custom image replaces it): a deterministic, versioned
  `preprocessing_deps.zip` (`features.py` + `lineage.py`) is uploaded under `processed/<run_id>/deps/` and mounted
  through our own `deps` input channel (`/opt/ml/processing/input/deps`, zip added to `sys.path` by `spark_job.py`), and `processing.spark_conf` is applied by `spark_job.py` on the SparkSession builder.
- The SDK stages `submit_app` in its default bucket (`sagemaker-<region>-<account>`), which the execution role cannot
  read. The launcher sets `default_bucket` to the data bucket with prefix `processed/_sdk`, which the role can read.
- The "run already exists" guard is an atomic S3 conditional write (`IfNoneMatch="*"`) of the staged config, so two
  concurrent launches cannot both proceed. A failed run keeps its id: fix and commit (new sha) to relaunch.

---

## 9. Hand off to training (step 4 of the main plan)

**Output contract** (`s3://<bucket>/processed/<run_id>/`):
- `train/`, `val/`, `test/`: parquet, columns `step`, the feature columns (in `metadata.json["features"]` order), then the label.
- `metadata.json`: `run_id`, `git_commit`, `spark_version`, `source_path`, `features`, `label`, `thresholds`,
  `scale_pos_weight`, `row_counts`, `fraud_counts`, `config`, `config_hash`.
- `config/` and `deps/`: the exact config and `features.py` the job ran with (lineage). Not read by training.
- Runs are immutable; never overwrite. The prefix is the only thing training needs (`--data-uri`).

- The training step reads `s3://.../processed/<run_id>/` and `metadata.json`.
- Training logs `run_id`, `git_commit`, the data URI and `scale_pos_weight` to MLflow, so any model
  traces back to this exact dataset (`ml-lineage-reproducibility` skill).
- Later (step 5), wrap the launcher's call as a `ProcessingStep` in a SageMaker Pipeline, with the
  same script and arguments. The packages are already installed; imports (verified to load):
  ```python
  from sagemaker.mlops.workflow.pipeline import Pipeline
  from sagemaker.mlops.workflow.steps import ProcessingStep
  ```
  We use SageMaker SDK v3 throughout (`sagemaker.core.*`, `sagemaker.mlops.*`), never the v2
  `sagemaker.workflow` / `sagemaker.spark.processing`. In v3, `ProcessingStep` takes `step_args`
  (not a processor plus inputs); confirm the exact usage against the installed package when we build it.

---

## Future step: scale to the full dataset (15GB+)

**Not part of the current scope.** We build and verify everything on `data-v0` (steps 1-9). Scaling
is a later, separate piece of work; the code is already written to scale by config only.

1. Upload the full dataset under `raw/` (a new versioned prefix, e.g. `raw/data-v1/`).
2. Raise the `processing:` block: more instances (e.g. 3-4 x `ml.m5.4xlarge`), a larger
   `volume_size_gb`, and `spark.sql.shuffle.partitions` sized for about 128-200MB per partition.
3. Enable `spark_event_logs_s3_uri` on `run()` if you want the Spark UI history for debugging.
4. Run, then record: total duration, cost, and skew (one task much slower than the rest usually
   means a hot account in a window). If a single partition OOMs, apply the fallback in the design
   doc (4.2).
5. Commit the tuned settings in `preprocessing.yaml` / `config/env/`.

**Done when:** the full run completes within the runtime guard and `metadata.json` row counts look
right (roughly 40% of raw rows kept, fraud in every split, ideally).

---

## Checklist

- [ ] 1. Java 17, `pyspark`, `pytest`, `sagemaker-core`, `sagemaker-mlops` (SDK v3) installed; local Spark runs
- [ ] 2. `processing:` block added to `preprocessing.yaml`
- [ ] 3. `features.py` ported (all functions, docstrings, types)
- [ ] 4. `spark_job.py` with args, metadata and output validation
- [ ] 5. Unit tests and local parity with the notebook pass (order-free counts in both)
- [x] 6. Terraform: S3 delete/multipart permissions, log group retention; plan reviewed and applied
- [x] 7. Launcher script
- [x] 8. SageMaker smoke run on `data-v0` matches the local output (verified: run `3c5bc15-2946fa33336b-dirty`, all splits and metadata identical to the notebook output)
- [x] 9. Output contract documented for the training step
- [ ] Future: scale test on the full dataset; sizing committed (out of current scope)

## Common pitfalls

| Symptom | Likely cause |
|---|---|
| Counts differ from the notebook | Notebook and Spark not both using order-free counts (`RANGE` frame on `step`, step 3.3) |
| `AccessDenied` on write or rename | Missing `s3:DeleteObject` on `processed/*` (step 6) |
| Job hangs or is very slow at scale (future step) | Skewed window key, too few shuffle partitions, or too small instances |
| One tiny file per task / thousands of files | Missing `repartition` before write (design doc 4.5) |
| `FileNotFoundError` for input on multi-node runs | Data passed via `ProcessingInput` (local disk); use an `s3://` URI in the script |
| Local Spark fails on Windows with `HADOOP_HOME` | Local winutils issue only; use WSL or install winutils |
