Q:
review parameters, do we have all parameters to build robus and reliable preprocessing job in production? also check if any production grade practice is missing

```yaml
processing:
  instance_type: "ml.m5.xlarge"
  instance_count: 1
  volume_size_gb: 20                            # dev; raise for the full 15GB+ dataset (shuffle spill)
  max_runtime_s: 3600
  target_file_mb: 128
  spark_conf:
    spark.sql.shuffle.partitions: "8"          # raise for the full dataset
    spark.sql.adaptive.enabled: "true"
    spark.sql.adaptive.skewJoin.enabled: "true"
```

A:
The configuration is good for a dev smoke run but not for production. Below are the missing parameters and practices, in priority order. I haven't edited anything.

## Missing parameters in `processing:`

| Parameter | Why it's needed |
|---|---|
| `spark.executor.memory`, `spark.driver.memory`, `spark.executor.cores`, `spark.executor.memoryOverhead` | Without these, Spark uses container defaults that don't match the instance. On the full run, a large window partition will OOM if executor memory isn't set explicitly. The plan mentions them, but the YAML omits them. |
| `spark.sql.files.maxPartitionBytes` and `spark.sql.shuffle.partitions` derived from data size | Read parallelism and shuffle sizing should scale with the input. A fixed `"8"` is only right for dev. |
| `spark.speculation: "false"` | Speculative duplicate tasks writing to S3 can conflict with the output committer. |
| `spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version: "2"`, or an S3-optimized committer | Without one, the rename on S3 is a slow copy and delete, and a failure mid-commit leaves partial output. |
| `spark.sql.parquet.compression.codec: "zstd"` (or snappy) | It should be pinned so file size and layout are reproducible. |
| `spark.sql.session.timeZone: "UTC"` | Pins time behaviour so results don't depend on the container's timezone. |
| `spark.sql.adaptive.coalescePartitions.enabled` and `spark.sql.adaptive.advisoryPartitionSizeInBytes` | This is how AQE targets the 128–200MB partition size the docs describe. |
| `framework_version`, `image_uri`, `py_version` | These are hardcoded in the launcher snippet (`"3.5"`). They should be config so upgrades are deliberate and reviewable. |
| `seed` | The plan promises it is passed through, but it isn't in the YAML. |
| `subnets`, `security_group_ids`, `encrypt_inter_container_traffic`, `volume_kms_key`, `output_kms_key` | These are the production network and encryption controls. The plan notes VPC hardening for later, but the parameters should exist now, defaulting to unset in dev. |
| `tags` | Needed for cost attribution (project, env, `run_id`). |
| `env` (environment variables) | This is the standard way to pass `git_sha` and `run_id` and to set `PYTHONHASHSEED`. |
| `spark_event_logs_s3_uri` | The docs make it optional. I'd make it a config field that is on by default for the scale run. |

## Production practices missing or weak

1. **A retry after a failed run gets blocked by its own partial output.** `run_id = git_sha-config_hash` is idempotent, and the writes use `errorifexists` per split. If the job dies after writing `train/` but before `val/`, a rerun with the same `run_id` fails on `train/`. The fix is to write to `processed/<run_id>/_staging/`, validate, then promote. A simpler option is to make `metadata.json` the last write and treat its presence as the completion marker (like `_SUCCESS`). The launcher should check for that marker, not just whether the prefix exists.
2. **Input snapshot isn't guaranteed.** The docs say to record the object count and bytes, but they don't verify that the input didn't change during the run. For prod, add an input manifest (S3 keys, sizes, ETags) to `metadata.json`, and have the raw prefix be write-once, ideally with S3 versioning or Object Lock, managed in Terraform.
3. **The schema isn't enforced.** The plan says to cast to explicit dtypes, but `select_raw` in the steps doc only checks column names. Add an explicit `StructType` with casts. Also add a data-quality gate with thresholds in config (max null fraction, minimum row count, minimum fraud count per split, schema drift). Today `validate_outputs` checks the output but never the input.
4. **No failure handling or alerting.** Add these in Terraform:
   - an EventBridge rule on `SageMaker Processing Job State Change` for `Failed` or `Stopped`, sent to SNS
   - a CloudWatch metric filter and alarm on `ERROR`
   - a step in the pipeline that retries on transient errors, because a standalone Processing job doesn't retry
5. **`max_runtime_s` is fixed at 3600.** This is too low for the full run and too generous for the smoke run. It should differ per env. Add `max_wait` or a budget alarm as well.
6. **No pinned code dependencies.** The job depends on whatever the AWS image ships, such as `pyyaml` and `boto3`. Pin these with a `requirements.txt` or a `submit_py_files` archive. Record the image URI and library versions in `metadata.json`.
7. **The output contract has no versioned schema.** Add `metadata_schema_version` and a feature schema hash. Training should fail fast if the data it reads isn't the version it expects.
8. **`dropDuplicates()` runs over all columns.** That is an expensive shuffle at 15GB. If a stable transaction key exists, deduplicating on it is cheaper and clearer. If not, document that.
9. **Repeated `count()` calls each trigger a full computation.** The docs mention `cache()`. At scale, `persist(MEMORY_AND_DISK)` is safer, or compute the counts in a single `agg`, because caching a 15GB frame in memory alone will spill.
10. **Two docs disagree on inputs.** [preprocessing-pyspark-processing-plan.md:50](DOCS/preprocessing-pyspark-processing-plan.md#L50) (architecture, section 2) still says to read data via `ProcessingInput`, while the steps doc says to use direct `s3://` reads. The steps doc is right for multi-node runs. Fix section 2 so the two docs agree.
11. **Cost guard in the scale run.** Add a Terraform-managed AWS Budget alarm and use Spot for non-prod, since Processing jobs support managed spot.
12. **CI is missing.** Add a GitHub Actions job that runs the local unit and parity tests on every PR.

Most of the parameter additions are small YAML changes. The items with real design impact are 1, 2 and 3. Do you want me to update both docs with these, or start with the config additions?

---
