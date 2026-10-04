# How do we make sure re-running preprocessing never overwrites the processed data a model was trained on?

**Background.** Our pipeline takes the raw transactions CSV and runs a preprocessing job that cleans it, adds features and splits it into train, validation and test sets. The result is saved to S3 in a folder named after the run, for example `processed/abc123/`. The name `abc123` encodes the git commit and config that produced it. Training then reads that folder, so model `M` is tied to the data in `abc123`.

**The problem.** If someone runs preprocessing again with the same run name, the job could write into `processed/abc123/` again. Then the folder holds different rows than the ones `M` was trained on. The model can no longer be reproduced, and nothing shows an error.

**The fix.** Make each run folder write-once: it can be created, never changed. Two settings do this, each guarding a different moment.

## 1. `mode("errorifexists")`: guards the data

A Spark write setting. If the output folder already exists, the job fails instead of writing.

- `overwrite` would replace the old rows.
- `append` would add duplicate rows next to them.
- `errorifexists` stops, and the old data stays untouched.

**Catch:** Spark checks the folder and then writes, with a tiny gap in between. If two people launch the same run in the same second, both can pass the check and mix their files.

## 2. `IfNoneMatch="*"`: guards the launch

An S3 upload setting. It means "create this file only if nothing is there yet", and S3 checks that atomically. Before starting Spark, the launcher uploads a small marker file into the run folder using it.

- First launcher: S3 accepts, and the job starts.
- Second launcher, even a split second later: S3 refuses, and the launcher exits before any cluster starts.

This closes the gap above, and it fails early, so you don't pay for a doomed job.

## Which one catches what

| Situation | Caught by |
|---|---|
| Accidental re-run of an old run | Both |
| Two launches at the same moment | `IfNoneMatch` (atomic) |
| Data folder exists, marker missing | `errorifexists` |
| Someone deletes or edits files directly in S3 | Neither |

For that last case, use S3 Versioning or Object Lock plus a bucket policy that denies deletes, defined in Terraform.

**Rule of thumb:** `IfNoneMatch` decides who gets to start, and `errorifexists` makes sure the data is never overwritten.

Code: [spark_job.py](../../src/preprocessing/spark_job.py), [run_preprocessing_job.py](../../src/preprocessing/run_preprocessing_job.py).

---

> Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
