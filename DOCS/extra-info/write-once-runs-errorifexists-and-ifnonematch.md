# How do we make sure re-running preprocessing never overwrites the processed data a model was trained on?

**Background.** Our pipeline takes the raw transactions CSV and runs a preprocessing job that cleans it, adds features and splits it into train, validation and test sets. The result is saved to S3 in a folder named after the run, for example `processed/abc123/`. The name `abc123` encodes the git commit and config that produced it. Training then reads that folder, so model `M` is tied to the data in `abc123`.

**The problem.** If someone runs preprocessing again with the same run name, the job could write into `processed/abc123/` again. Then the folder holds different rows than the ones `M` was trained on. The model can no longer be reproduced, and nothing shows an error.

**The fix.** Make each run folder write-once: it can be created, never changed. Two settings do this, each guarding a different moment.

Short answer:
- `mode("errorifexists")` : Adds check, if output folder of that job name is not present then only start spark job
- `IfNoneMatch="*"` : Before saving output, again check if 

## 1. `mode("errorifexists")`: guards the data

A Spark write setting. If the output folder already exists, the job fails instead of writing.

- `overwrite` would replace the old rows.
- `append` would add duplicate rows next to them.
- `errorifexists` stops, and the old data stays untouched.

**Catch:** Spark checks the folder and then writes, with a tiny gap in between. If two people launch the same run in the same second, both can pass the check and mix their files.

## 2. `IfNoneMatch="*"`: guards the launch

An S3 upload setting. It means "create this file only if nothing is there yet", and S3 checks that atomically. Before starting Spark, the launcher uploads [preprocessing.yaml](../../config/preprocessing/preprocessing.yaml) at `processed/<run_id>/` with `IfNoneMatch="*"`. Uploading it claims the run, and it stays next to the data as a record of the split fractions, feature flags and seed that produced it.

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

Q. I want to check my understanding. If I understood correctly, then this whole situation happens when we haven't made any change in the code or in the data, but still we are running a preprocessing job again. Am I right, or correct me if I am wrong anywhere?

Mostly right, with one correction.

**What you have right:** the scenario is a re-run of preprocessing under a run name that already has a folder. If neither code nor data changed, the run name is the same, so the job would target the same `processed/abc123/` folder. That is the classic case, and both guards catch it.

**Where it's too narrow:** the problem isn't limited to "nothing changed." It is "a run name that already exists gets used again." That can happen in other ways:

1. **Nothing changed, same name.** This is the case you described. The re-run would produce identical data, so it is harmless in theory. It is still blocked, because the guard can't tell identical output from different output. That is fine, since you can simply reuse the existing folder.
2. **Something changed, but the name didn't.** For example, someone edits the preprocessing code or the raw data without a new commit. Or the run name is something like `abc123` that doesn't capture every input. The re-run would then write different rows into the old folder. This is the dangerous case, and the one the doc is really protecting against.
3. **Two launches at the same moment.** There is no "re-run" here, just a race. Both launches could start before either folder exists. This is the case `IfNoneMatch` exists for.
4. **Retry after a partial failure.** A job crashes halfway and leaves some files in the folder. A re-run would mix old and new files. `errorifexists` blocks it, though you then have to clean up or choose a new run name.

Two other points:

- **Different code or config gives a different name.** If the run name really encodes the git commit and config, a change to either gives a different folder, so there is no conflict. The guard only matters when the name is reused.
- **The guard doesn't judge whether a re-run is wrong.** It enforces "one run name, one write," whatever the reason for the re-run. The fix is always a new run name, or reusing the existing output.

So I'd rephrase it as: "this happens whenever preprocessing runs again under a run name that already has output." Unchanged code and data is the simplest example, and the changed-but-same-name case is the one that does the real damage.

---

> Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
