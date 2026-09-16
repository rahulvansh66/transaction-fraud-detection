# MLflow wiring and reproducibility — current state and gaps

Where MLflow and run reproducibility stand after the preprocessing pass, and what
still has to be built. Companion to
[production_ml_experimentation_git_sagemaker_amt_mlflow.md](production_ml_experimentation_git_sagemaker_amt_mlflow.md)
and [best of his repo and our hpo approach.md](best%20of%20his%20repo%20and%20our%20hpo%20approach.md),
which describe the target operating model.

---

## 1. What "minimal MLflow wiring" means today

The only MLflow code is `_log_mlflow()` in
[src/model_build/training/train.py](../src/model_build/training/train.py):

- runs **only in the training step** (nothing in the Glue job or the Processing step)
- logs **5 params** — `max_depth`, `num_round`, `eta`, `scale_pos_weight`, `n_features`
- logs **1 metric** — `val_aucpr`
- logs **1 artifact** — the model, via `mlflow.xgboost.log_model`
- is wrapped in `try/except` and gated by `mlflow.enabled` + a tracking URI, so a
  tracking-server outage never fails training
- uses default experiment / run naming; no tags, no run linking, no registry call

It is a **safe hook**, not lifecycle integration.

## 2. Why it is minimal

Scope of the last pass was the preprocessing design doc (Glue + SageMaker
Processing). The pieces MLflow needs are not in place yet:

| Missing piece | Where it will come from |
| --- | --- |
| MLflow tracking server | `infra/terraform/modules/mlflow/` — still a stub. `cfg` has `mlflow.tracking_uri_ssm` pointing at an SSM value set to `"TODO"`. |
| Model promotion store | Registration currently targets a **SageMaker Model Package Group** (`PendingManualApproval`). Whether the **MLflow Model Registry** owns promotion state instead (or as well) is an open decision — see §5. |
| Experimentation modes (manual / HPO / production) | The experimentation architecture docs — a separate workstream. |

So the hook was kept working and non-fragile, and the rest deferred.

---

## 3. Reproducibility — the lineage a model must carry

A production model should be traceable to all of the following
([production_ml_experimentation doc, §15–16](production_ml_experimentation_git_sagemaker_amt_mlflow.md)).
Current state:

| Lineage input | Now | What to add |
| --- | --- | --- |
| **Code version** | not captured | git commit SHA as a run tag; CI stamps it into the pipeline execution (`ParameterString GitSha`) |
| **Config version** | not captured | env name + a hash of the merged `cfg`; attach the resolved config as a run artifact |
| **Data version** | **biggest gap** — Stage 2 reads the *moving* prefixes `curated/` and `aggregates/` | pin the exact curated partition(s) for the run and an immutable `aggregates/asof=<date>/` snapshot; log both S3 URIs with `mlflow.log_input` |
| **Pipeline version** | not captured | pipeline-definition hash + `PipelineExecutionArn` as a tag |
| **Environment / container version** | not captured | training + processing image URIs **with digest**; a locked `requirements.txt` (hashes) or `pip freeze` artifact; framework versions |
| **Execution parameters** | not captured | log every SageMaker Pipeline Parameter value used |
| **Hyperparameters** | subset (5) | full set, including `tree_method`, `n_jobs`, early-stopping settings |
| **Random seeds** | **none set anywhere** | set + log `random_state` on `XGBClassifier`; note that `percentile_approx` (median aggregate) and Spark shuffle are approximate by design; `approxQuantile` in the split already uses `relativeError = 0.0` |
| **Metrics** | `val_aucpr` only | train + val + **test**; threshold-free (AUPRC, ROC-AUC, recall@fixed-precision) **and** at a chosen operating threshold; confusion matrix; PR curve |
| **Artifacts** | model only | the fitted scaler (`prepare.py` already writes it to `output/model/scaler` — log it), `feature_names.json`, the evaluation report, the DQ result summary, an MLflow **model signature + input example** |

### Immutable data paths (do this first)

Reproducibility breaks if a re-run reads different bytes. Two concrete changes:

1. **Aggregates** — the Glue job already can write a dated snapshot
   (`aggregates/asof=YYYY-MM-DD/`). Make that the contract: Stage 2 reads a
   specific `asof=` snapshot passed as a pipeline parameter, not the latest.
2. **Curated** — Stage 2 should take an explicit list of `ingestion_date=`
   partitions (or a `<= date`) as a parameter, and record which it read.
3. The **train/val/test datasets** are already immutable
   (`datasets/<pipeline-exec-id>/…`) — keep that, and log the URIs.

---

## 4. MLflow across the ML lifecycle — gaps

1. **Tracking server** — provision it (`modules/mlflow/`: SageMaker-managed MLflow
   or self-hosted on ECS/RDS+S3), write the real URI to SSM, drop the `"TODO"`.
2. **Run structure** — one **parent run per pipeline execution**; the prepare,
   train, and evaluate steps log as **nested runs** under it. Then a model traces
   back to the exact curated snapshot, DQ outcome, `scale_pos_weight`, and split
   boundaries in one place.
3. **Stage 1 / Stage 2 lineage** — the Glue job and Processing step log nothing
   today. They should record: input partitions, row counts in/out, DQ rule
   outcomes, class balance, split cut points, `scale_pos_weight`.
4. **Evaluation step** — does not exist. Add a Processing step that scores the
   **test** fold and emits gating metrics (`mlflow.evaluate` or custom). The
   register step becomes **conditional** on it passing
   (`ConditionStep` in the pipeline).
5. **Registry decision** — pick one promotion-state store and wire transitions
   (`None → Staging → Production`) with the manual approval gate. Options:
   - MLflow Model Registry as the source of truth (matches the experimentation
     docs), SageMaker Model Package Group mirrors it, or
   - SageMaker Model Package Group stays authoritative and MLflow is
     tracking-only. Right now it is half-and-half — resolve it.
6. **`mode` tag** — tag every run `manual` / `hpo` / `production` so deliberate
   experiments, HPO trials, and scheduled retrains are distinguishable.
7. **HPO (AMT) integration** — each tuning trial as a child run; the winning
   config promoted, not auto-deployed (experimentation-doc §7, §10).
8. **Autologging** — `mlflow.xgboost.autolog()` + system-metrics logging instead
   of the hand-listed params.
9. **CI** — pass `GITHUB_SHA` / run number into the pipeline execution so every
   run and registered model links to a commit and a build.

---

## 5. Suggested order of work

| Pass | Delivers |
| --- | --- |
| **A. Tracking server** | `modules/mlflow/` + real SSM URI; `train.py` logs against it for real |
| **B. Immutable inputs** | dated `aggregates/asof=` + explicit curated partitions as pipeline parameters; `mlflow.log_input` on both |
| **C. Lineage + parent run** | parent-run-per-execution, nested step runs, git SHA / image digest / config hash / full HPs / seed logged; scaler + feature list + signature as artifacts |
| **D. Evaluation + gate** | test-fold evaluation step, gating metrics, `ConditionStep` before register |
| **E. Registry** | chosen promotion store wired with stage transitions + approval |
| **F. HPO** | AMT tuning step, trials as child runs, winner promoted |

Passes A–D are the reproducibility core; E–F are the experimentation operating
model.
