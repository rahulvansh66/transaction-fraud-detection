# Preprocessing: Glue Job + SageMaker Processing Step

How raw transaction data becomes train/val/test datasets. Two stages, each owning a
different part of the work so feature logic is written **once** (no train/serve skew).

- **Companion doc:** [serving-realtime-endpoint.md](serving-realtime-endpoint.md) — how the
trained model is served and how it consumes the account-history aggregates this pipeline
produces.
- **Reference notebook:** [experiment/predicting-fraud-in-financial-payment-services.ipynb](../experiment/predicting-fraud-in-financial-payment-services.ipynb)
- **Target use case:** real-time transaction authorization — the model scores a transaction
before it settles.

---



## 1. The two stages at a glance

```text
Raw transactions (S3, Glue Data Catalog)
        │
        ▼
┌───────────────────────────────┐
│ STAGE 1 — Glue job (PySpark)   │   runs on a schedule (data freshness)
│                               │
│  • row filter + row-level      │
│    feature derivation          │
│  • account-history aggregates  │
│    (rolling windows, profile)  │
│  • data-quality gate           │
└───────────────────────────────┘
        │                 │
        │ curated         │ per-account
        │ row-level       │ aggregates
        │ Parquet (S3)    │ Parquet (S3)
        ▼                 │
┌───────────────────────────────┐        │
│ STAGE 2 — SageMaker Processing │        │  (also read by the
│ step (PySpark)                 │        │   serving endpoint —
│                               │        │   see companion doc)
│  • read curated data as-is     │        │
│  • point-in-time join of ◄─────┼────────┘
│    the aggregates              │
│  • time-based train/val/test   │
│    split                       │
│  • fit transforms on train     │
│    fold only                   │
└───────────────────────────────┘
        │
        ▼
train / val / test datasets (S3)  ──►  SageMaker Training step
```

**Why split at all.** Both stages use PySpark and could technically do each other's work.
They are separated by *responsibility*:


|                         | Glue job                                                           | SageMaker Processing step                                           |
| ----------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------- |
| **Question it answers** | "What does this account normally look like, based on its history?" | "Given curated data, produce the datasets for *this* training run." |
| **Trigger**             | EventBridge Scheduler (cron) → Step Functions                      | A SageMaker Pipeline execution (a training run)                     |
| **Input**               | Raw transaction history (S3, via Data Catalog)                     | Curated data + aggregates the Glue job wrote (S3)                   |
| **Output**              | Curated row-level Parquet **+** per-account aggregates → S3        | Train/val/test datasets for one training job → S3                   |
| **Also consumed by**    | The serving endpoint (loads aggregates into memory)                | Only the Training step                                              |
| **Runs how often**      | On a schedule, independent of training                             | Once per training run                                               |


If both stages re-derived the same feature independently, the two definitions would drift
apart over time and the model would be trained on one version of a feature and scored in
production on another. Defining each feature in exactly one place prevents that.

---



## 2. Which features are built where



### Row filter (Glue, first thing)

Keep only `type in {TRANSFER, CASH_OUT}`. All fraud in this dataset lives in those two
types; everything downstream assumes the filter is already applied.

### Stage 1 — features the **Glue job** produces

These are written into the curated Parquet. The Processing step reads them as-is and never
recomputes them.

#### 1a. Row-level features (one value per transaction row)


| Feature  | Definition                     | Notes                                                       |
| -------- | ------------------------------ | ----------------------------------------------------------- |
| `type`   | `TRANSFER` → 0, `CASH_OUT` → 1 | Encoded here once. The Processing step never re-encodes it. |
| `amount` | Requested transaction amount   | Carried through from raw data unchanged.                    |
| `hour`   | `step % 24`                    | Hour of day the transaction happened.                       |
| `day`    | `step // 24`                   | Day index since the start of the data.                      |


That is the entire **v0 row-level feature set** — 4 features, deliberately minimal. No
balance columns, no beneficiary-side (Tier-2) features, no `isFlaggedFraud`. See
[Appendix A](#appendix-a--why-the-balance-columns-are-excluded) for the reasoning. This
makes v0 weak on purpose; the project goal is a working end-to-end pipeline, not SOTA
metrics. Real predictive signal is expected to come from the account-history aggregates
below, not from a single-row snapshot.

#### 1b. Account-history aggregates (one row per account, precomputed in batch)

This is the actual reason the Glue job exists as a separate scheduled job: it looks across
the whole history to summarise each account's normal behaviour.


| Feature                                             | Definition                                                                                                                                                                                            |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `txn_count_1h` / `txn_count_24h` / `txn_count_7d`   | Count of the account's prior transactions in the trailing 1h / 24h / 7d window before the current transaction.                                                                                        |
| `txn_sum_1h` / `txn_sum_24h` / `txn_sum_7d`         | Total amount the account transacted in the trailing 1h / 24h / 7d window.                                                                                                                             |
| `avg_txn_amount`                                    | Mean transaction size for the account over its history — its normal spend level.                                                                                                                      |
| `typical_txn_amount`                                | Robust "typical" transaction size (e.g. median) — less skewed by one large transaction than the mean.                                                                                                 |
| `time_since_last_txn`                               | Time elapsed (in `step` units) since the account's previous transaction.                                                                                                                              |
| `mule_linkage_flag` *(later, not in first version)* | Set when an account received a fraudulent `TRANSFER` as `nameDest` and is now the `nameOrig` of a following `CASH_OUT` — the account-takeover / mule pattern from [dataset-info.md](dataset-info.md). |


The exact aggregate list is still open — see [Open items](#open-items). Windows and profile
stats above are the first draft.

#### 1c. Data-quality gate (Glue, before writing anything)

Run Glue Data Quality (or Deequ) checks and **fail the job** rather than write bad data
downstream:

- null-rate thresholds per column
- class-balance sanity check (fraud rate within an expected band)
- schema conformance against the Data Catalog



### Stage 2 — what the **SageMaker Processing step** produces

This step derives **no new row-level fields**. It joins, splits, and fits transforms.

> **Implementation note (v0).** The Glue job keys each aggregate row at the `step` it
> describes and every window is built from history **strictly before** that `step`
> (upper bound `-1`). So the point-in-time join is a plain equi-join on
> `(nameOrig, step)` — a curated row at step *S* is matched with the aggregate computed
> as of step *S*, which by construction contains nothing from step *S* or later. Rows with
> no matching aggregate (an account's first transaction) take the shared
> "no history" fills (`txn_count_* = 0`, `time_since_last_txn = -1`). See
> `src/fraud_features/pit_join.py` and `src/fraud_features/nulls.py`. `day` is written to
> the curated Parquet but left out of the model feature list.


| Output                                                      | What it is                                                                                                                                                                                                 | Fit on which data |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| Curated row-level columns (`type`, `amount`, `hour`, `day`) | Read straight from the curated Parquet. No re-cleaning, no re-deriving.                                                                                                                                    | —                 |
| Joined account-history aggregates                           | The per-account aggregates joined onto every training row **point-in-time correct**: join on `step` so a row only sees aggregate values that existed *before* it — exactly what inference would have seen. | —                 |
| Imputed nulls                                              | Statistical fill (train-fold **median** by default) as a safety net over the structural "no history" fills. `strategy: none` disables it.                                                                   | train fold        |
| Skew-corrected features                                    | `log1p` applied to currency-scale columns whose **train-fold `|skewness|` exceeds `threshold`** (default 1.0). Which columns were transformed is recorded in `transforms.json`.                              | train fold        |
| Outlier-handled `amount`                                   | IQR fences (`Q1 - k·IQR .. Q3 + k·IQR`) fit on the **non-fraud** train rows; `clip` (all folds) or `drop` (train fold only). See the caveat below.                                                          | train fold (non-fraud) |
| Scaled numeric features                                     | Numeric columns scaled with a scaler **fit on the train fold only**, then applied to val/test.                                                                                                             | train fold        |
| Encoded categorical features                                | Any remaining categorical encoding, **fit on the train fold only**.                                                                                                                                        | train fold        |
| `scale_pos_weight`                                          | Class-weight value from the **final** train fold's fraud / non-fraud ratio (after any `drop`), handed to the Training step. Not a per-row feature.                                                          | train fold        |

**Transform order** (all fit on train, applied to every fold, params saved to
`meta/transforms.json` for train/serve parity): impute → skew (`log1p`) → IQR
outliers → scaler. Config lives under `model_build.processing.{impute,skew,outliers}`;
code is `src/fraud_features/transforms.py`, wired in `src/model_build/data_preparation/prepare.py`.

> **Caveat — fraud rows are themselves outliers.** The account-drain pattern makes
> fraudulent `amount`s legitimately extreme, so a naive IQR filter would preferentially
> delete the positive class. Defaults guard against this: fences are fit on the
> **non-fraud** class only (`scope: non_fraud_only`), the default action is `clip`
> (no rows lost, label balance preserved), and `drop` — when chosen — only ever removes
> rows from the **train** fold, never val/test (you evaluate on the real distribution).
> For the v0 XGBoost (tree) model the `log1p` and clip steps are near no-ops anyway
> (trees are monotonic-invariant); they matter if the model is later swapped for a
> linear one, and they keep the emitted datasets model-agnostic.

**Train/val/test split — time-based.** Split by `step`, never randomly. A fraud pair's
`TRANSFER` and its following `CASH_OUT` are close in time; a random split could put one in
train and the other in test and leak the label.

**Point-in-time correctness.** Both the aggregate join and the split exist to guarantee a
training row is only built from information available at authorization time. Anything fit
across the whole dataset (a global scaler, a global mean) would leak test-fold statistics
into training.

---



## 3. Orchestration of the data path

The Glue job runs on a schedule, independent of any training run. AWS Step Functions drives
it (not Airflow/MWAA — see note below).

```text
EventBridge Scheduler (cron)
   └─► Step Functions state machine
         1. StartCrawler        — refresh Glue Catalog partitions on the raw zone  [optional]
         2. Glue StartJobRun (.sync)  — run the ETL / precompute job, wait for completion
         3. Choice on job status      — fail the execution if the Glue DQ gate failed
         4. curated data + account-history aggregates now in S3
         5. SageMaker StartPipelineExecution (.sync)   — kick a training run        [optional]
         6. refresh the serving endpoint's in-memory aggregates table              [optional]
              (Lambda task: update/redeploy the endpoint, or ping its reload hook)
         └─► any failure ─► SNS / EventBridge alert
```

**Glue job configuration**

- Reads raw data via the **Glue Data Catalog** (crawler-cataloged), not a hardcoded schema
— needed to scale to the future 15GB+ dataset with no code change.
- **Job bookmarks enabled** — the raw zone is append-only and the job reruns on a schedule,
so bookmarks stop it reprocessing rows it already handled.
- **Two S3 outputs:**
  - curated row-level Parquet, partitioned (e.g. by ingestion date) → input to Stage 2
  - per-account aggregates Parquet → read by Stage 2 for training **and** loaded into
  memory by the serving endpoint (see companion doc)
- **Glue Catalog updated** on write so new partitions are queryable immediately.
- Sized **G.1X + auto-scaling**.

**Why Step Functions over Airflow/MWAA.** The data path is a short, linear
Glue → (train) → refresh sequence with no cross-DAG dependencies and no backfill needs. A
state machine covers it without running an MWAA environment. Native `.sync` integrations for
Glue and SageMaker remove polling sensors and any REST-trigger Lambda, EventBridge
Scheduler starts the execution directly, and the whole thing stays in the Terraform/IaC
stack. Revisit MWAA only if this grows into many interdependent DAGs or needs
Airflow-specific operators.

---



## Open items

- **Concrete aggregate list.** The batch-precomputed feature list (rolling windows, profile
stats, mule-account linkage) is a first draft, not finalised.
- **Deferred single-snapshot balance features** ([Appendix B](#appendix-b--deferred-single-snapshot-balance-features)) — revisit once real, non-synthetic balance data is available.
- **Tier-2 (beneficiary-side) features** — revisit if/when the deployment target is
confirmed to include same-institution transfers.
- **Scaling the aggregates.** If the per-account aggregates table outgrows what the serving
instance can hold in memory, move it to a DynamoDB / SageMaker Feature Store online store.
Tracked in the companion doc.

---



## Appendix A — why the balance columns are excluded



### Leakage rule

Prediction time **T** = the moment the transaction is submitted for authorization, before it
posts. Any feature only knowable after T is leakage.


| Reference notebook feature                                   | Available at T?                                                      | Decision                                  |
| ------------------------------------------------------------ | -------------------------------------------------------------------- | ----------------------------------------- |
| `oldBalanceOrig`, `oldBalanceDest`                           | Yes (pre-transaction), but **corrupted in this dataset** — see below | drop                                      |
| `newBalanceOrig`, `newBalanceDest`                           | No (post-transaction)                                                | drop                                      |
| `errorBalanceOrig`, `errorBalanceDest` (derived from `new*`) | No                                                                   | drop                                      |
| `isFlaggedFraud`                                             | n/a                                                                  | drop (no discernible logic, 16 / 6M rows) |


The reference notebook's headline AUPRC leans heavily on `errorBalanceOrig`, which is partly
an artifact of how the PaySim simulator fails to update destination balances on fraud. It
would not be legally usable at auth time and may not generalize to real data anyway.

### Why `oldBalanceOrig` / `oldBalanceDest` are dropped too (not just a leakage call)

Unlike `new*`, these are pre-transaction and so are not *temporally* leaky. But
[dataset-info.md](dataset-info.md) gives two independent reasons to distrust them here:

1. **The dataset creator's own NOTE** names all four balance columns and says they "must not
  be used" for fraud detection, because fraud transactions are cancelled in the simulator.
2. **Documented corruption**, independent of that NOTE: `oldBalanceOrig == amount` is the
  mechanical signature of how fraud rows were generated (drain-the-account), so it
   correlates with the label by construction, not by real-world causality. And ~47% of
   *genuine* transactions also show `oldBalanceOrig == newBalanceOrig == 0` for a nonzero
   amount — the balance was often simply never logged for legitimate accounts, not a true
   zero. The column conflates "real balance" with "unlogged/unknown" for both classes.

Net effect: a model trained on `oldBalanceOrig` / `oldBalanceDest` risks fitting this
simulator's label-construction and logging artifacts rather than a real balance signal — the
same failure mode as `errorBalance*`, just less obvious.

### Why Tier-2 (beneficiary-side) features are dropped for now

Only the transacting account's own data is used — this is the one thing an issuer/PSP always
knows about itself, regardless of who the money is going to or which bank they are at.
Beneficiary-side features (e.g. `oldBalanceDest`) only exist when the recipient is on the
same ledger, which does not hold in general. Dropped until the deployment target is confirmed
same-institution-only.

---



## Appendix B — deferred: single-snapshot balance features

**Not built in v0.** Parked for a later pass once real (non-synthetic) balance data is
available. In this dataset `oldBalanceOrig` is corrupted / label-constructed (Appendix A),
so building on it now would just teach the model the simulator's artifacts.

When revisited: put it behind a Processing-step feature flag (e.g.
`include_balance_features`) so it can be turned on without touching the Glue job, and
re-evaluate whether it actually adds signal.


| Feature               | Definition                         |
| --------------------- | ---------------------------------- |
| `oldBalanceOrig`      | payer balance before txn           |
| `amountToBalanceOrig` | `amount / (oldBalanceOrig + 1)`    |
| `drainsAccountOrig`   | `abs(amount - oldBalanceOrig) < 1` |
| `overdraft`           | `amount > oldBalanceOrig`          |
| `zeroBalanceOrig`     | `oldBalanceOrig == 0`              |


