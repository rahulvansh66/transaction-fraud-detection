# Serving: Real-Time Fraud Scoring Endpoint

How the trained model is served for **real-time transaction authorization** — scoring a
transaction before it settles.

- **Companion doc:** [preprocessing-glue-and-sagemaker.md](preprocessing-glue-and-sagemaker.md)
  — how the model and the account-history aggregates this endpoint reads are built.
- **Target latency:** authorization SLA (well under API Gateway's ~29s timeout).

---

## 1. Architecture

```text
Request-driven (serving) path:

  Client
    │  POST transaction record(s)
    ▼
  API Gateway
    │
    ▼
  Lambda  ──────────────►  SageMaker real-time endpoint  ──►  fraud score(s)
    │                          (model + in-memory
    │                           account-history aggregates)
    │  ◄──────────────────────────────────────────────────────┘
    ▼
  Lambda writes prediction(s) to S3
    │
    ▼
  Response back to client
```

This path has **no cron and no Glue** in it. Data freshness is handled by a completely
separate scheduled path (see companion doc); the two only meet through the aggregates file
in S3 that this endpoint loads into memory.

---

## 2. Request / response shape

- **Input:** a list of `1..N` transaction records in one request. There is **no separate
  single-record vs. batch code path** — the endpoint scores whatever list it is given. This
  covers both "authorize one transaction" and "score a small ad-hoc batch" with the same
  code.
- **Output:** the endpoint returns one prediction per input record. The Lambda then writes
  those predictions to S3 before responding to the client.
- **Row filter:** callers are expected to send only `type in {TRANSFER, CASH_OUT}`
  transactions — the model is only trained on those.

### Practical limit on N

This is a **synchronous real-time call**, so it inherits:

| Limit | Value | Source |
| --- | --- | --- |
| Request timeout | ~29 s | API Gateway |
| Request payload size | ~6 MB | SageMaker endpoint invoke |

Fine for one transaction or a few thousand rows. It is **not** a way to reprocess the full
dataset (millions of rows / 15GB+). If that need ever comes up it is a separate offline job
(e.g. SageMaker Batch Transform), not this endpoint — out of scope for now.

---

## 3. How the endpoint uses account-history aggregates

The model needs each account's history features (rolling txn counts/sums, typical amount,
time since last txn — see companion doc). Those are precomputed by the Glue job and written
to S3 as a per-account table.

**The endpoint does NOT do a per-request `GetObject` against S3.** Reasons:

- S3 is not a point-lookup store — a `GetObject` per authorization is too slow and
  inconsistent for the latency SLA.
- Per-request reads would also hammer S3 under load.

**Instead:** the endpoint container **loads the latest aggregates table into memory on
startup** and refreshes it on the Glue job's schedule. Refresh options (not yet decided —
see Open items):

- reload on a timer / container health-check interval, or
- redeploy / update the endpoint after each Glue run (the scheduled Step Functions path can
  trigger this as its last step).

**This is a real design constraint, not an implementation detail.** It assumes the
per-account aggregates table fits in the serving instance's memory. When it stops fitting,
that is the point to move the aggregates to a **DynamoDB or SageMaker Feature Store online
store** — a real point-lookup store that would then earn its cost.

---

## 4. Output

The Lambda writes every prediction returned by the endpoint to S3 (e.g. partitioned by
date). This is the record of what was scored and what the model said — used for monitoring,
audit, and later label-joining once ground-truth fraud outcomes are known.

---

## Open items

- **Aggregate refresh mechanism** — scheduled in-memory reload vs. endpoint redeploy after
  each Glue run. Not yet decided.
- **Aggregates outgrowing instance memory** — move to DynamoDB / Feature Store online store
  when the in-memory table no longer fits. Trigger and cutover plan not yet defined.
- **Full-dataset reprocessing** — if a need to run the whole dataset through the model shows
  up, design a SageMaker Batch Transform job; do not extend this endpoint.
- **Model refresh** — how a newly trained/approved model gets promoted to this endpoint
  (blue/green, canary) is owned by the deploy pipeline, tracked separately.
