About Dataset
Context
There is a lack of public available datasets on financial services and specially in the emerging mobile money transactions domain. Financial datasets are important to many researchers and in particular to us performing research in the domain of fraud detection. Part of the problem is the intrinsically private nature of financial transactions, that leads to no publicly available datasets.

We present a synthetic dataset generated using the simulator called PaySim as an approach to such a problem. PaySim uses aggregated data from the private dataset to generate a synthetic dataset that resembles the normal operation of transactions and injects malicious behaviour to later evaluate the performance of fraud detection methods.

Content
PaySim simulates mobile money transactions based on a sample of real transactions extracted from one month of financial logs from a mobile money service implemented in an African country. The original logs were provided by a multinational company, who is the provider of the mobile financial service which is currently running in more than 14 countries all around the world.

This synthetic dataset is scaled down 1/4 of the original dataset and it is created just for Kaggle.

NOTE: Transactions which are detected as fraud are cancelled, so for fraud detection these columns (oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest ) must not be used.
Headers
This is a sample of 1 row with headers explanation:

1,PAYMENT,1060.31,C429214117,1089.0,28.69,M1591654462,0.0,0.0,0,0

step - maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation).

type - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.

amount -
amount of the transaction in local currency.

nameOrig - customer who started the transaction

oldbalanceOrg - initial balance before the transaction

newbalanceOrig - new balance after the transaction.

nameDest - customer who is the recipient of the transaction

oldbalanceDest - initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants).

newbalanceDest - new balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants).

isFraud - This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.

isFlaggedFraud - The business model aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer more than 200.000 in a single transaction.

---

## Additional notes (things to understand before modelling)

### Size and shape
- Full Kaggle file (`PS_20174392719_1491204439457_log.csv`) has ~6.36M rows, 11 columns, ~470 MB. Plan for out-of-core / chunked reads or Spark; do not assume it fits comfortably in memory on a small instance.
- `step` runs 1..744 (30 days x 24 h). It is the only time axis - there are no real timestamps, dates, or timezones.

### Severe class imbalance
- `isFraud == 1` is ~8,213 rows = **~0.13%** of the data. Accuracy is meaningless here.
- Use PR-AUC / precision-recall, recall at a fixed precision, or cost-based metrics. ROC-AUC looks deceptively high on this data.
- Stratify every split on `isFraud`. Apply resampling (SMOTE / undersampling / class weights) **inside** CV folds only, never before splitting.

### Fraud only occurs in two transaction types
- Every `isFraud == 1` row is either `TRANSFER` or `CASH_OUT`. `PAYMENT`, `CASH_IN`, `DEBIT` have zero fraud.
- Common simplification: filter to `TRANSFER` + `CASH_OUT` before training. Document this assumption - it will not hold if the real production data ever has fraud in other types.

### The classic fraud pattern (account takeover)
- Fraud comes in pairs: a `TRANSFER` that empties an account, immediately followed by a `CASH_OUT` of a similar amount, often within the same or next `step`.
- The `nameDest` of the fraudulent `TRANSFER` frequently becomes the `nameOrig` of the following `CASH_OUT` (a mule account). Linking these is a strong signal but requires sequential / graph features, not row-independent features.

### Balance columns are leaky (see also the NOTE above)
- For fraud rows the balances are internally inconsistent: origin is drained to exactly 0 (`oldbalanceOrg == amount`, `newbalanceOrig == 0`), and destination balances are often `0.0 / 0.0` even for large amounts.
- `newbalanceOrig` / `newbalanceDest` are **post-transaction** state - using them is temporal leakage (you score a transaction before it settles).
- Engineered "error" features (`oldbalanceOrg - amount - newbalanceOrig`, `newbalanceDest - oldbalanceDest - amount`) capture the non-reconciliation pattern and hugely boost offline scores, but that lift is mostly a PaySim artifact and will not transfer to real data. Gate them behind a feature flag.

### `isFlaggedFraud` is nearly useless as-is
- Only 16 rows in the full dataset are flagged, and the stated rule ("transfer > 200,000") does not even match all of them.
- It catches a negligible fraction of real fraud. Treat it as a weak legacy-rule baseline or drop it - do not use it as a feature (it is a deterministic function of `type` and `amount` and effectively hand-codes a rule).

### Merchants vs customers
- `nameOrig` is always a customer (`C...`). `nameDest` starting with `M...` is a merchant.
- Merchants appear only as destinations of `PAYMENT`, and their `oldbalanceDest` / `newbalanceDest` are always `0.0` (no info) - not real zeros. Do not impute or reason over them.
- Customer IDs are high-cardinality and mostly appear a handful of times; raw ID is not a usable categorical feature. Aggregate per-account behaviour (counts, rolling sums per `step`) instead.

### No missing values, but hidden ones
- The file has no NaNs, but the merchant `0.0` balances and drained-to-0 fraud balances are effectively "unknown" values disguised as zeros.
- `amount == 0` transactions exist and are worth inspecting separately.

### Data-quality quirks
- Duplicate-looking rows exist (same `step`, `type`, `amount`, parties) - decide whether to dedupe.
- Amount distribution is extremely right-skewed; log-transform for linear models.
- `CASH_OUT` and `TRANSFER` dominate total value moved; `PAYMENT` dominates row count.

### Leakage / methodology checklist
- Split by `step` (time-based) if you want a realistic evaluation, not a random split - random splitting lets the model see "future" transactions of the same fraud pair.
- Fit all scalers / encoders / resamplers on the training fold only.
- Keep `nameOrig` / `nameDest` out of the model as raw strings; use them only to build aggregated or graph features, computed with past-only windows.

### Relevance to this project
- Goal is a working end-to-end pipeline, not SOTA metrics. A small, imbalance-aware model on `TRANSFER` + `CASH_OUT` with `type`, `amount` (logged), `step`-derived time features, and optional gated balance-error features is a reasonable v0.
- The tiny hand-made `dataset/raw/data-v0.csv` (see `.tmp/data-v0-synthetic-notes.md`) intentionally uses an unrealistic ~20% fraud rate so both branches of the pipeline are exercised in tests; do not read anything into its statistics.

---

