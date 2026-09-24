# Feature Engineering (v1 base model)

The notebook's strongest engineered features, `errorBalanceOrig` and `errorBalanceDest`, are built from `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest` and `newbalanceDest`, which are excluded, so they're out. The features below use only `step`, `type`, `amount`, `nameOrig` and `nameDest`. None have been run on the data yet.

## Scope: only `TRANSFER` and `CASH_OUT`

Fraud occurs only in `TRANSFER` and `CASH_OUT` (per the notebook), so the base model trains and scores on these two types only.

- `PAYMENT`, `CASH_IN` and `DEBIT` have zero fraud, so they add no positive examples, only noise and cost.
- Filtering cuts data volume (roughly 40% of rows in the full dataset, holding 100% of the fraud) and lowers class imbalance.
- **Serving guard:** inference must route other types to a rule (treated as non-fraud) or reject them. The model never saw them in training.

## Features (8)

| # | Feature | Definition | Intuition / example |
|---|---|---|---|
| 1 | `is_transfer` | `1` if `type == TRANSFER`, else `0` (`CASH_OUT`) | The two types have different fraud patterns. Fraud typically moves money by `TRANSFER` to a mule, then `CASH_OUT`. |
| 2 | `amount` | raw amount | Fraud amounts are much larger on average. A `TRANSFER` of 2,000,000 is more suspicious than one of 2,000. |
| 3 | `log_amount` | `log1p(amount)` | Amounts are heavily skewed, so this helps linear and distance-based models. Tree models are unaffected, but it's cheap. |
| 4 | `hour_of_day` | `step % 24` (`step` = 1 hour) | Legit traffic drops sharply at night while fraud stays flat around the clock. `step=27` → hour 3. |
| 5 | `is_night` | `1` if `hour_of_day` is 0–6 | A simple version of #4. A 3am `CASH_OUT` is riskier than the same one at 2pm. |
| 6 | `day_of_month` | `step // 24` | Catches day-level bursts, such as a fraud ring active on specific days. Skip it if it overfits on the small sample. |
| 7 | `orig_txn_count` | number of transactions by `nameOrig` | Fraud originators transact about once, so a one-shot account is suspicious. `C1` appearing once → `1`. |
| 8 | `dest_txn_count` | number of transactions received by `nameDest` | Mule accounts receive money from many victims. A `nameDest` seen 15 times is riskier than one seen once. |

## Notes

- **Leakage:** #7 and #8 must be computed on the **training split only**, or from past transactions only (a window over `step`), then applied to validation and test data. Counting over the full dataset leaks future information. Decide this before writing the PySpark processor.
- **Count scope:** decide whether #7 and #8 count within the filtered set or over all transactions. Counting over all preserves more signal but means filtering happens after feature computation.
- **Dropped from earlier draft:**
  - `dest_is_merchant` — merchant (`M…`) destinations appear almost only in `PAYMENT`, so it is redundant after filtering. Re-check if the scope changes.
  - `amount_vs_type_median` — only two types remain, so it adds little.
- **Drop from the model input:** raw `nameOrig`, `nameDest`, `type` (replaced by `is_transfer`) and `isFlaggedFraud` (the notebook concludes it is meaningless, with only 16 rows set).
- **Priority if you want fewer:** #1, #2, #4, #7 and #8 are the core. #3 and #5 are cheap refinements, and #6 is optional.
- **Config:** thresholds such as the night hours belong in `config/`, not in code.
