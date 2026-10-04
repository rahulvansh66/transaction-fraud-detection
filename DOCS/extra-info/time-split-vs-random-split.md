# How to prevent data leakage in time-dependent feature engineering like average amount of the last 3 months?

**Short answer.** Only let each row look at the past, and split train and test by time, not randomly.

## Building the feature: only look back

Take "average amount of the last 3 months" for a card. For a transaction on 15 June, the window must cover 15 March up to just before this transaction. Three rules:

- **Sort by timestamp first.** Without sorting, "previous rows" means nothing.
- **Exclude the current row.** If the row's own amount is in the average, its value leaks into its own feature.
- **Never include later rows.** A window centred on the row, or a group average over the whole dataset, quietly uses the future.

In Spark this is a window ordered by time that ends one row before the current one. In pandas it is a rolling window with the current row shifted out.

## After the feature exists: shuffling is fine

The value is now stored in the row and travels with it, so shuffling doesn't change it. XGBoost doesn't care about row order. Shuffling inside the training set is harmless.

## The split: this is where leakage sneaks back in

Even with a correct feature, a random split gives a dishonest test score. With data from January to June, a random split puts some March rows in train and some February rows in test, so the model has already seen the future of its test rows. In production we always train on the past and predict what comes next, so the live model will do worse than the test score promised.

So split by time: oldest data for train, next slice for validation, newest for test.

## Watch the edges

- **Start of the dataset.** The first 3 months have incomplete windows. Don't pretend they're full. Either drop those rows or mark the feature as missing, and do the same in production.
- **Window crossing the split.** A test row in July may use a window reaching back into training months. That's fine, because it's still the past. The reverse is not fine: a train row must never use test-period data.
- **Label-based features.** Example: "did this card have fraud before?" A fraud on 8 June is only reported on 20 June. In our data it is already marked fraud, so the 10 June row gets "yes". In real life, on 10 June nobody knew yet. Avoid such features, or ignore labels from the last 30 days.

**Rule of thumb:** each row sees only what existed at its own timestamp, and the test set must be newer than the training set.

---

> Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
