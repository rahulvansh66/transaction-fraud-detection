The split follows one rule: **separate code by why it changes and where it runs.**

**1. [features.py](src/preprocessing/features.py): pure transformations**
- These are functions that take a DataFrame and return a DataFrame. They do no I/O, no argument parsing, no S3 and no MLflow.
- The steps are clean, add past counts, filter, basic features, leakage gate, time split and class weight.
- It is the part most likely to change as the modelling evolves, and it holds the actual fraud logic.
- Because it has no side effects, it can be tested on a tiny local Spark session or pandas frame. That is what [test_features.py](tests/preprocessing/test_features.py) does.
- It is also the part where correctness matters most. The account-count feature has to avoid leaking future rows, and a time-based split is only trustworthy if it can be tested directly.

**2. [spark_job.py](src/preprocessing/spark_job.py): orchestration**
- It parses the config and arguments, reads from the SageMaker paths, calls the feature functions in order, and validates the output.
- It writes the partitioned output plus `metadata.json`, and sets up logging.
- It is the only place that knows it is running as a SageMaker Processing step. It also holds the run identifiers and lineage logging that CLAUDE.md requires.
- Changing the input or output layout, or moving to another runner such as Glue, touches this file and not the feature logic.

**3. [run_preprocessing_job.py](src/preprocessing/run_preprocessing_job.py): launcher**
- It submits the job to SageMaker from the client side: instance type, S3 URIs, and job naming.
- It helps to run job on your laptop or sagemaker or any other machine. It is a thin wrapper that will later become a pipeline step.

**4. Config in YAML, infrastructure in Terraform**
- Hyperparameters and paths such as time-split cutoffs and type filters live in [preprocessing.yaml](config/preprocessing/preprocessing.yaml), not in code. This is the `ml-repo-structure` rule.
- Buckets, roles and log groups live in Terraform.

**5. Tests mirror the code, plus a parity test**
- In unit test, `test_features.py` tests each feature function on its own.
- `test_parity.py` is for integration test, that checks that the PySpark output matches the original notebook's pandas logic. That protects reproducibility during the migration.

**Why it helps at 15GB+ scale**
- The same `features.py` runs unchanged on a local subset or a multi-node cluster. Scaling up only changes the launcher's instance count and the Spark configuration.
- Each layer can be swapped without touching the others. Each also has one clear reason to change and one way to test it.