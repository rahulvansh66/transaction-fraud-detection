Both commits are on `feature/training`, and I ran no preprocessing or training jobs.

- `5a8e333` has the lineage and code changes.
- `86093d4` has the Terraform: the bucket policy, the launcher access policy and the CI ECR statement. It's already applied to dev.

I ran the tests against a clean checkout of exactly what was committed: 50 unit tests pass and ruff is clean.

**Your in-progress work is not in either commit.** Three files held both my changes and yours: `train.py`, `run_training_job.py` and `train.yml`. I committed them with your hunks backed out, and your edits are still in the working tree. The rest of your work is also left alone: `experiment_config.py` and its test, the two new experiment YAMLs, `README.md`, the extra-info docs, and your edits to the skills. Your `run_training_job.py` change needs `experiment_config.py`, so commit those together.

**Do I need to run a job to test anything here?** Not for this commit. Unit tests cover the code paths with fake S3 and ECR clients, and the policies are confirmed from the live deny check and the ECR digest lookup. Two things only a real run can show, and neither blocks anything:
1. **Preprocessing launch:** that the launcher's `IfNoneMatch` writes pass the new bucket policy, and that the raw manifest uploads. If a launch is refused, the bucket policy rule is the first suspect.
2. **CI training run:** that `image_digest` comes back as a real digest rather than `unresolved`, and that the new tags and data hashes appear in MLflow.

The first launch of each kind would confirm both, whenever you next run them anyway. A dedicated run isn't needed.

---

