**Yes, build it next, but commit and verify the current work first.** A pipeline is easiest to build when the steps it wraps already work and don't change much.

**Why now is a good time**
- Preprocessing, training, evaluation, HPO (`hpo_tuner.py`, `select_winner.py`) and the lineage code already exist as separate launchers. The pipeline wires them together and adds little new logic.
- There is no `pipeline.py` yet, and your own skills assume one. The register step needs a `ConditionStep` on evaluation, and retraining needs Pipeline Parameters.
- If you wait, you will keep adding per-script glue (the `run_*_job.py` launchers) that the pipeline would replace.

**Do these first**
1. **Commit the 20+ modified files.** The experiment config, validation, lineage and launcher changes are all uncommitted. A pipeline built on moving interfaces means rework.
2. **Run each stage once end to end through its launcher.** That means preprocessing, then a manual training run (`experiment-002`), then an HPO run (`experiment-003`), then evaluate. A pipeline step failing inside SageMaker is much harder to debug than the same launcher failing on its own.
3. **Settle the preprocessing-to-training contract.** Training should consume an immutable, versioned processed-data snapshot, not "latest". That is the `data.run_id` field in the experiment configs. The pipeline passes it as a Pipeline Parameter, and the lineage rules need it.

**Pipeline shape**
```
Preprocess (Spark Processing)
  → Train (manual mode: one config)  or  Tune (HPO mode: AMT)
  → Evaluate
  → Condition (metric gate)
      ├─ fail → stop
      └─ pass → register in MLflow → manual approval
```
- **Parameters:** only what really changes between runs, such as the dataset or snapshot id, the experiment name and the HPO limits. Hyperparameters stay in the YAML.
- **Scheduled retraining:** it should use `config/production/model.yaml` and skip the Tune step. This follows the "production retraining is not HPO" rule in your experimentation skill. You could make this a parameter, or use two pipeline definitions that share the same steps. I'd start with a mode parameter that switches Train and Tune.
- **Infrastructure:** the pipeline's role, any EventBridge schedule and the log groups go in Terraform, per CLAUDE.md. `pipeline.py` upserts the definition, and GitHub Actions triggers an execution.

**Suggested order**
1. Commit the current work.
2. Run the stages once each.
3. Build the pipeline with Preprocess → Train → Evaluate → Condition (no HPO or register yet).
4. Add Tune.
5. Add register and approval.
6. Add the schedule.

Do you want me to start with step 1 (reviewing the diff and committing) or go straight to a `pipeline.py` skeleton? If you've already run all the stages on AWS, tell me, because that changes the order.