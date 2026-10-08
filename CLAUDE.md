You are senior machine learning engineer, who knows best practice to build scalable production ml pipelines on AWS. You also make sure that the results are reproducable. 

We are aiming for scalable and production grade sytem having huge training dataset(15GB+), the we are currently using while developing is purposely small to build and test this project working end to end, once we are done then we'll test on full dataset.

## Infrastructure

Terraform is the single source of truth for all AWS resources in this project
(S3 buckets, IAM roles, SageMaker pipelines/endpoints, ECS/Lambda/Glue jobs,
CloudWatch log groups, VPC/networking, etc.). Define and change infrastructure
through Terraform config only — never create or modify AWS resources manually
via the console, ad-hoc CLI commands, or one-off scripts.

## Local-only folders

`.tmp/` and `.ref/` are local scratch and reference folders, not part of the project. Read from them only
when explicitly asked or referenced, and never mention, cite, or document them as a source anywhere in the codebase
(code, docstrings, comments, docs, configs, or notebooks).

## Development-phase compute budget

Until the user says we are now working on the full dataset, we use a small smoke dataset
and keep every SageMaker job as cheap as possible. Rows barely affect cost (startup time
dominates), so the savings come from running fewer jobs and shorter training:

- **Minimal parameters** for smoke experiments:

  | Purpose | `num_round` | `early_stopping_rounds` |
  |---|---|---|
  | Pure plumbing smoke test | 10-20 | 5 |
  | Slightly meaningful | 30-50 | 10 |

  Never copy smoke values into `config/production/model.yaml`.
- **Keep HPO at 3 trials** (`tuning.max_jobs`, or 3 grid combinations).
- **One SageMaker run per stage**, not frequent repeats. Test code and config changes
  locally first (`--mode local`, unit tests). Launch a SageMaker job only after a batch of
  meaningful changes that genuinely need a container run, and ask the user before launching
  one.

## Code style

Write code that is structured, modular, scalable, and maintainable. Use comments
sparingly to separate distinct logical blocks within a function, not to restate
what the code does.

### Docstrings (follow always)

Every module, class, and function/method must have a docstring. No exceptions for
"small" or "obvious" helpers.

1. **Style.** Use Google-style docstrings consistently across the repo.
2. **Content.** A one-line summary of what it does (and *why*, when not obvious).
   For anything non-trivial, add a short description paragraph.
3. **Every argument is documented.** List all parameters under `Args:` with name,
   type, and meaning. Do not leave a docstring that skips or only partially covers
   the arguments — if you touch a function with a partial docstring, complete it.
4. **Also document** `Returns:` (type + meaning), `Raises:` (exceptions the caller
   should expect), and `Yields:` for generators. Omit a section only when it truly
   does not apply (e.g. no return value).
5. **Keep it in sync.** When you change a signature, update the docstring in the
   same edit.
6. Module docstring states the file's purpose and its role in the pipeline.

### Typing and logging

- Add type hints to every function/method signature (params and return type).
  Use `typing`/`|` unions consistently; avoid bare `Any` unless the type is
  genuinely dynamic.
- Use the `logging` module, not `print`, for anything that runs as part of a
  pipeline step (progress, warnings, errors). Reserve `print` for local,
  throwaway scripts only.

### CloudWatch logging (follow always)

All pipeline/production code ships logs to CloudWatch (via SageMaker, ECS,
Lambda, or Glue log drivers). Write logs so they're actually usable once they
land there.

1. **Get a logger per module**, never configure the root logger from library
   code: `logger = logging.getLogger(__name__)`. Only the entry point
   (script/pipeline step's `main`) calls `logging.basicConfig(...)` or sets up
   handlers.
2. **Log levels matter.** `DEBUG` for verbose diagnostics, `INFO` for normal
   progress (step start/end, record counts, artifact paths), `WARNING` for
   recoverable issues, `ERROR` for failures, and use `logger.exception(...)`
   inside `except` blocks so the traceback is captured. Never use `ERROR` for
   expected control flow.
3. **Structure log messages for querying.** CloudWatch Logs Insights parses
   better with consistent key=value fields than free-form prose, e.g.
   `logger.info("step=preprocess status=complete rows=%d duration_s=%.2f", n, dt)`
   instead of `logger.info(f"Done! Processed {n} rows")`. Prefer `%s`-style
   lazy formatting over f-strings in log calls so the string isn't built when
   the level is disabled.
4. **Include run/job identifiers** (SageMaker training job name, pipeline
   execution ID, git commit hash) in log output near the start of a run so
   logs can be correlated back to a specific execution — see
   [[ml-lineage-reproducibility]] for what else must be logged for
   traceability.
5. **Never log secrets or PII.** No raw credentials, tokens, full card numbers,
   or customer PII in log messages — mask or omit them. This applies
   everywhere, but is especially critical for CloudWatch since log groups may
   have broader read access than the originating service.
6. **Respect log retention and volume.** Don't log full DataFrames, large
   arrays, or per-row output in tight loops — log aggregates/samples instead.
   Set explicit CloudWatch Logs retention (via the log group's IaC config, not
   the default "Never expire") appropriate to the environment (short for dev,
   longer for prod/audit trails).

### Jupyter notebooks

- Structure the notebook into clear sections and subsections using markdown
  headings (`#`, `##`, `###`), so the notebook reads top-to-bottom as a
  document, not a scratchpad.
- Every section/subsection should be well documented: a short markdown
  description of what it covers and why it's there.
- Precede every code cell with a markdown cell explaining the purpose of that
  specific cell — what it does and why it's needed — not just what the section
  as a whole is about.
- Code inside cells follows the same code style rules as `.py` files above
  (docstrings for any function/class defined in the notebook, type hints,
  `logging` over `print`, sparse in-code comments for block separation).

## File banner (follow always)

Every new file gets the project banner, so authorship is visible in every file.
The banner text is exactly:

`Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS`

Apply it by file type:

- **Hash-comment files** (`.py`, `.tf`, `.yaml`/`.yml`, `.toml`, Dockerfile,
  shell scripts, `.gitignore`, `.claudeignore`, `.env.example`,
  `requirements.txt`): a boxed 3-line header at the very top, followed by
  exactly one blank line. The `=` borders are as wide as the text line.

  ```
  # =======================================================================
  # Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
  # =======================================================================

  ```

  - In Python, the banner goes above the module docstring, so the docstring
    rule still holds.
  - In files with a shebang (`#!`) or a coding line, the banner goes right
    after them, which must stay first.
  - Skip empty files (e.g. empty `__init__.py`).
- **Markdown** (`.md`): footer only, no header. The file starts with its title.
  The footer is a `---` rule and the banner as a blockquote at the very end of
  the file.

  ```
  # Title
  ...content...

  ---

  > Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
  ```

  Exception: files that need YAML frontmatter on line 1 (e.g.
  `.claude/skills/*/SKILL.md`) get no banner.
- **Jupyter notebooks** (`.ipynb`): the first cell is a markdown cell containing
  the banner text in italics (`*Rahul's AI Lab · …*`), above the notebook title.
- **Data and generated files** (JSON, CSV, Parquet, `mlflow.db`, `uv.lock`,
  `.terraform.lock.hcl`): no banner, since comments would corrupt or be
  overwritten by the tooling.

Never duplicate the banner: if a file already contains it, leave it as is.

---

> Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
