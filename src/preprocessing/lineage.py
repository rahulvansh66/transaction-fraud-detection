"""Run-identity helpers shared by the preprocessing launcher and the Spark job.

The launcher derives the immutable ``run_id`` and the job logs the same
``config_hash`` into ``metadata.json``; keeping both in one dependency-free module
guarantees the two sides can never compute different identifiers. The module is
shipped to SageMaker inside the versioned deps zip, so it must import only the
standard library.
"""

import hashlib
import json
from typing import Any

HASH_LENGTH = 12


def config_hash(config: dict[str, Any]) -> str:
    """Returns a short stable hash of the config for lineage.

    Args:
        config: Parsed preprocessing configuration.

    Returns:
        First 12 hex characters of the SHA-256 of the canonical JSON.
    """
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:HASH_LENGTH]


def build_run_id(git_sha: str, cfg_hash: str, data_version: str, dirty_hash: str | None = None) -> str:
    """Builds the run identifier that names the immutable ``processed/<run_id>/`` prefix.

    The id pins code (commit), parameters (config hash) and data (raw dataset version),
    so identical inputs deduplicate while any change yields a new prefix.

    Args:
        git_sha: Short git commit sha the job was launched from.
        cfg_hash: Output of :func:`config_hash`.
        data_version: Raw dataset version under ``raw/`` (e.g. ``"data-v0"``).
        dirty_hash: Hash of the uncommitted diff when the git tree is dirty, else ``None``.

    Returns:
        ``<git_sha>-<cfg_hash>-<data_version>`` with a ``-dirty-<dirty_hash>`` suffix for dirty trees.
    """
    run_id = f"{git_sha}-{cfg_hash}-{data_version}"
    return f"{run_id}-dirty-{dirty_hash}" if dirty_hash else run_id
