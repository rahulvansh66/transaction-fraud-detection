# =======================================================================
# Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
# =======================================================================

"""Lineage helpers for training runs (code, config and data identifiers).

Reuses the preprocessing hash logic so preprocessing and training compute identical
identifiers. Used by the launcher and ``train.py`` to tag every MLflow run.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from src.preprocessing.lineage import build_run_id, config_hash

__all__ = ["build_run_id", "config_hash", "current_git_sha", "full_git_sha"]

REPO_ROOT = Path(__file__).resolve().parents[2]


def full_git_sha() -> str:
    """Returns the full 40-character commit sha (the short sha can become ambiguous).

    Returns:
        The full commit identifier, or ``"unknown"`` if git is unavailable.
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def current_git_sha() -> str:
    """Returns the short git sha, suffixed ``-dirty-<hash>`` when the tree has uncommitted changes.

    Returns:
        The commit identifier, or ``"unknown"`` if git is unavailable.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    if not status:
        return sha
    return f"{sha}-dirty-{hashlib.sha256(status.encode()).hexdigest()[:8]}"
