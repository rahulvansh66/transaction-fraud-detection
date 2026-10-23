# =======================================================================
# Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
# =======================================================================

"""Data loading for training: reads processed parquet splits, their metadata and a content digest."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

METADATA_FILE = "metadata.json"
DIGEST_CHUNK_BYTES = 1 << 20


def directory_digest(path: str | Path) -> str:
    """Hashes the exact bytes of a split so a run records which data it really read.

    Hashes raw file bytes (not a DataFrame), so the digest does not depend on the pandas
    version and cannot be fooled by a sample. Files are visited in sorted relative-path
    order and each name is hashed with its content, so renames and reordering change it.

    Args:
        path: A parquet file or a directory of part files.

    Returns:
        Hex SHA-256 digest.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"cannot digest missing path: {root}")
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode() if root.is_file() else file.relative_to(root).as_posix().encode())
        with file.open("rb") as handle:
            while chunk := handle.read(DIGEST_CHUNK_BYTES):
                digest.update(chunk)
    return digest.hexdigest()


def load_split(
    path: str | Path, features: list[str], label: str
) -> tuple[pd.DataFrame, np.ndarray]:
    """Reads one parquet split and separates features from the label.

    Args:
        path: Directory (or file) holding the split's parquet part files.
        features: Ordered feature column names to select.
        label: Label column name.

    Returns:
        Tuple of (feature DataFrame, label array).

    Raises:
        KeyError: If a requested column is missing.
    """
    df = pd.read_parquet(path)
    return df[features], df[label].to_numpy()


def read_metadata(path: str | Path) -> dict[str, Any]:
    """Reads ``metadata.json`` written by the preprocessing job.

    Args:
        path: Directory containing ``metadata.json`` (the run root, or a split dir
            whose parent holds it).

    Returns:
        Parsed metadata (features, label, ``scale_pos_weight``, ...).

    Raises:
        FileNotFoundError: If no metadata file is found.
    """
    path = Path(path)
    for candidate in (path / METADATA_FILE, path.parent / METADATA_FILE):
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"{METADATA_FILE} not found under {path} or its parent.")
