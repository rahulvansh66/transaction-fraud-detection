# =======================================================================
# Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
# =======================================================================

"""Data loading for training: reads processed parquet splits and their metadata."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

METADATA_FILE = "metadata.json"


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
