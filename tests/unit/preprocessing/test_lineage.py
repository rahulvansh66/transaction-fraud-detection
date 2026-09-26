"""Unit tests for run identity helpers in ``src/preprocessing/lineage.py``."""

from src.preprocessing import lineage


def test_run_id_includes_data_version_and_dirty_suffix() -> None:
    """Different data versions or dirty edits must yield different run ids."""
    clean = lineage.build_run_id("abc123", "cfg", "data-v0")
    assert clean == "abc123-cfg-data-v0"
    assert lineage.build_run_id("abc123", "cfg", "data-v1") != clean
    assert lineage.build_run_id("abc123", "cfg", "data-v0", "deadbeef") == f"{clean}-dirty-deadbeef"


def test_config_hash_is_key_order_independent() -> None:
    """The hash is computed on canonical JSON so key order does not matter."""
    assert lineage.config_hash({"a": 1, "b": 2}) == lineage.config_hash({"b": 2, "a": 1})
