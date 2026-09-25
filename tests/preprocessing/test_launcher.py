"""Unit tests for run identity (``lineage.py``) and the launcher's deps packaging."""

import importlib
import io
import sys
import zipfile

import pytest
from botocore.exceptions import ClientError

from src.preprocessing import lineage
from src.preprocessing import run_sagemaker_preprocessing_job as launcher


def test_run_id_includes_data_version_and_dirty_suffix() -> None:
    """Different data versions or dirty edits must yield different run ids."""
    clean = lineage.build_run_id("abc123", "cfg", "data-v0")
    assert clean == "abc123-cfg-data-v0"
    assert lineage.build_run_id("abc123", "cfg", "data-v1") != clean
    assert lineage.build_run_id("abc123", "cfg", "data-v0", "deadbeef") == f"{clean}-dirty-deadbeef"


def test_config_hash_is_key_order_independent() -> None:
    """The hash is computed on canonical JSON so key order does not matter."""
    assert lineage.config_hash({"a": 1, "b": 2}) == lineage.config_hash({"b": 2, "a": 1})


def test_deps_zip_is_deterministic_and_importable(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The zip is byte-stable and ``spark_job``'s fallback imports resolve from it."""
    first, second = launcher.build_deps_zip(), launcher.build_deps_zip()
    assert first == second
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert sorted(archive.namelist()) == sorted(launcher.DEPS_MODULES)

    zip_path = tmp_path / launcher.DEPS_ZIP_NAME
    zip_path.write_bytes(first)
    monkeypatch.syspath_prepend(str(zip_path))
    monkeypatch.delitem(sys.modules, "lineage", raising=False)
    assert importlib.import_module("lineage").config_hash({"a": 1}) == lineage.config_hash({"a": 1})


class _FakeS3:
    """Minimal S3 stub whose ``put_object`` raises a configurable error."""

    def __init__(self, code: str | None) -> None:
        """Stores the error code to raise, or ``None`` for success."""
        self.code = code

    def put_object(self, **kwargs: object) -> None:
        """Raises ``ClientError`` with the configured code, else succeeds."""
        if self.code:
            raise ClientError({"Error": {"Code": self.code, "Message": "x"}}, "PutObject")


def test_put_immutable_refuses_existing_key() -> None:
    """A failed conditional write becomes a clean SystemExit; other errors propagate."""
    launcher.put_immutable(_FakeS3(None), "b", "k", b"x")
    with pytest.raises(SystemExit):
        launcher.put_immutable(_FakeS3("PreconditionFailed"), "b", "k", b"x")
    with pytest.raises(ClientError):
        launcher.put_immutable(_FakeS3("AccessDenied"), "b", "k", b"x")
