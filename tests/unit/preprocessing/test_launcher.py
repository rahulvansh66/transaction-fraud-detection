# =======================================================================
# Rahul's AI Lab · Fraud Detection · Production-grade ML pipelines on AWS
# =======================================================================

"""Unit tests for the launcher's deps packaging and immutable S3 writes."""

import importlib
import io
import sys
import zipfile

import pytest
from botocore.exceptions import ClientError

from src.preprocessing import lineage
from src.preprocessing import run_preprocessing_job as launcher


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


class _ListingS3:
    """S3 stub that serves a fixed ``list_objects_v2`` page and records ``put_object`` calls."""

    def __init__(self, objects: list[dict]) -> None:
        """Stores the objects to list.

        Args:
            objects: Entries shaped like ``list_objects_v2`` ``Contents`` items.
        """
        self.objects, self.puts = objects, []

    def get_paginator(self, name: str) -> "_ListingS3":
        """Returns itself as the paginator."""
        return self

    def paginate(self, **kwargs: object) -> list[dict]:
        """Returns one page with the stored objects."""
        return [{"Contents": self.objects}]

    def put_object(self, **kwargs: object) -> None:
        """Records the upload."""
        self.puts.append(kwargs)


def test_raw_manifest_is_sorted_and_stable() -> None:
    """The manifest lists each raw object by key with size and ETag, independent of listing order."""
    a = {"Key": "raw/v0/day=0/a.parquet", "Size": 10, "ETag": '"e1"'}
    b = {"Key": "raw/v0/day=1/b.parquet", "Size": 20, "ETag": '"e2"'}
    folder = {"Key": "raw/v0/day=1/", "Size": 0, "ETag": '"x"'}
    first = launcher.build_raw_manifest(_ListingS3([b, folder, a]), "bkt", "raw/v0/")
    assert [o["key"] for o in first["objects"]] == [a["Key"], b["Key"]]
    assert first["object_count"] == 2 and first["total_bytes"] == 30 and first["objects"][0]["etag"] == "e1"
    assert first == launcher.build_raw_manifest(_ListingS3([a, b]), "bkt", "raw/v0/")


def test_raw_manifest_rejects_empty_prefix_and_is_staged_immutably() -> None:
    """An empty raw prefix aborts the launch; staging uses the write-once helper and returns a hash."""
    with pytest.raises(SystemExit):
        launcher.build_raw_manifest(_ListingS3([]), "bkt", "raw/v0/")
    s3 = _ListingS3([])
    digest = launcher.stage_raw_manifest(s3, "bkt", "processed/r1/", {"objects": []})
    assert len(digest) == 12
    assert s3.puts[0]["Key"] == "processed/r1/config/raw_manifest.json" and s3.puts[0]["IfNoneMatch"] == "*"


def test_put_immutable_refuses_existing_key() -> None:
    """A failed conditional write becomes a clean SystemExit; other errors propagate."""
    launcher.put_immutable(_FakeS3(None), "b", "k", b"x")
    with pytest.raises(SystemExit):
        launcher.put_immutable(_FakeS3("PreconditionFailed"), "b", "k", b"x")
    with pytest.raises(ClientError):
        launcher.put_immutable(_FakeS3("AccessDenied"), "b", "k", b"x")
