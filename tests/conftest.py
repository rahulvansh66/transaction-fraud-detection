"""Shared pytest fixtures for the preprocessing tests (one local SparkSession)."""

import os
import sys
from collections.abc import Iterator

import pytest
from pyspark.sql import SparkSession

# Spark launches ``python`` for its workers; on Windows that resolves to the Store
# shortcut, so pin it to the interpreter running the tests.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


def _add_hadoop_bin_to_path() -> None:
    """Puts ``%HADOOP_HOME%\\bin`` on ``PATH`` so Windows can load ``hadoop.dll``.

    Reading parquet on Windows needs the Hadoop native library (``hadoop.dll``) and
    ``winutils.exe``; ``HADOOP_HOME`` alone is not enough because the DLL is resolved
    through ``PATH``. No-op on other platforms, when ``HADOOP_HOME`` is unset, or when
    its ``bin`` directory is missing.
    """
    hadoop_home = os.environ.get("HADOOP_HOME")
    if sys.platform != "win32" or not hadoop_home:
        return
    bin_dir = os.path.join(hadoop_home, "bin")
    if os.path.isdir(bin_dir) and bin_dir not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]


_add_hadoop_bin_to_path()


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    """Provides a small local SparkSession shared by all tests.

    Yields:
        A ``local[2]`` SparkSession with few shuffle partitions and no UI.
    """
    session = (
        SparkSession.builder.master("local[2]")
        .appName("fraud-preprocessing-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
