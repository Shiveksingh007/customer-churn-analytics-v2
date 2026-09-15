"""
===========================================================
Project : Customer Churn Analytics Platform
Module  : Ingestion Loaders
Author  : Shivek Singh
===========================================================

Bring-your-own-dataset loaders for CSV, Parquet, and JSON files from
local paths, raw URLs (GitHub), and Kaggle datasets.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".json", ".jsonl"}
KAGGLE_DATASET_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?kaggle\.com/datasets/(?P<slug>[\w\-]+/[\w\-]+)",
    re.IGNORECASE,
)
GITHUB_BLOB_URL_RE = re.compile(
    r"https?://github\.com/(?P<user>[\w\-.]+)/(?P<repo>[\w\-.]+)/blob/(?P<branch>[^/]+)/(?P<path>.+)",
    re.IGNORECASE,
)
DATETIME_NAME_HINTS = ("date", "time", "timestamp", "created", "updated", "dob")


def _get_spark(spark: SparkSession | None = None) -> SparkSession:
    if spark is not None:
        return spark
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


def _infer_format(file_path: str | Path) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".parquet":
        return "parquet"
    if suffix in {".json", ".jsonl"}:
        return "json"
    raise ValueError(
        f"Unsupported file type '{suffix}'. "
        f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def _read_tabular_file(
    spark: SparkSession,
    file_path: str | Path,
    file_format: str | None = None,
) -> DataFrame:
    path = str(file_path)
    fmt = file_format or _infer_format(path)

    if fmt == "csv":
        return (
            spark.read.format("csv")
            .option("header", True)
            .option("inferSchema", True)
            .load(path)
        )

    if fmt == "parquet":
        return spark.read.parquet(path)

    if fmt == "json":
        return spark.read.option("multiLine", True).json(path)

    raise ValueError(f"Unsupported format: {fmt}")


def _find_tabular_file(directory: Path, preferred_name: str | None = None) -> Path:
    if preferred_name:
        candidate = directory / preferred_name
        if candidate.is_file():
            return candidate

    for pattern in ("*.csv", "*.parquet", "*.json", "*.jsonl"):
        matches = sorted(directory.rglob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"No supported tabular file found under {directory}. "
        f"Expected one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def _download_url_to_temp(url: str, suffix: str = ".csv") -> Path:
    request = Request(url, headers={"User-Agent": "Customer-Churn-Platform/1.0"})
    with urlopen(request, timeout=120) as response:
        content = response.read()

    temp_file = Path(tempfile.mkstemp(suffix=suffix)[1])
    temp_file.write_bytes(content)
    return temp_file


def _normalize_github_raw_url(url: str) -> str:
    match = GITHUB_BLOB_URL_RE.match(url.strip())
    if not match:
        return url

    user = match.group("user")
    repo = match.group("repo")
    branch = match.group("branch")
    path = match.group("path")
    return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"


def _parse_kaggle_slug(url_or_slug: str) -> str:
    text = url_or_slug.strip().rstrip("/")
    match = KAGGLE_DATASET_URL_RE.search(text)
    if match:
        return match.group("slug")
    if re.fullmatch(r"[\w\-]+/[\w\-]+", text):
        return text
    raise ValueError(
        "Expected a Kaggle dataset slug 'owner/dataset-name' or a "
        "kaggle.com/datasets/... URL."
    )


def _download_with_kaggle_cli(dataset_slug: str, target_dir: Path) -> Path:
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(target_dir), "--unzip"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Kaggle CLI download failed. Ensure the Kaggle API token is configured "
            f"at ~/.kaggle/kaggle.json.\n{result.stderr or result.stdout}"
        )
    return _find_tabular_file(target_dir)


def load_from_upload(
    file_path: str,
    spark: SparkSession | None = None,
    file_format: str | None = None,
) -> DataFrame:
    """
    Load a tabular dataset from a local file path.

    Supports CSV, Parquet, and JSON/JSONL. Format is inferred from the
    file extension unless ``file_format`` is provided.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    return _read_tabular_file(_get_spark(spark), path, file_format=file_format)


def load_from_url(
    url: str,
    spark: SparkSession | None = None,
    file_format: str | None = None,
) -> DataFrame:
    """
    Load a tabular dataset from a remote URL.

    Supports raw GitHub CSV/Parquet/JSON URLs, GitHub ``/blob/`` links
    (auto-converted to raw URLs), and Kaggle dataset page URLs.
    """
    normalized = url.strip()
    if KAGGLE_DATASET_URL_RE.search(normalized):
        return load_from_kaggle(_parse_kaggle_slug(normalized), spark=spark)

    normalized = _normalize_github_raw_url(normalized)
    suffix = Path(urlparse(normalized).path).suffix or ".csv"
    if suffix.lower() not in SUPPORTED_EXTENSIONS:
        suffix = ".csv"

    temp_file = _download_url_to_temp(normalized, suffix=suffix)
    try:
        return _read_tabular_file(
            _get_spark(spark),
            temp_file,
            file_format=file_format or _infer_format(temp_file),
        )
    finally:
        temp_file.unlink(missing_ok=True)


def load_from_kaggle(
    dataset_slug: str,
    spark: SparkSession | None = None,
    file_name: str | None = None,
) -> DataFrame:
    """
    Load a tabular file from a Kaggle dataset.

    Uses ``kagglehub`` when available, otherwise falls back to the ``kaggle`` CLI.
    ``dataset_slug`` should be ``owner/dataset-name`` or a Kaggle datasets URL.
    """
    slug = _parse_kaggle_slug(dataset_slug)
    download_dir: Path | None = None

    try:
        import kagglehub

        download_dir = Path(kagglehub.dataset_download(slug))
        tabular_path = _find_tabular_file(download_dir, preferred_name=file_name)
        return _read_tabular_file(_get_spark(spark), tabular_path)
    except ImportError:
        temp_root = Path(tempfile.mkdtemp(prefix="kaggle_download_"))
        try:
            tabular_path = _download_with_kaggle_cli(slug, temp_root)
            return _read_tabular_file(_get_spark(spark), tabular_path)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
