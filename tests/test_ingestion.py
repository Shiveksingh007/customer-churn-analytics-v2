"""Tests for ingestion loaders (Phase 2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from chispa import assert_df_equality

from ingestion.loaders import (
    _infer_format,
    _normalize_github_raw_url,
    _parse_kaggle_slug,
    load_from_upload,
)


def test_infer_format_csv():
    assert _infer_format("data/raw/customers.csv") == "csv"


def test_infer_format_parquet():
    assert _infer_format("data/processed/gold.parquet") == "parquet"


def test_infer_format_unsupported():
    with pytest.raises(ValueError, match="Unsupported"):
        _infer_format("notes.txt")


def test_parse_kaggle_slug_from_url():
    slug = _parse_kaggle_slug("https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
    assert slug == "blastchar/telco-customer-churn"


def test_parse_kaggle_slug_plain():
    assert _parse_kaggle_slug("owner/dataset-name") == "owner/dataset-name"


def test_normalize_github_blob_url():
    url = _normalize_github_raw_url(
        "https://github.com/user/repo/blob/main/data/churn.csv"
    )
    assert url == "https://raw.githubusercontent.com/user/repo/main/data/churn.csv"


def test_load_from_upload_csv(spark, tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "customerID,Churn,tenure\nC001,Yes,12\nC002,No,24\n",
        encoding="utf-8",
    )

    df = load_from_upload(str(csv_path), spark=spark)
    assert df.count() == 2
    assert set(df.columns) == {"customerID", "Churn", "tenure"}


def test_load_from_upload_parquet(spark, tmp_path):
    import pandas as pd

    parquet_path = tmp_path / "sample.parquet"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_parquet(parquet_path)

    df = load_from_upload(str(parquet_path), spark=spark)
    assert_df_equality(df, spark.createDataFrame([(1, "x"), (2, "y")], ["a", "b"]))
