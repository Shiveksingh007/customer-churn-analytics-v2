"""Tests for preprocessing / validation pipeline logic."""

from __future__ import annotations

from chispa import assert_df_equality
from pyspark.sql import Row

from preprocessing.cleaning import clean_total_charges
from validation.data_validator import (
    blank_value_count,
    duplicate_count,
    get_column_count,
    get_row_count,
    validate_primary_key,
)


def test_clean_total_charges_blanks_to_null(spark):
    source = spark.createDataFrame(
        [
            Row(customerID="1", TotalCharges="29.85"),
            Row(customerID="2", TotalCharges=""),
            Row(customerID="3", TotalCharges="  "),
        ]
    )
    cleaned = clean_total_charges(source)
    rows = {r.customerID: r.TotalCharges for r in cleaned.collect()}
    assert rows["1"] == 29.85
    assert rows["2"] is None
    assert rows["3"] is None


def test_clean_total_charges_preserves_other_columns(spark):
    source = spark.createDataFrame([Row(customerID="1", TotalCharges="10.5", tenure=12)])
    cleaned = clean_total_charges(source)
    assert_df_equality(cleaned, spark.createDataFrame([Row(customerID="1", TotalCharges=10.5, tenure=12)]))


def test_validator_row_and_column_counts(spark):
    df = spark.createDataFrame([Row(id=1), Row(id=2), Row(id=3)])
    assert get_row_count(df) == 3
    assert get_column_count(df) == 1


def test_validator_duplicate_count(spark):
    df = spark.createDataFrame([Row(k="a"), Row(k="a"), Row(k="b")])
    assert duplicate_count(df) == 1


def test_validator_primary_key(spark):
    df = spark.createDataFrame([Row(customerID="A"), Row(customerID="B")])
    assert validate_primary_key(df, "customerID") is True

    dupes = spark.createDataFrame([Row(customerID="A"), Row(customerID="A")])
    assert validate_primary_key(dupes, "customerID") is False


def test_blank_value_count(spark):
    df = spark.createDataFrame(
        [Row(TotalCharges="10"), Row(TotalCharges=""), Row(TotalCharges="  ")]
    )
    assert blank_value_count(df, "TotalCharges") == 2
