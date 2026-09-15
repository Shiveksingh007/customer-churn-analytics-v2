"""Silver-layer cleaning transforms (Notebook 03)."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def clean_total_charges(df: DataFrame) -> DataFrame:
    """
    Replace blank TotalCharges strings with null and cast to double.

    Mirrors Notebook 03 Data Cleaning logic for testable reuse in CI.
    """
    if "TotalCharges" not in df.columns:
        return df

    return df.withColumn(
        "TotalCharges",
        F.when(F.trim(F.col("TotalCharges").cast("string")) == "", None).otherwise(
            F.col("TotalCharges")
        ),
    ).withColumn("TotalCharges", F.col("TotalCharges").cast("double"))
