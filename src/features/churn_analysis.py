"""
Business analysis functions for Customer Churn Platform
"""

from pyspark.sql import functions as F


def analyze_churn(df, column_name):
    """
    Calculate churn statistics for any categorical column.

    Parameters
    ----------
    df : Spark DataFrame
        Input dataframe.

    column_name : str
        Column to analyze.

    Returns
    -------
    Spark DataFrame
        Summary containing:
        - Category
        - Total Customers
        - Churned Customers
        - Churn Rate
    """

    result = (
        df
        .groupBy(column_name)
        .agg(
            F.count("*").alias("Total_Customers"),
            F.sum(
                F.when(F.col("Churn") == "Yes", 1).otherwise(0)
            ).alias("Churned_Customers")
        )
        .withColumn(
            "Churn_Rate",
            F.round(
                (
                    F.col("Churned_Customers")
                    / F.col("Total_Customers")
                ) * 100,
                2
            )
        )
        .orderBy(F.desc("Churn_Rate"))
    )

    return result