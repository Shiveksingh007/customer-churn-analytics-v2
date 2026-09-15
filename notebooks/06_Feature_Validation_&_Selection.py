# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Validation & Selection
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC Evaluate engineered and original features to identify the most informative variables for churn prediction.

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

gold_df = spark.table(
    "workspace.default.customer_churn_gold"
)

display(gold_df.limit(5))

# COMMAND ----------

print("=" * 60)
print(" GOLD DATASET")
print("=" * 60)

print(f"Rows    : {gold_df.count()}")
print(f"Columns : {len(gold_df.columns)}")

# COMMAND ----------

family_churn = (
    gold_df
    .groupBy("HasFamily")
    .agg(
        F.count("*").alias("Customers"),
        F.sum(
            F.when(F.col("Churn") == "Yes",1)
            .otherwise(0)
        ).alias("Churned")
    )
    .withColumn(
        "Churn_Rate",
        F.round(
            (F.col("Churned")/F.col("Customers"))*100,
            2
        )
    )
)

display(family_churn)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business Insight
# MAGIC
# MAGIC ### Question
# MAGIC
# MAGIC Does having a family reduce customer churn?
# MAGIC
# MAGIC ### Evidence
# MAGIC
# MAGIC Customers with family: 19.82% churn
# MAGIC
# MAGIC Customers without family: 34.24% churn
# MAGIC
# MAGIC ### Insight
# MAGIC
# MAGIC Customers who have a partner or dependents are substantially more likely to stay with the company.
# MAGIC
# MAGIC ### Business Impact
# MAGIC
# MAGIC 🔴 High
# MAGIC
# MAGIC ### Recommendation
# MAGIC
# MAGIC Target customers without family through retention campaigns, loyalty benefits, and personalized offers.

# COMMAND ----------

services_summary = (
    gold_df
    .groupBy("TotalServicesSubscribed")
    .agg(
        F.count("*").alias("Customers"),
        F.sum(
            F.when(F.col("Churn")=="Yes",1)
            .otherwise(0)
        ).alias("Churned")
    )
    .withColumn(
        "Churn_Rate",
        F.round(
            (F.col("Churned")/F.col("Customers"))*100,
            2
        )
    )
    .orderBy("TotalServicesSubscribed")
)

display(services_summary)