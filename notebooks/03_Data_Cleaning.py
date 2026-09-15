# Databricks notebook source
# MAGIC %md
# MAGIC # Customer Churn Analytics Platform
# MAGIC
# MAGIC ## Notebook 03 - Data Cleaning
# MAGIC
# MAGIC ### Objective
# MAGIC
# MAGIC Transform Bronze data into a clean Silver dataset.
# MAGIC
# MAGIC ### Input
# MAGIC
# MAGIC Bronze Table
# MAGIC
# MAGIC ### Output
# MAGIC
# MAGIC Silver Table

# COMMAND ----------

import sys

PROJECT_ROOT = "/Workspace/Users/thakurshivek777@gmail.com/Customer-Churn-Platform"
SRC_PATH = f"{PROJECT_ROOT}/src"

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

print(SRC_PATH)
print(sys.path)

# COMMAND ----------

from pyspark.sql import functions as F

from config.settings import (
    BRONZE_TABLE,
    SILVER_TABLE
)

from validation.data_validator import (
    blank_value_count
)

# COMMAND ----------

bronze_df = spark.read.table(BRONZE_TABLE)

display(bronze_df.limit(5))

# COMMAND ----------

silver_df = bronze_df

# COMMAND ----------

silver_df.filter(
    F.trim(
        F.col("TotalCharges")
    ) == ""
).count()

# COMMAND ----------

silver_df = silver_df.withColumn(
    "TotalCharges",
    F.when(
        F.trim(F.col("TotalCharges")) == "",
        None
    ).otherwise(
        F.col("TotalCharges")
    )
)

# COMMAND ----------

silver_df = silver_df.withColumn(
    "TotalCharges",
    F.col("TotalCharges").cast("double")
)

# COMMAND ----------

cleaning_report = {
    "rows": 7043,
    "null_totalcharges": 11,
    "datatype": "double",
    "status": "SUCCESS"
}

# COMMAND ----------

display(cleaning_report)

# COMMAND ----------

(
    silver_df.write
        .mode("overwrite")
        .saveAsTable(SILVER_TABLE)
)

# COMMAND ----------

silver_check = spark.read.table(
    SILVER_TABLE
)

display(
    silver_check.limit(5)
)

# COMMAND ----------

print("=" * 70)
print("SILVER LAYER CREATED SUCCESSFULLY")
print("=" * 70)

print(f"Rows Loaded    : {silver_check.count()}")
print(f"Columns Loaded : {len(silver_check.columns)}")
print("Layer          : Silver")
print("Status         : SUCCESS")