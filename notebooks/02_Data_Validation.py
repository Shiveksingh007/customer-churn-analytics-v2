# Databricks notebook source
# MAGIC %md
# MAGIC ## Customer Churn Analytics Platform
# MAGIC
# MAGIC ### Notebook 02 - Data Validation
# MAGIC
# MAGIC #### Objective
# MAGIC Validate the Bronze dataset before moving to the Silver layer.
# MAGIC
# MAGIC #### Output
# MAGIC Data Quality Report

# COMMAND ----------

# MAGIC %md
# MAGIC #### Imports

# COMMAND ----------

# ==========================================================
# Project Initialization
# ==========================================================

import sys

PROJECT_ROOT = "/Workspace/Users/thakurshivek777@gmail.com/Customer-Churn-Platform"
SRC_PATH = f"{PROJECT_ROOT}/src"

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

print("Project initialized successfully.")

# COMMAND ----------

from pyspark.sql import functions as F

from config.settings import BRONZE_TABLE

from validation.data_validator import (
    get_row_count,
    get_column_count,
    duplicate_count,
    validate_primary_key,
    blank_value_count,
    missing_value_report,
)

# COMMAND ----------

bronze_df = spark.read.table(BRONZE_TABLE)

display(bronze_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Structural Validation

# COMMAND ----------

print("=" * 70)
print("STRUCTURAL VALIDATION")
print("=" * 70)

rows = get_row_count(bronze_df)
columns = get_column_count(bronze_df)
duplicates = duplicate_count(bronze_df)

print(f"Rows       : {rows}")
print(f"Columns    : {columns}")
print(f"Duplicates : {duplicates}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Business Rule Validation

# COMMAND ----------

primary_key_valid = validate_primary_key(
    bronze_df,
    "customerID"
)

print(f"Primary Key Valid : {primary_key_valid}")

# COMMAND ----------

blank_totalcharges = blank_value_count(
    bronze_df,
    "TotalCharges"
)

print(
    f"Blank TotalCharges : {blank_totalcharges}"
)

# COMMAND ----------

print("Distinct Churn Values")

bronze_df.select("Churn").distinct().show()

# COMMAND ----------

print("=" * 70)
print("MISSING VALUE REPORT")
print("=" * 70)

missing_value_report(bronze_df).show()

# COMMAND ----------

validation_report = {
    "Rows": rows,
    "Columns": columns,
    "Duplicate Rows": duplicates,
    "Primary Key Valid": primary_key_valid,
    "Blank TotalCharges": blank_totalcharges,
}

# COMMAND ----------

print("=" * 70)
print("DATA QUALITY REPORT")
print("=" * 70)

for key, value in validation_report.items():
    print(f"{key:<25}: {value}")