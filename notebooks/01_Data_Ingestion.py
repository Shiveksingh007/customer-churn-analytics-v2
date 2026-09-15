# Databricks notebook source
# MAGIC %md
# MAGIC ### ==========================================================
# MAGIC ### Project     : Customer Churn Analytics Platform
# MAGIC ### Notebook    : `01_Data_Ingestion`
# MAGIC ### Layer       : Bronze
# MAGIC ### Author      : Shivek Singh
# MAGIC ### Created On  : 03-07-2026
# MAGIC #
# MAGIC ### Purpose:
# MAGIC ### Read the raw customer churn dataset from Unity Catalog
# MAGIC ### Volume, validate the ingestion, and save it as a Bronze
# MAGIC ### Delta table.
# MAGIC ### ==========================================================

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Ingestion

# COMMAND ----------

RAW_DATA_PATH = "/Volumes/workspace/default/customer_churn_data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

BRONZE_TABLE = "workspace.default.customer_churn_bronze"

# COMMAND ----------

# ==========================================================
# Add Project src Directory to Python Path
# ==========================================================

import sys

PROJECT_ROOT = "/Workspace/Users/thakurshivek777@gmail.com/Customer-Churn-Platform"
SRC_PATH = f"{PROJECT_ROOT}/src"

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

print("Project src added successfully!")

# COMMAND ----------

from config.settings import PROJECT_NAME

print(PROJECT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Dataset

# COMMAND ----------

customer_df = (
    spark.read
         .format("csv")
         .option("header", True)
         .option("inferSchema", True)
         .load(RAW_DATA_PATH)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate Dataset

# COMMAND ----------

print("=" * 50)
print("Customer Churn Dataset - Initial Validation")
print("=" * 50)

print(f"Total Rows    : {customer_df.count()}")
print(f"Total Columns : {len(customer_df.columns)}")

# COMMAND ----------

customer_df.printSchema()

# COMMAND ----------

display(customer_df.limit(10))

# COMMAND ----------

total_rows = customer_df.count()
unique_rows = customer_df.dropDuplicates().count()

print(f"Total Rows      : {total_rows}")
print(f"Unique Rows     : {unique_rows}")
print(f"Duplicate Rows  : {total_rows - unique_rows}")

# COMMAND ----------

from pyspark.sql.functions import col, count, when

missing_df = customer_df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in customer_df.columns
])

display(missing_df)

# COMMAND ----------

customer_df.groupBy("Churn").count().show()

# COMMAND ----------

customer_df.describe().show()

# COMMAND ----------

display(customer_df.dtypes)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Bronze Layer

# COMMAND ----------

(
    customer_df.write
    .mode("overwrite")
    .saveAsTable(BRONZE_TABLE)
)

# COMMAND ----------

spark.read.table(BRONZE_TABLE).show(5)

# COMMAND ----------

bronze_df = spark.read.table(BRONZE_TABLE)

display(bronze_df.limit(10))

# COMMAND ----------

print(f"Rows : {bronze_df.count()}")
print(f"Columns : {len(bronze_df.columns)}")

# COMMAND ----------

print("=" * 70)
print("BRONZE LAYER CREATED SUCCESSFULLY")
print("=" * 70)

print(f"Source File    : {RAW_DATA_PATH}")
print(f"Bronze Table   : {BRONZE_TABLE}")
print(f"Rows Loaded    : {bronze_df.count()}")
print(f"Columns Loaded : {len(bronze_df.columns)}")
print("Status         : SUCCESS")

# COMMAND ----------

bronze_df.count()

# COMMAND ----------

row_count = bronze_df.count()
column_count = len(bronze_df.columns)

# COMMAND ----------

print(f"Rows Loaded    : {row_count}")
print(f"Columns Loaded : {column_count}")