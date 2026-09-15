# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer & Feature Engineering
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC Create business-friendly features from the cleaned Silver dataset to improve predictive modeling and business insights.
# MAGIC
# MAGIC Input Table:
# MAGIC workspace.default.customer_churn_silver
# MAGIC
# MAGIC Output Table:
# MAGIC workspace.default.customer_churn_gold

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

silver_df = spark.table(
    "workspace.default.customer_churn_silver"
)

display(silver_df.limit(5))

# COMMAND ----------

print("=" * 60)
print("SILVER LAYER LOADED")
print("=" * 60)

print(f"Rows    : {silver_df.count()}")
print(f"Columns : {len(silver_df.columns)}")

# COMMAND ----------

gold_df = silver_df

# COMMAND ----------

gold_df = gold_df.fillna({
    "TotalCharges": 0.0
})

# COMMAND ----------

print(
    gold_df.filter(
        F.col("TotalCharges").isNull()
    ).count()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Features

# COMMAND ----------

gold_df = gold_df.withColumn(
    "HasFamily",
    F.when(
        (F.col("Partner") == "Yes") |
        (F.col("Dependents") == "Yes"),
        "Yes"
    ).otherwise("No")
)

# COMMAND ----------

display(
    gold_df.groupBy("HasFamily").count()
)

# COMMAND ----------

gold_df = gold_df.withColumn(
    "TenureGroup",
    F.when(F.col("tenure") < 12, "New Customer")
     .when(F.col("tenure") < 36, "Regular Customer")
     .when(F.col("tenure") < 60, "Loyal Customer")
     .otherwise("Very Loyal Customer")
)

# COMMAND ----------

display(
    gold_df.groupBy("TenureGroup").count()
)

# COMMAND ----------

gold_df = gold_df.withColumn(
    "MonthlyChargeLevel",
    F.when(F.col("MonthlyCharges") < 35, "Low")
     .when(F.col("MonthlyCharges") < 70, "Medium")
     .otherwise("High")
)

# COMMAND ----------

display(
    gold_df.groupBy("MonthlyChargeLevel").count()
)

# COMMAND ----------

gold_df = gold_df.withColumn(
    "IsLongTermContract",
    F.when(
        F.col("Contract") == "Month-to-month",
        "No"
    ).otherwise("Yes")
)

# COMMAND ----------

display(
    gold_df.groupBy("IsLongTermContract").count()
)

# COMMAND ----------

gold_df = gold_df.withColumn(
    "HasInternet",
    F.when(
        F.col("InternetService") == "No",
        "No"
    ).otherwise("Yes")
)

# COMMAND ----------

display(
    gold_df.groupBy("HasInternet").count()
)

# COMMAND ----------

gold_df = gold_df.withColumn(
    "TotalServicesSubscribed",
    (
        F.when(F.col("PhoneService") == "Yes", 1).otherwise(0) +
        F.when(F.col("OnlineSecurity") == "Yes", 1).otherwise(0) +
        F.when(F.col("OnlineBackup") == "Yes", 1).otherwise(0) +
        F.when(F.col("DeviceProtection") == "Yes", 1).otherwise(0) +
        F.when(F.col("TechSupport") == "Yes", 1).otherwise(0) +
        F.when(F.col("StreamingTV") == "Yes", 1).otherwise(0) +
        F.when(F.col("StreamingMovies") == "Yes", 1).otherwise(0)
    )
)

# COMMAND ----------

display(
    gold_df.select("TotalServicesSubscribed")
)

# COMMAND ----------

gold_table = "workspace.default.customer_churn_gold"

(
    gold_df.write
    .mode("overwrite")
    .format("delta")
    .saveAsTable(gold_table)
)

print("=" * 60)
print(" GOLD LAYER CREATED SUCCESSFULLY")
print("=" * 60)
print(f"Table : {gold_table}")

# COMMAND ----------

gold_df = spark.table(gold_table)

print("=" * 60)
print(" GOLD TABLE LOADED")
print("=" * 60)
print(f"Rows    : {gold_df.count()}")
print(f"Columns : {len(gold_df.columns)}")

# COMMAND ----------

gold_df.printSchema()

# COMMAND ----------

print("=" * 60)
print(" GOLD LAYER VALIDATION")
print("=" * 60)

print(f"Rows                : {gold_df.count()}")
print(f"Columns             : {len(gold_df.columns)}")
print(f"Duplicate Rows      : {gold_df.count() - gold_df.dropDuplicates().count()}")
print(f"Null TotalCharges   : {gold_df.filter('TotalCharges IS NULL').count()}")