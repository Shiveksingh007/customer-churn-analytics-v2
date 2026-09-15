# Databricks notebook source
# MAGIC %md
# MAGIC # Machine Learning Pipeline
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC Prepare the Gold dataset for machine learning by selecting features and creating an automated PySpark ML pipeline.

# COMMAND ----------

from pyspark.ml import Pipeline

from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)

from pyspark.ml.classification import (
    RandomForestClassifier
)

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

from pyspark.sql import functions as F

# COMMAND ----------

gold_df = spark.table(
    "workspace.default.customer_churn_gold"
)

display(gold_df.limit(5))

# COMMAND ----------

print("="*60)
print(" GOLD DATASET READY ")
print("="*60)

print(f"Rows    : {gold_df.count()}")
print(f"Columns : {len(gold_df.columns)}")

# COMMAND ----------

label_column = "Churn"

# COMMAND ----------

categorical_columns = [
    "Contract",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "PaymentMethod",
    "HasFamily",
    "IsLongTermContract"
]

# COMMAND ----------

numeric_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "TotalServicesSubscribed"
]

# COMMAND ----------

indexers = [

    StringIndexer(

        inputCol=column,

        outputCol=f"{column}_Index",

        handleInvalid="keep"

    )

    for column in categorical_columns

]

# COMMAND ----------

from pyspark.ml import Pipeline

index_pipeline = Pipeline(stages=indexers)

index_model = index_pipeline.fit(gold_df)

indexed_df = index_model.transform(gold_df)

# COMMAND ----------

display(
    indexed_df.select(
        "Contract",
        "Contract_Index",
        "InternetService",
        "InternetService_Index"
    ).limit(5)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Onehotencoder

# COMMAND ----------

encoder = OneHotEncoder(

    inputCols=[
        f"{column}_Index"
        for column in categorical_columns[:-1]
    ],

    outputCols=[
        f"{column}_Encoded"
        for column in categorical_columns[:-1]
    ]

)

# COMMAND ----------

# phase 3
numeric_columns = [

    "SeniorCitizen",

    "tenure",

    "MonthlyCharges",

    "TotalCharges",

    "TotalServicesSubscribed"

]

# COMMAND ----------

encoded_columns = [

    f"{column}_Encoded"

    for column in categorical_columns[:-1]

]

# COMMAND ----------

assembler = VectorAssembler(

    inputCols=encoded_columns + numeric_columns,

    outputCol="features"

)

# COMMAND ----------

# Phase 4 — Label
label_indexer = StringIndexer(

    inputCol="Churn",

    outputCol="label"

)

# COMMAND ----------

preprocessing_pipeline = Pipeline(

    stages=

        indexers +

        [encoder,

         assembler,

         label_indexer]

)

# COMMAND ----------

preprocessing_model = preprocessing_pipeline.fit(gold_df)

# COMMAND ----------

spark.version

# COMMAND ----------

gold_df.count()

# COMMAND ----------

spark.conf.get("spark.databricks.clusterUsageTags.clusterWorkers", "unknown")