# Databricks notebook source
# MAGIC %md
# MAGIC # Telecom Customer Intelligence Platform
# MAGIC
# MAGIC # Notebook 07 — Machine Learning Pipeline
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC Prepare the Gold Layer dataset for machine learning using PySpark ML.
# MAGIC
# MAGIC ## Input
# MAGIC workspace.default.customer_churn_gold
# MAGIC
# MAGIC ## Output
# MAGIC Machine Learning Ready Dataset

# COMMAND ----------

from pyspark.sql import functions as F

from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)

from pyspark.ml.classification import RandomForestClassifier

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

# COMMAND ----------

gold_table = "workspace.default.customer_churn_gold"

gold_df = spark.table(gold_table)

display(gold_df.limit(5))

# COMMAND ----------

print("="*60)
print(" GOLD DATASET VALIDATION ")
print("="*60)

print(f"Rows    : {gold_df.count()}")
print(f"Columns : {len(gold_df.columns)}")

# COMMAND ----------

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "TotalServicesSubscribed"
]

print("="*60)
print(" NULL VALUE VALIDATION ")
print("="*60)

for col in numeric_features:

    print(

        f"{col:<30}:",

        gold_df.filter(

            F.col(col).isNull()

        ).count()

    )

# COMMAND ----------

gold_df.select(
    numeric_features
).printSchema()

# COMMAND ----------

categorical_features = [

    "Contract",

    "InternetService",

    "OnlineSecurity",

    "TechSupport",

    "PaymentMethod",

    "HasFamily",

    "IsLongTermContract"

]

# COMMAND ----------

numeric_features = [

    "SeniorCitizen",

    "tenure",

    "MonthlyCharges",

    "TotalCharges",

    "TotalServicesSubscribed"

]

# COMMAND ----------

ml_df = gold_df

# COMMAND ----------

indexer_models = {}

for col in categorical_features:

    print(f"Indexing {col}")

    indexer = StringIndexer(

        inputCol=col,

        outputCol=f"{col}_Index",

        handleInvalid="keep"

    )

    model = indexer.fit(ml_df)

    indexer_models[col] = model

    ml_df = model.transform(ml_df)

print()

print("All categorical columns indexed successfully.")

# COMMAND ----------

display(

    ml_df.select(

        "Contract",

        "Contract_Index",

        "InternetService",

        "InternetService_Index",

        "HasFamily",

        "HasFamily_Index"

    ).limit(10)

)

# COMMAND ----------

label_indexer = StringIndexer(

    inputCol="Churn",

    outputCol="label"

)

label_model = label_indexer.fit(ml_df)

ml_df = label_model.transform(ml_df)

# COMMAND ----------

display(

    ml_df.select(

        "Churn",

        "label"

    ).limit(10)

)

# COMMAND ----------

print("="*60)
print(" PART 1 COMPLETED ")
print("="*60)

print(f"Rows    : {ml_df.count()}")

print(f"Columns : {len(ml_df.columns)}")

print("Ready for OneHot Encoding")

# COMMAND ----------

# MAGIC %md
# MAGIC ## One-Hot Encoding
# MAGIC
# MAGIC Convert indexed categorical variables into binary vectors to prevent the model from interpreting category indexes as ordinal values.

# COMMAND ----------

encoder = OneHotEncoder(

    inputCols=[
        f"{col}_Index"
        for col in categorical_features
    ],

    outputCols=[
        f"{col}_Encoded"
        for col in categorical_features
    ]

)

# COMMAND ----------

encoder_model = encoder.fit(ml_df)

ml_df = encoder_model.transform(ml_df)

print("✅ One-Hot Encoding completed successfully.")

# COMMAND ----------

display(

    ml_df.select(

        "Contract",

        "Contract_Index",

        "Contract_Encoded"

    ).limit(10)

)

# COMMAND ----------

encoded_features = [

    f"{col}_Encoded"

    for col in categorical_features

]

# COMMAND ----------

assembler = VectorAssembler(

    inputCols=encoded_features + numeric_features,

    outputCol="features",

    handleInvalid="error"

)

# COMMAND ----------

ml_df = assembler.transform(ml_df)

print("✅ Feature vector created successfully.")

# COMMAND ----------

display(

    ml_df.select(

        "features",

        "label"

    ).limit(10)

)

# COMMAND ----------

final_ml_df = ml_df.select(
    "customerID",
    "Contract",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "PaymentMethod",
    "InternetService",
    "HasFamily",
    "features",
    "label"
)

# COMMAND ----------

print("="*60)
print(" FINAL ML DATASET ")
print("="*60)

print(f"Rows    : {final_ml_df.count()}")

print(f"Columns : {len(final_ml_df.columns)}")

# COMMAND ----------

final_ml_df.groupBy(

    "label"

).count().show()

# COMMAND ----------

print("="*60)
print(" PART 2 COMPLETED ")
print("="*60)

print("ML Dataset is ready for model training.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training
# MAGIC
# MAGIC Train a Random Forest Classifier using the processed machine learning dataset.

# COMMAND ----------

train_df, test_df = final_ml_df.randomSplit(
    [0.8, 0.2],
    seed=42
)

# COMMAND ----------

print("=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print(f"Training Rows : {train_df.count()}")
print(f"Testing Rows  : {test_df.count()}")

# COMMAND ----------

rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=100,
    maxDepth=8,
    seed=42
)

# COMMAND ----------

print("=" * 60)
print("TRAINING RANDOM FOREST MODEL")
print("=" * 60)

rf_model = rf.fit(train_df)

print("✅ Model Training Completed")

# COMMAND ----------

predictions = rf_model.transform(test_df)

print("✅ Predictions Generated")

# COMMAND ----------

display(

    predictions.select(

        "prediction",

        "probability",

        "label"

    ).limit(20)

)

# COMMAND ----------

print("=" * 60)
print("PREDICTION DISTRIBUTION")
print("=" * 60)

predictions.groupBy("prediction").count().show()

# COMMAND ----------

print("=" * 60)
print("ACTUAL LABEL DISTRIBUTION")
print("=" * 60)

predictions.groupBy("label").count().show()

# COMMAND ----------

# ============================================================
# SAVE TRAINED RANDOM FOREST MODEL
# ============================================================

model_path = "/Volumes/workspace/default/customer_churn_data/random_forest_model"

rf_model.write().overwrite().save(model_path)

print("=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)
print(f"Model Path : {model_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Importance Analysis
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC Understand which features contribute the most to customer churn prediction.
# MAGIC
# MAGIC This improves model explainability and provides business insights into
# MAGIC the key factors influencing customer churn.

# COMMAND ----------

feature_columns = assembler.getInputCols()

feature_importance = list(
    zip(
        feature_columns,
        rf_model.featureImportances.toArray().tolist()
    )
)

importance_df = spark.createDataFrame(
    feature_importance,
    ["Feature", "Importance"]
)

importance_df = importance_df.orderBy(
    F.col("Importance").desc()
)

display(importance_df)

# COMMAND ----------

print("=" * 60)
print("TOP 10 MOST IMPORTANT FEATURES")
print("=" * 60)

display(
    importance_df.limit(10)
)

# COMMAND ----------

print("=" * 60)
print("MODEL EXPLAINABILITY")
print("=" * 60)

top_feature = importance_df.first()

print(f"Most Important Feature : {top_feature['Feature']}")
print(f"Importance Score       : {top_feature['Importance']:.4f}")

# COMMAND ----------

print("=" * 60)
print("NOTEBOOK 07 SUMMARY")
print("=" * 60)

print("Gold Dataset Loaded        ✅")
print("Data Validation           ✅")
print("String Indexing           ✅")
print("One-Hot Encoding          ✅")
print("Vector Assembler          ✅")
print("Final ML Dataset          ✅")
print("Train/Test Split          ✅")
print("Random Forest Trained     ✅")
print("Predictions Generated     ✅")
print("Notebook Completed        ✅")

# COMMAND ----------

assert train_df.count() > 0
assert test_df.count() > 0
assert predictions.count() > 0

print("=" * 60)
print("ALL VALIDATIONS PASSED")
print("=" * 60)

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

accuracy = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
).evaluate(predictions)

print(f"Accuracy : {accuracy:.4f}")

# COMMAND ----------

f1 = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
).evaluate(predictions)

print(f"F1 Score : {f1:.4f}")

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

auc = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
).evaluate(predictions)

print(f"ROC AUC : {auc:.4f}")

# COMMAND ----------

print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {auc:.4f}")

print("=" * 60)
print("Notebook 07 Completed Successfully")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC # Business Prediction Report
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC Convert machine learning predictions into actionable business insights.
# MAGIC
# MAGIC Each customer will be assigned:
# MAGIC
# MAGIC - Churn Probability
# MAGIC - Risk Level
# MAGIC - Recommended Action
# MAGIC
# MAGIC This report can be used by the customer retention team.

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

extract_probability = udf(
    lambda x: float(x[1]),
    DoubleType()
)

prediction_report = predictions.withColumn(
    "Churn_Probability",
    extract_probability("probability")
)

# COMMAND ----------

from pyspark.sql.functions import round

prediction_report = prediction_report.withColumn(
    "Churn_Percentage",
    round(
        prediction_report.Churn_Probability * 100,
        2
    )
)

# COMMAND ----------

from pyspark.sql.functions import when

prediction_report = prediction_report.withColumn(

    "Risk_Level",

    when(
        prediction_report.Churn_Probability >= 0.80,
        "High"
    )

    .when(
        prediction_report.Churn_Probability >= 0.60,
        "Medium"
    )

    .otherwise("Low")

)

# COMMAND ----------

prediction_report = prediction_report.withColumn(

    "Recommended_Action",

    when(
        prediction_report.Risk_Level == "High",
        "Immediate Retention Call"
    )

    .when(
        prediction_report.Risk_Level == "Medium",
        "Offer Discount / Bundle Plan"
    )

    .otherwise(
        "Loyalty Program"
    )

)

# COMMAND ----------

display(

    customer_report

    .filter("Risk_Level = 'High'")

    .orderBy("Churn_Percentage", ascending=False)

)

# COMMAND ----------

# ============================================================
# SAVE CUSTOMER PREDICTION REPORT
# ============================================================

customer_report.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.default.customer_churn_predictions")

print("=" * 60)
print("PREDICTION REPORT SAVED SUCCESSFULLY")
print("=" * 60)
print("Table : workspace.default.customer_churn_predictions")

# COMMAND ----------

# ============================================================
# GENERATE PREDICTIONS FOR ALL CUSTOMERS
# ============================================================

all_predictions = rf_model.transform(ml_df)

print("=" * 60)
print("FULL CUSTOMER PREDICTIONS GENERATED")
print("=" * 60)

print(f"Rows : {all_predictions.count()}")

# COMMAND ----------

from pyspark.sql.functions import udf, round, when
from pyspark.sql.types import DoubleType

extract_probability = udf(
    lambda x: float(x[1]),
    DoubleType()
)

prediction_report = (
    all_predictions
    .withColumn(
        "Churn_Probability",
        extract_probability("probability")
    )
    .withColumn(
        "Churn_Percentage",
        round(F.col("Churn_Probability") * 100, 2)
    )
    .withColumn(
        "Risk_Level",
        when(F.col("Churn_Probability") >= 0.80, "High")
        .when(F.col("Churn_Probability") >= 0.60, "Medium")
        .otherwise("Low")
    )
    .withColumn(
        "Recommended_Action",
        when(F.col("Risk_Level") == "High", "Immediate Retention Call")
        .when(F.col("Risk_Level") == "Medium", "Offer Discount / Bundle Plan")
        .otherwise("Loyalty Program")
    )
)

# COMMAND ----------

customer_report = prediction_report.select(
    "customerID",
    "Contract",
    "tenure",
    "MonthlyCharges",
    "Churn_Percentage",
    "Risk_Level",
    "Recommended_Action"
)

display(customer_report)
