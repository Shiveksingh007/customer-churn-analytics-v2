# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 11 — Model Promotion (Champion / Challenger)
# MAGIC
# MAGIC ## Objective
# MAGIC Promote a newly trained Random Forest to **Production** in MLflow Model Registry
# MAGIC only when its ROC-AUC on the held-out test set beats the current Production model.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Notebook **07** completed (model trained, metrics logged to MLflow)
# MAGIC - MLflow tracking + Model Registry enabled on Databricks
# MAGIC
# MAGIC ## Registry lifecycle
# MAGIC `None` → **Staging** → **Production** → **Archived**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

MODEL_NAME = "customer_churn_rf"
CHALLENGER_RUN_ID = dbutils.widgets.get("challenger_run_id")  # noqa: F821
CHALLENGER_AUC = float(dbutils.widgets.get("challenger_auc"))  # noqa: F821
MIN_IMPROVEMENT = float(dbutils.widgets.get("min_auc_improvement") or "0.0")  # noqa: F821

print("=" * 60)
print("MODEL PROMOTION GATE")
print("=" * 60)
print(f"Model name      : {MODEL_NAME}")
print(f"Challenger run  : {CHALLENGER_RUN_ID}")
print(f"Challenger AUC  : {CHALLENGER_AUC:.4f}")
print(f"Min improvement : {MIN_IMPROVEMENT:.4f}")

# COMMAND ----------

import sys

PROJECT_ROOT = "/Workspace/Users/thakurshivek777@gmail.com/Customer-Churn-Platform"
SRC_PATH = f"{PROJECT_ROOT}/src"
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from mlops.model_registry import promote_model_if_better

# COMMAND ----------

decision = promote_model_if_better(
    model_name=MODEL_NAME,
    challenger_run_id=CHALLENGER_RUN_ID,
    challenger_auc=CHALLENGER_AUC,
    min_improvement=MIN_IMPROVEMENT,
)

# COMMAND ----------

print("=" * 60)
print("PROMOTION DECISION")
print("=" * 60)
print(f"Promoted           : {decision.promoted}")
print(f"Challenger version : {decision.challenger_version}")
print(f"Production AUC     : {decision.production_auc}")
print(f"Challenger AUC     : {decision.challenger_auc}")
print(f"Archived versions  : {decision.archived_versions}")
print(f"Reason             : {decision.reason}")

# COMMAND ----------

if decision.promoted:
    dbutils.notebook.exit(f"PROMOTED:v{decision.challenger_version}")  # noqa: F821
else:
    dbutils.notebook.exit("STAGING_ONLY")  # noqa: F821
