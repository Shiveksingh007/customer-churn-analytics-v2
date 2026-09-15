"""
===========================================================
Project : Customer Churn Analytics Platform
Module  : Configuration
Author  : Shivek Singh
===========================================================
"""

# ==========================================================
# Project Information
# ==========================================================

PROJECT_NAME = "Customer Churn Analytics Platform"
PROJECT_VERSION = "1.0.0"

# ==========================================================
# Raw Data
# ==========================================================

RAW_DATA_PATH = (
    "/Volumes/workspace/default/customer_churn_data/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# ==========================================================
# Bronze Layer
# ==========================================================

BRONZE_TABLE = "workspace.default.customer_churn_bronze"

# ==========================================================
# Silver Layer
# ==========================================================

SILVER_TABLE = "workspace.default.customer_churn_silver"

# ==========================================================
# Gold Layer
# ==========================================================

GOLD_TABLE = "workspace.default.customer_churn_gold"

# ==========================================================
# Predictions
# ==========================================================

PREDICTIONS_TABLE = "workspace.default.customer_churn_predictions"

# ==========================================================
# Ingestion / Bring Your Own Dataset
# ==========================================================

SUPPORTED_UPLOAD_FORMATS = (".csv", ".parquet", ".json", ".jsonl")

# Local fallback when Unity Catalog volumes are unavailable.
LOCAL_RAW_DATA_DIR = "data/raw"

# Common target-column aliases used by auto_profiler target detection.
TARGET_COLUMN_ALIASES = (
    "churn",
    "target",
    "label",
    "y",
    "default",
    "outcome",
)

# ==========================================================
# GenAI
# ==========================================================

GENAI_MODEL = "gemini-2.5-flash"
GENAI_QUERY_LOG_TABLE = "workspace.default.genai_query_log"

# ==========================================================
# Document RAG (Phase 7 — separate from churn analytics)
# ==========================================================

RAG_STORE_DIR = "data/rag/chroma"
RAG_CHUNK_SIZE = 500  # approximate tokens (word count)
RAG_CHUNK_OVERLAP = 50

# ==========================================================
# MLOps
# ==========================================================

MLFLOW_MODEL_NAME = "customer_churn_rf"
DRIFT_ALERT_THRESHOLD = 0.25  # alert when >25% of monitored columns drift
DRIFT_REPORT_DIR = "reports/drift"
ML_ALERT_WEBHOOK_URL = ""  # Slack/Teams webhook; override via env var
