"""
===========================================================
Project : Customer Churn Analytics Platform
Module  : GenAI Query Logger
Author  : Shivek Singh
===========================================================

Audit trail for NL → code → result interactions.
Persists to a Delta table when Spark is available, otherwise
appends JSON lines under data/genai_query_log/.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LOG_TABLE = "workspace.default.genai_query_log"
LOCAL_LOG_DIR = Path("data") / "genai_query_log"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_preview(result_preview: Any) -> str:
    try:
        return json.dumps(result_preview, default=str)[:8000]
    except Exception:
        return str(result_preview)[:8000]


def log_query(
    question: str,
    code: str,
    result_preview: Any,
    answer: str = "",
    chart_type: str = "none",
    error: str | None = None,
    spark=None,
    table_name: str = DEFAULT_LOG_TABLE,
) -> dict[str, Any]:
    """
    Write one audit row for a GenAI query.

    Prefer Delta via Spark; fall back to a local JSONL file so local Jupyter
    / Streamlit sessions still keep an audit trail.
    """
    record = {
        "query_id": str(uuid.uuid4()),
        "timestamp_utc": _utc_now_iso(),
        "question": question,
        "generated_code": code,
        "result_preview": _serialize_preview(result_preview),
        "answer": answer,
        "chart_type": chart_type,
        "error": error,
        "status": "error" if error else "ok",
    }

    if spark is not None:
        _log_to_delta(spark, record, table_name)
    else:
        _log_to_jsonl(record)

    return record


def _log_to_delta(spark, record: dict[str, Any], table_name: str) -> None:
    from pyspark.sql import Row

    row = Row(**record)
    (
        spark.createDataFrame([row])
        .write.format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(table_name)
    )


def _log_to_jsonl(record: dict[str, Any]) -> None:
    LOCAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOCAL_LOG_DIR / "queries.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
