"""GenAI layer: NL query engine, auto-insights, and retention copywriting."""

from genai.insight_generator import collect_statistical_checks, generate_auto_insights
from genai.query_engine import UnsafeQueryCodeError, ask, build_chart, execute_sandbox
from genai.query_logger import log_query
from genai.retention_copywriter import (
    filter_high_risk,
    generate_retention_email,
    generate_retention_emails,
)

__all__ = [
    "UnsafeQueryCodeError",
    "ask",
    "build_chart",
    "collect_statistical_checks",
    "execute_sandbox",
    "filter_high_risk",
    "generate_auto_insights",
    "generate_retention_email",
    "generate_retention_emails",
    "log_query",
]
