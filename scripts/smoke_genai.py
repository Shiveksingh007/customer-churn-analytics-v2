"""Offline Phase 3 safety checks (no Anthropic API required)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from genai.insight_generator import collect_statistical_checks, generate_auto_insights
from genai.query_engine import UnsafeQueryCodeError, execute_sandbox, validate_query_code
from genai.query_logger import log_query
from genai.retention_copywriter import generate_retention_emails


def main() -> None:
    df = pd.DataFrame(
        {
            "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"],
            "tenure": [2, 24, 5, 48],
            "MonthlyCharges": [70.0, 55.0, 85.0, 40.0],
            "Churn": ["Yes", "No", "Yes", "No"],
            "customerID": ["A", "B", "C", "D"],
            "Risk_Level": ["High", "Low", "High", "Low"],
            "Churn_Percentage": [90, 10, 85, 5],
            "Recommended_Action": [
                "Immediate Retention Call",
                "Loyalty Program",
                "Immediate Retention Call",
                "Loyalty Program",
            ],
        }
    )

    code = "result = (df['Churn'] == 'Yes').mean()"
    assert execute_sandbox(code, df) == 0.5

    for bad in ("import os", 'open("x")', "df.__class__", "eval('1')"):
        try:
            validate_query_code(bad)
            raise AssertionError(f"should reject: {bad}")
        except UnsafeQueryCodeError:
            pass

    findings = collect_statistical_checks(df)
    assert findings[0]["metric"] == "overall_churn_or_risk_rate"
    assert len(generate_auto_insights(df, narrate=False)) >= 1

    emails = generate_retention_emails(df, limit=2, use_llm=False)
    assert len(emails) == 2 and "subject" in emails[0]

    record = log_query("test?", "result = 1", {"ok": True}, answer="hi")
    assert "query_id" in record
    print("PHASE3 UNIT CHECKS PASSED")


if __name__ == "__main__":
    main()
