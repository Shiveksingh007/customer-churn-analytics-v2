"""
===========================================================
Project : Customer Churn Analytics Platform
Module  : Retention Copywriter
Author  : Shivek Singh
===========================================================

Generate personalized retention outreach email drafts for
high-risk customers using contract, tenure, charges, and
top churn-driving features as context.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

import pandas as pd

from genai.llm_client import call_llm

DEFAULT_FEATURE_COLUMNS = (
    "Contract",
    "tenure",
    "MonthlyCharges",
    "InternetService",
    "TechSupport",
    "PaymentMethod",
    "TotalServicesSubscribed",
    "IsLongTermContract",
)


def _to_pandas(df) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    if hasattr(df, "toPandas"):
        return df.toPandas()
    raise TypeError("Expected a pandas or Spark DataFrame.")


def filter_high_risk(
    df: pd.DataFrame,
    risk_column: str = "Risk_Level",
    high_value: str = "High",
) -> pd.DataFrame:
    if risk_column not in df.columns:
        raise KeyError(
            f"Column '{risk_column}' not found. "
            "Pass the prediction report from notebook 07/09."
        )
    return df[df[risk_column].astype(str).str.strip().str.lower() == high_value.lower()].copy()


def _customer_context(
    row: pd.Series,
    feature_columns: Iterable[str],
) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for col in feature_columns:
        if col in row.index and pd.notna(row[col]):
            context[col] = row[col]

    for col in (
        "customerID",
        "Churn_Percentage",
        "Churn_Probability",
        "Risk_Level",
        "Recommended_Action",
    ):
        if col in row.index and pd.notna(row[col]):
            context[col] = row[col]

    return context


def _fallback_email(context: dict[str, Any]) -> dict[str, str]:
    customer_id = context.get("customerID", "valued customer")
    contract = context.get("Contract", "your current plan")
    tenure = context.get("tenure", "several")
    charges = context.get("MonthlyCharges", "your")
    action = context.get("Recommended_Action", "a retention offer")

    subject = f"A better plan for you — keeping {customer_id} happy"
    body = (
        f"Hi,\n\n"
        f"We've noticed you've been with us for {tenure} months on a {contract} plan "
        f"at about ${charges}/month. We'd love to keep you on board.\n\n"
        f"Based on your usage profile, our retention team recommends: {action}. "
        f"Reply to this email or call us and we'll set it up within one business day.\n\n"
        f"Thank you for being a customer,\n"
        f"Customer Success Team"
    )
    return {"subject": subject, "body": body}


def generate_retention_email(
    customer: dict[str, Any] | pd.Series,
    *,
    feature_columns: Iterable[str] = DEFAULT_FEATURE_COLUMNS,
    use_llm: bool = True,
) -> dict[str, str]:
    """
    Draft one personalized retention email for a high-risk customer.

    Returns ``{"subject": ..., "body": ..., "customerID": ...}``.
    """
    if isinstance(customer, pd.Series):
        context = _customer_context(customer, feature_columns)
    else:
        context = dict(customer)

    customer_id = str(context.get("customerID", "unknown"))

    if not use_llm:
        draft = _fallback_email(context)
        draft["customerID"] = customer_id
        return draft

    try:
        system = (
            "You write short, professional customer-retention emails for a telecom company. "
            "Use ONLY the provided customer context. Do not invent discounts that are not "
            "implied by Recommended_Action. Output ONLY JSON: "
            '{"subject": "...", "body": "..."}'
        )
        user = (
            "Write a personalized retention outreach email.\n"
            f"Customer context:\n{json.dumps(context, default=str)}"
        )
        text = call_llm(system, user, max_output_tokens=700)
        start, end = text.find("{"), text.rfind("}")
        payload = json.loads(text[start : end + 1])
        return {
            "customerID": customer_id,
            "subject": str(payload.get("subject", "")).strip(),
            "body": str(payload.get("body", "")).strip(),
        }
    except Exception:
        draft = _fallback_email(context)
        draft["customerID"] = customer_id
        return draft


def generate_retention_emails(
    prediction_df,
    *,
    limit: int = 10,
    risk_column: str = "Risk_Level",
    feature_columns: Iterable[str] = DEFAULT_FEATURE_COLUMNS,
    use_llm: bool = True,
) -> list[dict[str, str]]:
    """
    Generate retention email drafts for the top high-risk customers.

    Expects the prediction report shape from notebook 07
    (``customerID``, ``Contract``, ``tenure``, ``MonthlyCharges``,
    ``Churn_Percentage``, ``Risk_Level``, ``Recommended_Action``, ...).
    """
    df = _to_pandas(prediction_df)
    high_risk = filter_high_risk(df, risk_column=risk_column)

    sort_col = "Churn_Percentage" if "Churn_Percentage" in high_risk.columns else None
    if sort_col:
        high_risk = high_risk.sort_values(sort_col, ascending=False)

    emails: list[dict[str, str]] = []
    for _, row in high_risk.head(limit).iterrows():
        emails.append(
            generate_retention_email(
                row,
                feature_columns=feature_columns,
                use_llm=use_llm,
            )
        )
    return emails
