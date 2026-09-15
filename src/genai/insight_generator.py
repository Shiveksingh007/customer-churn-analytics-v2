"""
===========================================================
Project : Customer Churn Analytics Platform
Module  : GenAI Insight Generator
Author  : Shivek Singh
===========================================================

Runs a fixed set of statistical checks, then asks Gemini to
narrate them as an automatic "state of the business" executive summary.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from genai.llm_client import call_llm

TARGET_CANDIDATES = (
    "Churn",
    "churn",
    "target",
    "label",
    "y",
    "Risk_Level",
)


def _resolve_target(df: pd.DataFrame, target_column: str | None) -> str | None:
    if target_column and target_column in df.columns:
        return target_column
    for name in TARGET_CANDIDATES:
        if name in df.columns:
            return name
    return None


def _is_churn_yes(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"yes", "1", "true", "churned", "high"}
    )


def _overall_churn_stats(df: pd.DataFrame, target: str) -> dict[str, Any]:
    rate = float(_is_churn_yes(df[target]).mean() * 100)
    return {
        "metric": "overall_churn_or_risk_rate",
        "target": target,
        "rate_pct": round(rate, 2),
        "n_customers": int(len(df)),
    }


def _segment_rates(
    df: pd.DataFrame,
    target: str,
    segment_col: str,
    top_n: int = 8,
) -> dict[str, Any] | None:
    if segment_col not in df.columns:
        return None

    work = df[[segment_col, target]].copy()
    work["_flag"] = _is_churn_yes(work[target]).astype(int)
    grouped = (
        work.groupby(segment_col, dropna=False)["_flag"]
        .agg(rate="mean", count="size")
        .reset_index()
    )
    grouped["rate_pct"] = (grouped["rate"] * 100).round(2)
    grouped = grouped.sort_values("rate_pct", ascending=False).head(top_n)
    return {
        "metric": "churn_rate_by_segment",
        "segment": segment_col,
        "rows": grouped[[segment_col, "rate_pct", "count"]].to_dict(orient="records"),
    }


def _top_correlations(
    df: pd.DataFrame,
    target: str,
    top_n: int = 5,
) -> dict[str, Any] | None:
    numeric = df.select_dtypes(include="number").copy()
    if numeric.empty:
        return None

    flag = _is_churn_yes(df[target]).astype(float)
    corrs = []
    for col in numeric.columns:
        series = numeric[col]
        if series.nunique(dropna=True) < 2:
            continue
        value = float(series.corr(flag))
        if pd.isna(value):
            continue
        corrs.append({"feature": col, "correlation_with_target": round(value, 4)})

    if not corrs:
        return None

    corrs.sort(key=lambda row: abs(row["correlation_with_target"]), reverse=True)
    return {
        "metric": "top_feature_correlations",
        "rows": corrs[:top_n],
    }


def _date_trend(df: pd.DataFrame, target: str) -> dict[str, Any] | None:
    date_cols = [
        c
        for c in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[c])
        or any(token in c.lower() for token in ("date", "month", "period", "timestamp"))
    ]
    if not date_cols:
        return None

    col = date_cols[0]
    work = df[[col, target]].copy()
    work[col] = pd.to_datetime(work[col], errors="coerce")
    work = work.dropna(subset=[col])
    if work.empty:
        return None

    work["_month"] = work[col].dt.to_period("M").astype(str)
    work["_flag"] = _is_churn_yes(work[target]).astype(int)
    trend = (
        work.groupby("_month")["_flag"]
        .mean()
        .mul(100)
        .round(2)
        .reset_index()
        .rename(columns={"_month": "month", "_flag": "rate_pct"})
        .tail(12)
    )
    return {
        "metric": "month_over_month_trend",
        "date_column": col,
        "rows": trend.to_dict(orient="records"),
    }


def collect_statistical_checks(
    df: pd.DataFrame,
    target_column: str | None = None,
) -> list[dict[str, Any]]:
    """Run the fixed statistical battery used for auto-insights."""
    target = _resolve_target(df, target_column)
    if target is None:
        return [
            {
                "metric": "no_target",
                "message": "No churn/target column found; pass target_column explicitly.",
            }
        ]

    findings: list[dict[str, Any]] = [_overall_churn_stats(df, target)]

    for segment in ("Contract", "InternetService", "PaymentMethod", "TenureGroup", "gender"):
        result = _segment_rates(df, target, segment)
        if result:
            findings.append(result)

    corr = _top_correlations(df, target)
    if corr:
        findings.append(corr)

    trend = _date_trend(df, target)
    if trend:
        findings.append(trend)

    if "Risk_Level" in df.columns and target != "Risk_Level":
        risk = (
            df["Risk_Level"]
            .astype(str)
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
            .to_dict()
        )
        findings.append({"metric": "risk_level_mix", "rows": risk})

    return findings


def generate_auto_insights(
    df: pd.DataFrame,
    target_column: str | None = None,
    *,
    narrate: bool = True,
) -> list[str]:
    """
    Produce a short list of executive-ready insight strings.

    Statistical checks always run locally. When ``narrate=True`` and an API key
    is available, Gemini turns the findings into a "state of the business"
    narrative; otherwise a deterministic fallback summary is returned.
    """
    findings = collect_statistical_checks(df, target_column=target_column)

    if not narrate:
        return [_format_finding(item) for item in findings]

    try:
        return _narrate_findings(findings)
    except Exception:
        return [_format_finding(item) for item in findings]


def _format_finding(item: dict[str, Any]) -> str:
    metric = item.get("metric", "insight")
    if metric == "overall_churn_or_risk_rate":
        return (
            f"Overall {item['target']} rate is {item['rate_pct']}% "
            f"across {item['n_customers']:,} customers."
        )
    if metric == "churn_rate_by_segment":
        top = (item.get("rows") or [{}])[0]
        seg = item.get("segment")
        key = top.get(seg, "n/a")
        return f"Highest {seg} risk: {key} at {top.get('rate_pct', 'n/a')}%."
    if metric == "top_feature_correlations":
        top = (item.get("rows") or [{}])[0]
        return (
            f"Strongest numeric signal vs target: {top.get('feature')} "
            f"(corr={top.get('correlation_with_target')})."
        )
    if metric == "month_over_month_trend":
        rows = item.get("rows") or []
        if len(rows) >= 2:
            return (
                f"Latest MoM trend ({item['date_column']}): "
                f"{rows[-2]['month']} → {rows[-1]['month']} "
                f"({rows[-2]['rate_pct']}% → {rows[-1]['rate_pct']}%)."
            )
    if metric == "risk_level_mix":
        return f"Risk mix: {item.get('rows')}"
    return json.dumps(item, default=str)


def _narrate_findings(findings: list[dict[str, Any]]) -> list[str]:
    system = (
        "You are a senior analytics advisor. Given statistical findings as JSON, "
        "write a crisp executive 'state of the business' summary as a JSON array "
        "of 4 to 7 short strings. No markdown, no preamble — ONLY the JSON array."
    )
    user = f"Findings:\n{json.dumps(findings, default=str)}"
    text = call_llm(system, user, max_output_tokens=1200)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return [_format_finding(item) for item in findings]
    payload = json.loads(text[start : end + 1])
    return [str(item) for item in payload]
