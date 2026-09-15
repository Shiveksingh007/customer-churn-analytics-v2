"""
Pandas analytics mirroring notebooks 01–09 for the Streamlit dashboard.

Works fully offline without Spark/Databricks — suitable for demos and
employee walkthroughs of the Medallion + ML + BI pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SERVICE_COLS = (
    "PhoneService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)

INSIGHT_SEGMENT_COLS = (
    "gender",
    "SeniorCitizen",
    "Contract",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "DeviceProtection",
    "PaymentMethod",
    "Partner",
    "Dependents",
)

MODEL_METRICS = {
    "accuracy": 80.30,
    "f1": 79.20,
    "roc_auc": 85.85,
}

FEATURE_IMPORTANCE = [
    ("Contract", 0.22),
    ("MonthlyCharges", 0.14),
    ("TechSupport", 0.11),
    ("IsLongTermContract", 0.10),
    ("InternetService", 0.09),
    ("tenure", 0.08),
    ("OnlineSecurity", 0.07),
    ("PaymentMethod", 0.06),
    ("TotalCharges", 0.05),
    ("HasFamily", 0.04),
]


def is_churned(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"yes", "1", "true", "churned", "high"}
    )


def build_demo_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Larger demo dataset resembling Telco churn for end-to-end UI demos."""
    rng = np.random.default_rng(seed)
    contracts = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n,
        p=[0.55, 0.25, 0.20],
    )
    internet = rng.choice(
        ["Fiber optic", "DSL", "No"],
        size=n,
        p=[0.44, 0.34, 0.22],
    )
    tenure = rng.integers(1, 73, size=n)
    monthly = np.round(rng.uniform(20, 120, size=n), 2)
    partner = rng.choice(["Yes", "No"], size=n)
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.3, 0.7])
    senior = rng.choice([0, 1], size=n, p=[0.84, 0.16])
    gender = rng.choice(["Male", "Female"], size=n)
    tech = rng.choice(["Yes", "No", "No internet service"], size=n, p=[0.3, 0.5, 0.2])
    security = rng.choice(["Yes", "No", "No internet service"], size=n, p=[0.28, 0.52, 0.2])
    device = rng.choice(["Yes", "No", "No internet service"], size=n, p=[0.35, 0.45, 0.2])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n,
    )
    phone = rng.choice(["Yes", "No"], size=n, p=[0.9, 0.1])

    # Churn propensity aligned with notebook business findings.
    p_churn = np.full(n, 0.18)
    p_churn = np.where(contracts == "Month-to-month", p_churn + 0.22, p_churn)
    p_churn = np.where(contracts == "Two year", p_churn - 0.12, p_churn)
    p_churn = np.where(internet == "Fiber optic", p_churn + 0.12, p_churn)
    p_churn = np.where(senior == 1, p_churn + 0.10, p_churn)
    p_churn = np.where(tech == "No", p_churn + 0.10, p_churn)
    p_churn = np.where(security == "No", p_churn + 0.08, p_churn)
    p_churn = np.where(tenure < 12, p_churn + 0.08, p_churn)
    p_churn = np.clip(p_churn, 0.02, 0.85)
    churn = np.where(rng.random(n) < p_churn, "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": [f"DEMO-{i:04d}" for i in range(1, n + 1)],
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], size=n),
            "DeviceProtection": device,
            "TechSupport": tech,
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], size=n),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], size=n),
            "Contract": contracts,
            "PaperlessBilling": rng.choice(["Yes", "No"], size=n),
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": np.round(monthly * tenure * rng.uniform(0.85, 1.05, size=n), 2),
            "Churn": churn,
        }
    )
    return enrich_pipeline(df)


def enrich_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Gold-style features + risk scoring when prediction columns are missing."""
    out = df.copy()
    out = _coerce_total_charges(out)
    out = engineer_gold_features(out)
    if "Risk_Level" not in out.columns:
        out = assign_risk_scores(out)
    return out


def _coerce_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    if "TotalCharges" not in df.columns:
        return df
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    return out


def engineer_gold_features(df: pd.DataFrame) -> pd.DataFrame:
    """Notebook 05 Gold feature engineering (pandas)."""
    out = df.copy()

    if {"Partner", "Dependents"}.issubset(out.columns):
        out["HasFamily"] = np.where(
            (out["Partner"].astype(str) == "Yes") | (out["Dependents"].astype(str) == "Yes"),
            "Yes",
            "No",
        )

    if "tenure" in out.columns:
        out["TenureGroup"] = pd.cut(
            pd.to_numeric(out["tenure"], errors="coerce"),
            bins=[-np.inf, 12, 36, 60, np.inf],
            labels=["New", "Regular", "Loyal", "Very Loyal"],
        ).astype(str)

    if "MonthlyCharges" in out.columns:
        out["MonthlyChargeLevel"] = pd.cut(
            pd.to_numeric(out["MonthlyCharges"], errors="coerce"),
            bins=[-np.inf, 35, 70, np.inf],
            labels=["Low", "Medium", "High"],
        ).astype(str)

    if "Contract" in out.columns:
        out["IsLongTermContract"] = np.where(
            out["Contract"].astype(str).isin(["One year", "Two year"]),
            "Yes",
            "No",
        )

    if "InternetService" in out.columns:
        out["HasInternet"] = np.where(
            out["InternetService"].astype(str).isin(["No", "No internet service"]),
            "No",
            "Yes",
        )

    present_services = [c for c in SERVICE_COLS if c in out.columns]
    if present_services:
        yes_flags = out[present_services].apply(
            lambda col: col.astype(str).str.strip().eq("Yes").astype(int)
        )
        out["TotalServicesSubscribed"] = yes_flags.sum(axis=1)

    return out


def assign_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heuristic risk proxy when Databricks ML predictions are unavailable.
    Not a replacement for the RF model — useful for local demos.
    """
    out = df.copy()
    score = np.zeros(len(out))

    if "Contract" in out.columns:
        score += np.where(out["Contract"].astype(str) == "Month-to-month", 0.35, 0.0)
        score += np.where(out["Contract"].astype(str) == "One year", 0.10, 0.0)

    if "InternetService" in out.columns:
        score += np.where(out["InternetService"].astype(str) == "Fiber optic", 0.15, 0.0)

    if "TechSupport" in out.columns:
        score += np.where(out["TechSupport"].astype(str) == "No", 0.12, 0.0)

    if "OnlineSecurity" in out.columns:
        score += np.where(out["OnlineSecurity"].astype(str) == "No", 0.10, 0.0)

    if "tenure" in out.columns:
        tenure = pd.to_numeric(out["tenure"], errors="coerce").fillna(0)
        score += np.where(tenure < 12, 0.12, 0.0)
        score += np.where(tenure < 6, 0.05, 0.0)

    if "MonthlyCharges" in out.columns:
        charges = pd.to_numeric(out["MonthlyCharges"], errors="coerce").fillna(0)
        score += np.where(charges >= 70, 0.08, 0.0)

    if "Churn" in out.columns:
        # Align slightly with known labels for demo coherence.
        score += np.where(is_churned(out["Churn"]), 0.15, 0.0)

    score = np.clip(score, 0.05, 0.97)
    out["Churn_Probability"] = np.round(score, 4)
    out["Churn_Percentage"] = np.round(score * 100, 2)
    out["Risk_Level"] = np.where(
        score >= 0.80,
        "High",
        np.where(score >= 0.60, "Medium", "Low"),
    )
    out["Recommended_Action"] = out["Risk_Level"].map(
        {
            "High": "Immediate Retention Call",
            "Medium": "Offer Discount / Bundle Plan",
            "Low": "Loyalty Program",
        }
    )
    return out


def data_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """Notebooks 01–03 style DQ summary."""
    row_count = len(df)
    nulls = df.isna().sum()
    blank_total = 0
    if "TotalCharges" in df.columns:
        blank_total = int(
            (df["TotalCharges"].astype(str).str.strip() == "").sum()
            + df["TotalCharges"].isna().sum()
        )

    pk_ok = None
    if "customerID" in df.columns:
        pk_ok = bool(df["customerID"].nunique(dropna=True) == row_count)

    return {
        "rows": row_count,
        "columns": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "null_cells": int(nulls.sum()),
        "columns_with_nulls": int((nulls > 0).sum()),
        "blank_or_null_TotalCharges": blank_total,
        "primary_key_unique": pk_ok,
        "nulls_by_column": nulls[nulls > 0].sort_values(ascending=False).to_dict(),
    }


def executive_kpis(df: pd.DataFrame) -> dict[str, Any]:
    total = len(df)
    churn_col = "Churn" if "Churn" in df.columns else None
    churned = int(is_churned(df[churn_col]).sum()) if churn_col else None
    active = (total - churned) if churned is not None else None
    churn_rate = round(churned / total * 100, 2) if churned is not None and total else None

    risk_counts = (
        df["Risk_Level"].astype(str).value_counts().to_dict()
        if "Risk_Level" in df.columns
        else {}
    )
    high = int(risk_counts.get("High", 0))
    medium = int(risk_counts.get("Medium", 0))
    low = int(risk_counts.get("Low", 0))

    avg_monthly = (
        float(pd.to_numeric(df["MonthlyCharges"], errors="coerce").mean())
        if "MonthlyCharges" in df.columns
        else None
    )
    avg_tenure = (
        float(pd.to_numeric(df["tenure"], errors="coerce").mean())
        if "tenure" in df.columns
        else None
    )
    avg_churn_pct = (
        float(pd.to_numeric(df["Churn_Percentage"], errors="coerce").mean())
        if "Churn_Percentage" in df.columns
        else None
    )

    monthly_at_risk = 0.0
    if "Risk_Level" in df.columns and "MonthlyCharges" in df.columns:
        high_mask = df["Risk_Level"].astype(str) == "High"
        monthly_at_risk = float(
            pd.to_numeric(df.loc[high_mask, "MonthlyCharges"], errors="coerce").sum()
        )

    return {
        "total_customers": total,
        "active_customers": active,
        "churned_customers": churned,
        "churn_rate_pct": churn_rate,
        "high_risk": high,
        "medium_risk": medium,
        "low_risk": low,
        "avg_monthly_charges": round(avg_monthly, 2) if avg_monthly is not None else None,
        "avg_tenure": round(avg_tenure, 2) if avg_tenure is not None else None,
        "avg_churn_percentage": round(avg_churn_pct, 2) if avg_churn_pct is not None else None,
        "monthly_revenue_at_risk": round(monthly_at_risk, 2),
        "annual_revenue_at_risk": round(monthly_at_risk * 12, 2),
    }


def churn_by_segment(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Notebook 04 analyze_churn equivalent."""
    if column not in df.columns or "Churn" not in df.columns:
        return pd.DataFrame()

    work = df[[column, "Churn"]].copy()
    work["_churned"] = is_churned(work["Churn"]).astype(int)
    grouped = (
        work.groupby(column, dropna=False)
        .agg(Total_Customers=("_churned", "size"), Churned_Customers=("_churned", "sum"))
        .reset_index()
    )
    grouped["Churn_Rate"] = (
        grouped["Churned_Customers"] / grouped["Total_Customers"] * 100
    ).round(2)
    return grouped.sort_values("Churn_Rate", ascending=False)


def available_insight_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in INSIGHT_SEGMENT_COLS if c in df.columns]


def risk_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if "Risk_Level" not in df.columns:
        return pd.DataFrame()
    return (
        df["Risk_Level"]
        .astype(str)
        .value_counts()
        .rename_axis("Risk_Level")
        .reset_index(name="count")
    )


def contract_risk_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if not {"Risk_Level", "Contract"}.issubset(df.columns):
        return pd.DataFrame()
    return (
        df.groupby(["Risk_Level", "Contract"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["Risk_Level", "count"], ascending=[True, False])
    )


def risk_profile_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "Risk_Level" not in df.columns:
        return pd.DataFrame()
    aggs: dict[str, Any] = {"Customers": ("Risk_Level", "size")}
    if "MonthlyCharges" in df.columns:
        aggs["Avg_MonthlyCharges"] = (
            "MonthlyCharges",
            lambda s: round(pd.to_numeric(s, errors="coerce").mean(), 2),
        )
    if "tenure" in df.columns:
        aggs["Avg_Tenure"] = (
            "tenure",
            lambda s: round(pd.to_numeric(s, errors="coerce").mean(), 2),
        )
    if "Churn_Percentage" in df.columns:
        aggs["Avg_Churn_Percentage"] = (
            "Churn_Percentage",
            lambda s: round(pd.to_numeric(s, errors="coerce").mean(), 2),
        )
    return df.groupby("Risk_Level", dropna=False).agg(**aggs).reset_index()


def top_high_risk(df: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    if "Risk_Level" not in df.columns:
        return pd.DataFrame()
    high = df[df["Risk_Level"].astype(str) == "High"].copy()
    if "Churn_Percentage" in high.columns:
        high = high.sort_values("Churn_Percentage", ascending=False)
    cols = [
        c
        for c in (
            "customerID",
            "Contract",
            "tenure",
            "MonthlyCharges",
            "InternetService",
            "Churn_Percentage",
            "Risk_Level",
            "Recommended_Action",
        )
        if c in high.columns
    ]
    return high[cols].head(limit)


def feature_distribution(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame()
    return (
        df[column]
        .astype(str)
        .value_counts()
        .rename_axis(column)
        .reset_index(name="count")
    )


def pipeline_steps() -> list[dict[str, str]]:
    return [
        {"step": "01", "name": "Data Ingestion", "layer": "Bronze", "desc": "Load raw CSV and create bronze table"},
        {"step": "02", "name": "Data Validation", "layer": "Quality", "desc": "Row counts, PK, nulls, duplicates"},
        {"step": "03", "name": "Data Cleaning", "layer": "Silver", "desc": "Fix TotalCharges blanks, standardize types"},
        {"step": "04", "name": "Business Insights", "layer": "Analytics", "desc": "Churn KPIs by segment"},
        {"step": "05", "name": "Feature Engineering", "layer": "Gold", "desc": "HasFamily, TenureGroup, services, etc."},
        {"step": "06", "name": "Feature Validation", "layer": "Gold QA", "desc": "Validate engineered features vs churn"},
        {"step": "07", "name": "ML Pipeline", "layer": "Model", "desc": "Random Forest, importance, risk scores"},
        {"step": "08", "name": "Executive Dashboard", "layer": "BI", "desc": "Risk mix, revenue at risk, personas"},
        {"step": "09", "name": "Model Validation", "layer": "QA", "desc": "Business checks on predictions"},
    ]
