"""Business-focused validation for customer churn uploads."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import pandas as pd

REQUIRED_COLUMNS = [
    "customerID","gender","SeniorCitizen","Partner","Dependents","tenure",
    "PhoneService","InternetService","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
    "Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges",
]
NUMERIC_COLUMNS = {"SeniorCitizen","tenure","MonthlyCharges","TotalCharges"}
EXPECTED_CATEGORIES = {
    "gender":{"Male","Female"},
    "SeniorCitizen":{"0","1"},
    "Partner":{"Yes","No"},
    "Dependents":{"Yes","No"},
    "PhoneService":{"Yes","No"},
    "InternetService":{"DSL","Fiber optic","No","No internet service"},
    "OnlineSecurity":{"Yes","No","No internet service"},
    "OnlineBackup":{"Yes","No","No internet service"},
    "DeviceProtection":{"Yes","No","No internet service"},
    "TechSupport":{"Yes","No","No internet service"},
    "StreamingTV":{"Yes","No","No internet service"},
    "StreamingMovies":{"Yes","No","No internet service"},
    "Contract":{"Month-to-month","One year","Two year"},
    "PaperlessBilling":{"Yes","No"},
    "PaymentMethod":{"Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"},
}

@dataclass
class ValidationResult:
    valid: bool
    score: int
    errors: list[str]
    warnings: list[str]
    checks: list[dict[str, Any]]
    @property
    def status(self) -> str:
        return "Invalid" if self.errors else ("Valid with warnings" if self.warnings else "Ready")

def validate_customer_data(df: pd.DataFrame) -> ValidationResult:
    errors, warnings, checks = [], [], []
    if df is None or df.empty:
        return ValidationResult(False, 0, ["The uploaded dataset is empty."], [], [
            {"check":"Dataset not empty","status":"FAIL","details":"0 rows"}
        ])

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS and c not in {
        "Churn","Risk_Level","Churn_Percentage","Recommended_Action"
    }]
    checks.append({"check":"Required columns","status":"PASS" if not missing else "FAIL",
                   "details":"All core customer fields present." if not missing else "Missing: "+", ".join(missing)})
    if missing:
        errors.append(f"Missing {len(missing)} required column(s): {', '.join(missing)}")
    checks.append({"check":"Unexpected columns","status":"PASS" if not extra else "WARN",
                   "details":"No unexpected columns." if not extra else f"{len(extra)} extra column(s)"})
    if extra:
        warnings.append(f"{len(extra)} extra column(s) detected; they will be preserved.")

    dup_rows = int(df.duplicated().sum())
    checks.append({"check":"Duplicate rows","status":"PASS" if dup_rows == 0 else "WARN",
                   "details":f"{dup_rows:,} duplicate row(s)"})
    if dup_rows:
        warnings.append(f"{dup_rows:,} duplicate row(s) detected.")

    if "customerID" in df:
        ids = df["customerID"]
        missing_ids = int(ids.isna().sum() + ids.astype(str).str.strip().eq("").sum())
        dup_ids = int(ids.dropna().astype(str).duplicated().sum())
        checks.append({"check":"Customer ID integrity","status":"PASS" if not (missing_ids or dup_ids) else "FAIL",
                       "details":"Unique, non-blank customerID." if not (missing_ids or dup_ids)
                       else f"Missing/blank: {missing_ids:,}; duplicates: {dup_ids:,}"})
        if missing_ids:
            errors.append(f"customerID has {missing_ids:,} missing/blank value(s).")
        if dup_ids:
            errors.append(f"customerID contains {dup_ids:,} duplicate ID(s).")

    null_cells = int(df.isna().sum().sum())
    null_cols = int(df.isna().any(axis=0).sum())
    checks.append({"check":"Missing values","status":"PASS" if null_cells == 0 else "WARN",
                   "details":"No null cells." if null_cells == 0 else f"{null_cells:,} missing cell(s) across {null_cols} column(s)"})
    if null_cells:
        warnings.append(f"{null_cells:,} missing cell(s) found across {null_cols} column(s).")

    for col in NUMERIC_COLUMNS.intersection(df.columns):
        converted = pd.to_numeric(df[col], errors="coerce")
        bad = int(converted.isna().sum() - df[col].isna().sum())
        checks.append({"check":f"Numeric type: {col}","status":"PASS" if bad == 0 else "FAIL",
                       "details":"Numeric values." if bad == 0 else f"{bad:,} invalid value(s)"})
        if bad:
            errors.append(f"{col} contains {bad:,} non-numeric value(s).")

    if "tenure" in df:
        tenure = pd.to_numeric(df["tenure"], errors="coerce")
        invalid = int(((tenure < 0) | (tenure > 120)).fillna(False).sum())
        checks.append({"check":"Tenure range","status":"PASS" if invalid == 0 else "FAIL",
                       "details":"0–120 months." if invalid == 0 else f"{invalid:,} out-of-range value(s)"})
        if invalid:
            errors.append(f"tenure contains {invalid:,} value(s) outside the expected 0–120 month range.")

    if "MonthlyCharges" in df:
        charges = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        invalid = int((charges < 0).fillna(False).sum())
        checks.append({"check":"Monthly charges","status":"PASS" if invalid == 0 else "FAIL",
                       "details":"No negative charges." if invalid == 0 else f"{invalid:,} negative value(s)"})
        if invalid:
            errors.append(f"MonthlyCharges contains {invalid:,} negative value(s).")

    for col, allowed in EXPECTED_CATEGORIES.items():
        if col not in df:
            continue
        observed = set(df[col].dropna().astype(str).str.strip().unique())
        unknown = sorted(observed - allowed)
        checks.append({"check":f"Category values: {col}","status":"PASS" if not unknown else "WARN",
                       "details":"Expected categories only." if not unknown else f"{len(unknown)} unexpected value(s)"})
        if unknown:
            warnings.append(f"{col} contains unexpected value(s): {', '.join(unknown[:5])}{' …' if len(unknown)>5 else ''}")

    checks.append({"check":"Historical Churn label","status":"PASS" if "Churn" in df else "WARN",
                   "details":"Churn column present." if "Churn" in df else "No Churn column; unlabeled input"})
    if "Churn" not in df:
        warnings.append("No Churn target column found; historical churn analytics will be limited.")

    failures = sum(c["status"] == "FAIL" for c in checks)
    warns = sum(c["status"] == "WARN" for c in checks)
    score = max(0, min(100, 100 - failures*18 - warns*4))
    return ValidationResult(not errors, score, errors, warnings, checks)
