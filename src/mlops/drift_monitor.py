"""
Data drift monitoring with Evidently AI.

Compares reference (training) vs current (production/incoming) feature distributions
and raises alerts when drift exceeds a configurable threshold.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import DRIFT_ALERT_THRESHOLD, DRIFT_REPORT_DIR


@dataclass
class DriftReport:
    timestamp_utc: str
    drift_share: float
    drifted_columns: list[str]
    column_count: int
    alert_triggered: bool
    threshold: float
    report_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        status = "ALERT" if self.alert_triggered else "OK"
        cols = ", ".join(self.drifted_columns[:8]) or "none"
        suffix = "..." if len(self.drifted_columns) > 8 else ""
        return (
            f"[{status}] Drift share {self.drift_share:.1%} "
            f"(threshold {self.threshold:.0%}) · drifted: {cols}{suffix}"
        )


def _to_pandas(df) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df.copy()
    if hasattr(df, "toPandas"):
        return df.toPandas()
    raise TypeError("Expected pandas or Spark DataFrame.")


def _select_numeric(df: pd.DataFrame, columns: list[str] | None) -> pd.DataFrame:
    if columns:
        present = [c for c in columns if c in df.columns]
        return df[present].copy()
    numeric = df.select_dtypes(include="number")
    return numeric.copy()


def _run_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_html: Path | None,
) -> tuple[float, list[str]]:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)

    if output_html is not None:
        output_html.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(output_html))

    payload = json.loads(snapshot.json())
    drifted: list[str] = []
    drift_share = 0.0

    for item in payload.get("metrics", []):
        metric_result = item.get("result") or {}
        if item.get("metric_id") == "DriftedColumnsCount":
            drift_share = float(metric_result.get("share", 0.0) or 0.0)
        if item.get("metric_id") == "DatasetDriftMetric":
            drifted = list(metric_result.get("drifted_columns", []) or [])

    if not drifted and drift_share > 0:
        # Fallback: infer from per-column drift metrics when preset shape differs.
        for item in payload.get("metrics", []):
            result = item.get("result") or {}
            if result.get("drift_detected") is True:
                col = result.get("column_name")
                if col:
                    drifted.append(col)

    return drift_share, sorted(set(drifted))


def run_drift_check(
    reference_df,
    current_df,
    *,
    feature_columns: list[str] | None = None,
    threshold: float | None = None,
    output_dir: str | Path | None = None,
) -> DriftReport:
    """
    Compare reference vs current data and return a drift report.

    Parameters
    ----------
    reference_df : pandas or Spark DataFrame
        Training / baseline snapshot.
    current_df : pandas or Spark DataFrame
        New incoming or production snapshot.
    feature_columns : list, optional
        Numeric columns to monitor. Defaults to all numeric columns in reference.
    threshold : float, optional
        Alert when drift_share exceeds this value (0–1).
    output_dir : path, optional
        Directory for HTML report artifacts.
    """
    threshold = DRIFT_ALERT_THRESHOLD if threshold is None else threshold
    ref = _select_numeric(_to_pandas(reference_df), feature_columns)
    cur = _select_numeric(_to_pandas(current_df), feature_columns)

    common = [c for c in ref.columns if c in cur.columns]
    if not common:
        raise ValueError("No overlapping numeric columns between reference and current data.")

    ref = ref[common]
    cur = cur[common]

    out_dir = Path(output_dir or DRIFT_REPORT_DIR)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    html_path = out_dir / f"drift_report_{stamp}.html"

    drift_share, drifted = _run_evidently_report(ref, cur, html_path)
    alert = drift_share >= threshold

    return DriftReport(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        drift_share=drift_share,
        drifted_columns=drifted,
        column_count=len(common),
        alert_triggered=alert,
        threshold=threshold,
        report_path=str(html_path),
    )
