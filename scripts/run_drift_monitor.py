#!/usr/bin/env python3
"""
Weekly drift monitor — run as a Databricks Job or cron task.

Example:
    python scripts/run_drift_monitor.py \\
        --reference data/reference/gold_sample.parquet \\
        --current data/incoming/gold_latest.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import pandas as pd

from mlops.alerts import send_webhook_alert
from mlops.drift_monitor import run_drift_check


def _load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported format: {p.suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Evidently drift check")
    parser.add_argument("--reference", required=True, help="Reference/training snapshot path")
    parser.add_argument("--current", required=True, help="Current/incoming snapshot path")
    parser.add_argument("--threshold", type=float, default=None, help="Drift share alert threshold")
    parser.add_argument("--output-dir", default=None, help="HTML report output directory")
    parser.add_argument("--alert", action="store_true", help="Send webhook when drift exceeds threshold")
    args = parser.parse_args()

    reference = _load_table(args.reference)
    current = _load_table(args.current)

    report = run_drift_check(
        reference,
        current,
        threshold=args.threshold,
        output_dir=args.output_dir,
    )

    print(report.summary())
    print(json.dumps(report.to_dict(), indent=2))

    if report.alert_triggered and args.alert:
        sent = send_webhook_alert(
            f":warning: *Churn platform drift alert*\n{report.summary()}\n"
            f"Report: {report.report_path}",
        )
        print("Webhook sent." if sent else "Webhook URL not configured (ML_ALERT_WEBHOOK_URL).")

    return 1 if report.alert_triggered else 0


if __name__ == "__main__":
    raise SystemExit(main())
