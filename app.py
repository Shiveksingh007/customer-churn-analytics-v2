"""Deployment entry point for the Customer Churn Analytics Platform V2."""
from pathlib import Path
import subprocess
import sys

APP = Path(__file__).parent / "app" / "streamlit_app.py"
if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(APP)]))
