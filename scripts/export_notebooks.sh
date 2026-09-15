#!/usr/bin/env bash
# Regenerate .ipynb notebooks from Databricks-style .py sources.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m pip install -q jupytext
python scripts/export_notebooks.py
