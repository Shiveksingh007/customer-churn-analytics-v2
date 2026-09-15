#!/usr/bin/env python3
"""
Regenerate .ipynb notebooks from Databricks-style .py sources.

Usage:
    python scripts/export_notebooks.py

Requires: jupytext (pip install jupytext)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

SHIM_CELL_SOURCE = """\
# Local Jupyter / GitHub compatibility (no-op on Databricks)
import os
import sys
from pathlib import Path

if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    _root = next(
        (
            p
            for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "src" / "utils" / "databricks_compat.py").exists()
        ),
        None,
    )
    if _root and str(_root / "src") not in sys.path:
        sys.path.insert(0, str(_root / "src"))
    from utils.databricks_compat import inject_globals

    inject_globals(globals())
"""


def find_py_notebooks() -> list[Path]:
    return sorted(NOTEBOOKS_DIR.glob("*.py"))


def _is_markdown_cell(cell_lines: list[str]) -> bool:
    return any(line.strip().startswith("# MAGIC %md") for line in cell_lines)


def _to_percent_markdown(cell_lines: list[str]) -> list[str]:
    """Convert Databricks MAGIC markdown lines to jupytext percent markdown."""
    out = ["# %% [markdown]"]
    for line in cell_lines:
        stripped = line.strip()
        if stripped.startswith("# MAGIC %md"):
            continue
        if line.startswith("# MAGIC "):
            out.append("# " + line[8:])
        elif line.startswith("# MAGIC"):
            out.append("# " + line[7:].lstrip())
        else:
            out.append(line)
    return out


def databricks_py_to_percent(py_content: str) -> str:
    """
    Convert Databricks .py export (# COMMAND / # MAGIC) to jupytext percent format.

    Jupytext does not natively parse Databricks cell markers; we translate them
    to `# %%` / `# %% [markdown]` before conversion.
    """
    lines = py_content.splitlines()
    if lines and "Databricks notebook source" in lines[0]:
        lines = lines[1:]

    raw_cells: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        if line.strip() == "# COMMAND ----------":
            if current:
                raw_cells.append(current)
                current = []
        else:
            current.append(line)

    if current:
        raw_cells.append(current)

    percent_lines: list[str] = []
    for cell_lines in raw_cells:
        if not any(line.strip() for line in cell_lines):
            continue
        if _is_markdown_cell(cell_lines):
            percent_lines.extend(_to_percent_markdown(cell_lines))
        else:
            percent_lines.append("# %%")
            percent_lines.extend(cell_lines)
        percent_lines.append("")

    return "\n".join(percent_lines).rstrip() + "\n"


def convert_with_jupytext(py_path: Path) -> Path:
    ipynb_path = py_path.with_suffix(".ipynb")
    percent_source = databricks_py_to_percent(py_path.read_text(encoding="utf-8"))

    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupytext",
            "--from",
            "py:percent",
            "--to",
            "notebook",
            "-o",
            str(ipynb_path),
        ],
        input=percent_source.encode("utf-8"),
        check=True,
        cwd=REPO_ROOT,
    )
    if not ipynb_path.exists():
        raise FileNotFoundError(f"Expected output not found: {ipynb_path}")
    return ipynb_path


def insert_shim_cell(ipynb_path: Path) -> None:
    """Insert compatibility shim as the first code cell (after any leading markdown)."""
    notebook = json.loads(ipynb_path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])

    shim_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["databricks-compat-shim"]},
        "outputs": [],
        "source": [line + "\n" for line in SHIM_CELL_SOURCE.splitlines()],
    }

    insert_at = 0
    for cell in cells:
        if cell.get("cell_type") == "markdown":
            insert_at += 1
            continue
        break

    # Avoid duplicating the shim on re-export.
    for cell in cells:
        if cell.get("metadata", {}).get("tags") == ["databricks-compat-shim"]:
            cell["source"] = shim_cell["source"]
            notebook["cells"] = cells
            ipynb_path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
            return

    cells.insert(insert_at, shim_cell)
    notebook["cells"] = cells
    ipynb_path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


def export_all() -> int:
    py_notebooks = find_py_notebooks()
    if not py_notebooks:
        print("No .py notebooks found in notebooks/", file=sys.stderr)
        return 1

    print(f"Exporting {len(py_notebooks)} notebook(s) from {NOTEBOOKS_DIR}")
    for py_path in py_notebooks:
        print(f"  {py_path.name} -> {py_path.stem}.ipynb")
        ipynb_path = convert_with_jupytext(py_path)
        insert_shim_cell(ipynb_path)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(export_all())
