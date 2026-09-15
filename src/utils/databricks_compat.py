"""
Databricks → local Jupyter compatibility layer.

Provides spark, display, and dbutils when notebooks run outside Databricks.
No-op when DATABRICKS_RUNTIME_VERSION is set (native globals are used).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def is_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def find_project_root() -> Path:
    """Locate repo root (directory that contains src/config)."""
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "src" / "config").is_dir():
            return candidate
    raise RuntimeError(
        "Could not find project root. Run notebooks from the repo or notebooks/ directory."
    )


def _ensure_src_on_path(project_root: Path | None = None) -> Path:
    root = project_root or find_project_root()
    src_path = str(root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    return root


def display(obj: Any) -> None:
    """Databricks display() fallback for Jupyter and plain Python."""
    try:
        from IPython.display import display as ipy_display
    except ImportError:
        ipy_display = print

    if hasattr(obj, "toPandas"):
        ipy_display(obj.toPandas())
    elif hasattr(obj, "show"):
        obj.show(truncate=False)
    else:
        ipy_display(obj)


class _Widgets:
    """Minimal dbutils.widgets replacement for local runs."""

    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def get(self, name: str, default: str | None = None) -> str | None:
        env_key = f"WIDGET_{name.upper()}"
        return os.environ.get(env_key, self._values.get(name, default))

    def text(self, name: str, default: str, label: str = "") -> None:
        self._values[name] = default

    def dropdown(self, name: str, default: str, choices: list[str], label: str = "") -> None:
        self._values[name] = default

    def combobox(self, name: str, default: str, choices: list[str], label: str = "") -> None:
        self._values[name] = default


class _Dbutils:
    def __init__(self) -> None:
        self.widgets = _Widgets()


def create_local_spark(project_root: Path):
    """Create a local SparkSession with a warehouse under the project."""
    from pyspark.sql import SparkSession

    warehouse = project_root / "spark-warehouse"
    warehouse.mkdir(parents=True, exist_ok=True)

    builder = (
        SparkSession.builder.appName("Customer-Churn-Platform")
        .master("local[*]")
        .config("spark.sql.warehouse.dir", str(warehouse))
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
    )

    # Enable Delta Lake when the package is available (optional for local runs).
    try:
        from delta import configure_spark_with_delta_pip  # type: ignore[import-untyped]

        builder = configure_spark_with_delta_pip(builder)
    except ImportError:
        pass

    return builder.getOrCreate()


def inject_globals(namespace: dict[str, Any] | None = None) -> None:
    """
    Inject spark, display, and dbutils into the caller namespace.

    Safe to call from a notebook cell; skipped on Databricks.
    """
    if is_databricks():
        return

    target = namespace if namespace is not None else sys._getframe(1).f_globals

    root = _ensure_src_on_path()

    if "spark" not in target:
        target["spark"] = create_local_spark(root)

    target["display"] = display
    target["dbutils"] = _Dbutils()

    # Resolve Databricks-hardcoded project paths for local src imports.
    target.setdefault("PROJECT_ROOT", str(root))
    target.setdefault("SRC_PATH", str(root / "src"))
