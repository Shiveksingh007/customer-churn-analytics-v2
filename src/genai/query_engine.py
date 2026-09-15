"""
===========================================================
Project : Customer Churn Analytics Platform
Module  : GenAI Query Engine
Author  : Shivek Singh
===========================================================

Natural-language → Pandas query plans via Gemini, executed in a
restricted sandbox, then narrated as a business-language answer.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Safety: AST allow-list for model-generated code
# ---------------------------------------------------------------------------

_BLOCKED_NAME_PREFIXES = ("__",)
_BLOCKED_NAMES = {
    "open",
    "eval",
    "exec",
    "compile",
    "input",
    "exit",
    "quit",
    "help",
    "breakpoint",
    "memoryview",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "classmethod",
    "staticmethod",
    "property",
    "super",
    "type",
    "__import__",
    "importlib",
    "os",
    "sys",
    "subprocess",
    "pathlib",
    "socket",
    "shutil",
    "pickle",
    "requests",
    "urllib",
    "http",
}
_BLOCKED_NODE_TYPES = (
    ast.Import,
    ast.ImportFrom,
    ast.With,
    ast.AsyncWith,
    ast.Raise,
    ast.Try,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    # Lambda is allowed — common in pandas groupby/apply expressions.
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
)

from genai.llm_client import get_gemini_model

GEMINI_MODEL = get_gemini_model()


class UnsafeQueryCodeError(ValueError):
    """Raised when generated code fails the AST safety checks."""


def _call_llm(system: str, user: str) -> str:
    from genai.llm_client import call_llm

    return call_llm(system, user, max_output_tokens=2048)


def _schema_from_dataframe(df: pd.DataFrame) -> dict[str, str]:
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def _sample_rows_from_dataframe(df: pd.DataFrame, n: int = 5) -> list[dict[str, Any]]:
    return df.head(n).where(pd.notnull(df.head(n)), None).to_dict(orient="records")


def validate_query_code(code: str) -> ast.Module:
    """
    Parse model-generated code and reject anything unsafe.

    Allowed: assignments and expressions that only reference ``df`` / ``pd``
    (plus safe builtins like ``len``, ``round``, ``list``, ``dict``, etc.).
    Blocked: imports, ``open``, dunder access, attribute chains into
    dangerous modules, function/class definitions, etc.
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise UnsafeQueryCodeError(f"Generated code has invalid syntax: {exc}") from exc

    if not tree.body:
        raise UnsafeQueryCodeError("Generated code is empty.")

    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AugAssign, ast.Expr, ast.AnnAssign)):
            raise UnsafeQueryCodeError(
                f"Only assignments and expressions are allowed; found {type(node).__name__}."
            )

    for node in ast.walk(tree):
        if isinstance(node, _BLOCKED_NODE_TYPES):
            raise UnsafeQueryCodeError(
                f"Disallowed AST node: {type(node).__name__}."
            )

        if isinstance(node, ast.Name) and (
            node.id in _BLOCKED_NAMES or node.id.startswith(_BLOCKED_NAME_PREFIXES)
        ):
            raise UnsafeQueryCodeError(f"Disallowed name: {node.id}")

        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise UnsafeQueryCodeError(f"Disallowed attribute access: {node.attr}")

        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BLOCKED_NAMES:
                raise UnsafeQueryCodeError(f"Disallowed call: {func.id}()")
            if isinstance(func, ast.Attribute) and func.attr in {"system", "popen", "remove", "unlink"}:
                raise UnsafeQueryCodeError(f"Disallowed call: .{func.attr}()")

    return tree


def execute_sandbox(code: str, df: pd.DataFrame) -> Any:
    """Execute validated code with an extremely limited namespace."""
    validate_query_code(code)

    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "True": True,
        "False": False,
        "None": None,
    }

    namespace: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "df": df.copy(),
        "pd": pd,
    }

    # Prefer an explicit `result = ...` if the model uses it; otherwise capture
    # the last expression via a rewritten tree.
    tree = ast.parse(code, mode="exec")
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body[-1]
        tree.body[-1] = ast.Assign(
            targets=[ast.Name(id="_result", ctx=ast.Store())],
            value=last.value,
        )
        ast.fix_missing_locations(tree)
        compiled = compile(tree, filename="<genai_query>", mode="exec")
        exec(compiled, namespace, namespace)  # noqa: S102 — intentionally sandboxed
        if "_result" in namespace:
            return namespace["_result"]

    compiled = compile(code, filename="<genai_query>", mode="exec")
    exec(compiled, namespace, namespace)  # noqa: S102 — intentionally sandboxed

    for key in ("result", "output", "answer", "out"):
        if key in namespace and key not in {"df", "pd"}:
            return namespace[key]

    # Fall back to any newly assigned non-internal name.
    created = [
        k
        for k in namespace
        if k not in {"df", "pd", "__builtins__", "_result"}
    ]
    if len(created) == 1:
        return namespace[created[0]]
    if created:
        return {k: namespace[k] for k in created}

    raise RuntimeError("Generated code ran but produced no captureable result.")


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"Model did not return JSON: {text[:200]}")
        text = text[start : end + 1]
    return json.loads(text)


def _result_to_preview(result: Any, max_rows: int = 20) -> Any:
    if isinstance(result, pd.DataFrame):
        return result.head(max_rows).to_dict(orient="records")
    if isinstance(result, pd.Series):
        return result.head(max_rows).to_dict()
    if hasattr(result, "item"):
        try:
            return result.item()
        except Exception:
            pass
    if isinstance(result, (list, dict, str, int, float, bool)) or result is None:
        return result
    return str(result)


_PLAN_SYSTEM = """\
You translate business questions into safe pandas analysis code.

Rules:
- Output ONLY a single JSON object with keys: code, chart_type, explanation
- code: Python statements that operate on an existing pandas DataFrame named `df`
  and the pandas module as `pd`. The final answer must be assigned to `result`
  OR be the last expression.
- Do NOT import anything. Do NOT use open, os, sys, eval, exec, or dunder attributes.
- Prefer groupby / value_counts / mean / sum / filters that answer the question.
- chart_type: one of "bar", "line", "scatter", "none"
- explanation: one short sentence describing what the code computes

Schema and sample rows are provided as context. The dataframe variable is already loaded as `df`.
"""


_NARRATE_SYSTEM = """\
You are a business analyst for a customer analytics platform.
Given a user question and a tabular result from an analysis query, respond with ONLY a JSON object:
{
  "answer": "2-3 sentence business-language answer (no jargon dump)",
  "chart_type": "bar|line|scatter|none",
  "chart_hint": "optional short hint for axes / labels"
}
Prefer chart_type "none" when the result is a single number or short text.
"""


def build_chart(result: Any, chart_type: str, title: str = "Query Result"):
    """
    Build a Plotly or matplotlib figure from a query result.
    Returns None when chart_type is none or the result is not plottable.
    """
    chart_type = (chart_type or "none").lower().strip()
    if chart_type in {"none", "", "null"}:
        return None

    plot_df: pd.DataFrame | None = None
    if isinstance(result, pd.DataFrame):
        plot_df = result.copy()
    elif isinstance(result, pd.Series):
        plot_df = result.reset_index()
        plot_df.columns = ["category", "value"] if plot_df.shape[1] == 2 else list(plot_df.columns)
    elif isinstance(result, dict):
        plot_df = pd.DataFrame(
            {"category": list(result.keys()), "value": list(result.values())}
        )
    else:
        return None

    if plot_df is None or plot_df.empty:
        return None

    try:
        import plotly.express as px
    except ImportError:
        return _matplotlib_chart(plot_df, chart_type, title)

    x_col = plot_df.columns[0]
    y_col = plot_df.columns[1] if plot_df.shape[1] > 1 else None

    if chart_type == "bar" and y_col:
        return px.bar(plot_df, x=x_col, y=y_col, title=title)
    if chart_type == "line" and y_col:
        return px.line(plot_df, x=x_col, y=y_col, title=title)
    if chart_type == "scatter" and y_col:
        return px.scatter(plot_df, x=x_col, y=y_col, title=title)
    if y_col:
        return px.bar(plot_df, x=x_col, y=y_col, title=title)
    return None


def _matplotlib_chart(plot_df: pd.DataFrame, chart_type: str, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = plot_df.iloc[:, 0]
    y = plot_df.iloc[:, 1] if plot_df.shape[1] > 1 else None
    if y is None:
        plt.close(fig)
        return None

    if chart_type == "line":
        ax.plot(x, y, marker="o")
    elif chart_type == "scatter":
        ax.scatter(x, y)
    else:
        ax.bar(x.astype(str), y)

    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return fig


def ask(
    question: str,
    df: pd.DataFrame,
    df_schema: dict | None = None,
    sample_rows: list | None = None,
    *,
    spark=None,
    log: bool = True,
) -> dict[str, Any]:
    """
    Answer a natural-language question against a pandas DataFrame.

    Pipeline:
      1. Gemini translates the question into a JSON query plan (pandas code).
      2. Code is AST-validated and executed in a sandbox ``{df, pd}`` only.
      3. Gemini narrates the result in business language and suggests a chart.
      4. Optionally logs question / code / result to ``genai_query_log``.

    Parameters
    ----------
    question : str
        Plain-English business question.
    df : pandas.DataFrame
        Working dataset. Spark DataFrames should be converted with ``.toPandas()``.
    df_schema : dict, optional
        Column → dtype mapping. Inferred from ``df`` when omitted.
    sample_rows : list, optional
        A few sample records for Gemini context. Taken from ``df.head(5)`` when omitted.
    spark : SparkSession, optional
        When provided (and ``log=True``), writes an audit row to the Delta log table.
    log : bool
        Persist an audit trail row when True.

    Returns
    -------
    dict
        Keys: question, code, chart_type, explanation, answer, result, result_preview, figure
    """
    if not question or not str(question).strip():
        raise ValueError("question must be a non-empty string.")

    schema = df_schema or _schema_from_dataframe(df)
    samples = sample_rows if sample_rows is not None else _sample_rows_from_dataframe(df)

    plan_user = (
        f"Question: {question}\n\n"
        f"Schema: {json.dumps(schema)}\n\n"
        f"Sample rows: {json.dumps(samples, default=str)}\n"
    )
    plan_raw = _call_llm(_PLAN_SYSTEM, plan_user)
    plan = _extract_json(plan_raw)

    code = str(plan.get("code", "")).strip()
    if not code:
        raise ValueError("Gemini returned a plan without code.")

    execution_error: str | None = None
    result: Any = None
    try:
        result = execute_sandbox(code, df)
    except Exception as exc:  # noqa: BLE001 — surface cleanly to callers / UI
        execution_error = f"{type(exc).__name__}: {exc}"
        result = None

    preview = _result_to_preview(result) if result is not None else {"error": execution_error}

    narrate_user = (
        f"Question: {question}\n\n"
        f"Generated code:\n{code}\n\n"
        f"Result preview (JSON):\n{json.dumps(preview, default=str)}\n"
    )
    if execution_error:
        narrate_user += f"\nExecution error: {execution_error}\n"

    try:
        narration = _extract_json(_call_llm(_NARRATE_SYSTEM, narrate_user))
    except Exception:
        narration = {
            "answer": (
                f"The query ran but narration failed. "
                f"Raw preview: {preview}"
                if not execution_error
                else f"Could not answer because code execution failed: {execution_error}"
            ),
            "chart_type": plan.get("chart_type", "none"),
            "chart_hint": "",
        }

    chart_type = str(narration.get("chart_type") or plan.get("chart_type") or "none")
    figure = None
    if result is not None and not execution_error:
        try:
            figure = build_chart(result, chart_type, title=question[:80])
        except Exception:
            figure = None

    payload = {
        "question": question,
        "code": code,
        "chart_type": chart_type,
        "explanation": plan.get("explanation", ""),
        "answer": narration.get("answer", ""),
        "chart_hint": narration.get("chart_hint", ""),
        "result": result,
        "result_preview": preview,
        "figure": figure,
        "error": execution_error,
    }

    if log:
        try:
            from genai.query_logger import log_query

            log_query(
                question=question,
                code=code,
                result_preview=preview,
                answer=payload["answer"],
                chart_type=chart_type,
                error=execution_error,
                spark=spark,
            )
        except Exception:
            # Audit logging must never break the interactive QA path.
            pass

    return payload
