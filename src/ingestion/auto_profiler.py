"""
===========================================================
Project : Customer Churn Analytics Platform
Module  : Auto Profiler
Author  : Shivek Singh
===========================================================

Custom Spark-native dataset profiler: semantic column typing, target
detection, null/cardinality summaries, class balance, and correlation heatmaps.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
    TimestampType,
)

TARGET_NAME_PATTERNS = (
    "churn",
    "target",
    "label",
    "y",
    "default",
    "outcome",
    "class",
    "is_churn",
    "churned",
    "fraud",
    "response",
)
ID_NAME_PATTERNS = ("id", "uuid", "guid", "key", "code", "number")
DATETIME_NAME_HINTS = ("date", "time", "timestamp", "created", "updated", "dob")
FREE_TEXT_MIN_AVG_LENGTH = 40
CATEGORICAL_RATIO_THRESHOLD = 0.05
ID_LIKE_RATIO_THRESHOLD = 0.95
CORRELATION_SAMPLE_SIZE = 5_000


class ColumnSemanticType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    FREE_TEXT = "free_text"
    ID_LIKE = "id_like"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


@dataclass
class ColumnProfile:
    name: str
    spark_type: str
    semantic_type: ColumnSemanticType
    null_count: int
    null_pct: float
    distinct_count: int
    distinct_ratio: float
    sample_values: list[Any] = field(default_factory=list)
    is_binary: bool = False


@dataclass
class TargetDetectionResult:
    target_column: str | None
    candidates: list[str]
    requires_user_selection: bool
    reason: str


@dataclass
class DatasetProfile:
    row_count: int
    column_count: int
    columns: list[ColumnProfile]
    target: TargetDetectionResult
    null_summary: dict[str, int]
    cardinality_summary: dict[str, int]
    class_balance: dict[str, float] | None
    correlation_matrix: dict[str, dict[str, float | None]]
    numeric_columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["columns"] = [
            {**asdict(col), "semantic_type": col.semantic_type.value}
            for col in self.columns
        ]
        payload["target"]["candidates"] = list(self.target.candidates)
        return payload

    def summary_markdown(self) -> str:
        lines = [
            "# Dataset Profile",
            "",
            f"- **Rows:** {self.row_count:,}",
            f"- **Columns:** {self.column_count}",
            f"- **Suggested target:** {self.target.target_column or 'ambiguous — user selection required'}",
            "",
            "## Column Overview",
            "",
            "| Column | Spark Type | Semantic Type | Null % | Distinct | Binary |",
            "|--------|------------|---------------|--------|----------|--------|",
        ]

        for col in self.columns:
            lines.append(
                f"| {col.name} | {col.spark_type} | {col.semantic_type.value} | "
                f"{col.null_pct:.1f}% | {col.distinct_count:,} | "
                f"{'yes' if col.is_binary else 'no'} |"
            )

        if self.class_balance:
            lines.extend(["", "## Class Balance", ""])
            for label, pct in sorted(self.class_balance.items(), key=lambda item: item[1], reverse=True):
                lines.append(f"- **{label}:** {pct:.2f}%")

        if self.numeric_columns:
            lines.extend(
                [
                    "",
                    "## Numeric Columns",
                    "",
                    ", ".join(f"`{name}`" for name in self.numeric_columns),
                ]
            )

        return "\n".join(lines)


def _is_numeric_type(dtype: Any) -> bool:
    return isinstance(
        dtype,
        (IntegerType, LongType, ShortType, FloatType, DoubleType, DecimalType),
    )


def _is_datetime_type(dtype: Any) -> bool:
    return isinstance(dtype, (DateType, TimestampType))


def _name_matches(name: str, patterns: tuple[str, ...]) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return any(
        normalized == pattern
        or normalized.endswith(f"_{pattern}")
        or normalized.startswith(f"{pattern}_")
        for pattern in patterns
    )


def _column_stats(df: DataFrame, column: str, row_count: int) -> dict[str, Any]:
    dtype = dict(df.dtypes)[column]
    spark_type = df.schema[column].dataType

    stats_row = (
        df.select(
            F.count(F.when(F.col(column).isNull(), F.lit(1))).alias("null_count"),
            F.approx_count_distinct(column).alias("distinct_count"),
            F.avg(F.length(F.col(column).cast("string"))).alias("avg_length"),
        )
        .collect()[0]
    )

    null_count = int(stats_row["null_count"] or 0)
    distinct_count = int(stats_row["distinct_count"] or 0)
    distinct_ratio = distinct_count / row_count if row_count else 0.0
    null_pct = (null_count / row_count * 100) if row_count else 0.0
    avg_length = float(stats_row["avg_length"] or 0.0)

    sample_values = [
        row[column]
        for row in df.select(column).where(F.col(column).isNotNull()).limit(5).collect()
    ]

    non_null_count = row_count - null_count
    is_binary = non_null_count > 0 and distinct_count == 2

    semantic_type = _infer_semantic_type(
        column_name=column,
        spark_type=spark_type,
        dtype_label=dtype,
        distinct_ratio=distinct_ratio,
        avg_length=avg_length,
        is_binary=is_binary,
    )

    return {
        "name": column,
        "spark_type": dtype,
        "semantic_type": semantic_type,
        "null_count": null_count,
        "null_pct": null_pct,
        "distinct_count": distinct_count,
        "distinct_ratio": distinct_ratio,
        "sample_values": sample_values,
        "is_binary": is_binary,
    }


def _infer_semantic_type(
    column_name: str,
    spark_type: Any,
    dtype_label: str,
    distinct_ratio: float,
    avg_length: float,
    is_binary: bool,
) -> ColumnSemanticType:
    if isinstance(spark_type, BooleanType) or is_binary:
        return ColumnSemanticType.BOOLEAN

    if _is_datetime_type(spark_type) or _name_matches(column_name, DATETIME_NAME_HINTS):
        return ColumnSemanticType.DATETIME

    if _is_numeric_type(spark_type):
        if distinct_ratio >= ID_LIKE_RATIO_THRESHOLD and _name_matches(column_name, ID_NAME_PATTERNS):
            return ColumnSemanticType.ID_LIKE
        return ColumnSemanticType.NUMERIC

    if isinstance(spark_type, StringType):
        if distinct_ratio >= ID_LIKE_RATIO_THRESHOLD and _name_matches(column_name, ID_NAME_PATTERNS):
            return ColumnSemanticType.ID_LIKE
        if distinct_ratio <= CATEGORICAL_RATIO_THRESHOLD:
            return ColumnSemanticType.CATEGORICAL
        if avg_length >= FREE_TEXT_MIN_AVG_LENGTH and distinct_ratio > CATEGORICAL_RATIO_THRESHOLD:
            return ColumnSemanticType.FREE_TEXT
        if distinct_ratio <= 0.5:
            return ColumnSemanticType.CATEGORICAL
        return ColumnSemanticType.FREE_TEXT

    return ColumnSemanticType.UNKNOWN


def detect_target_column(
    df: DataFrame,
    profiles: list[ColumnProfile] | None = None,
    user_selection: str | None = None,
) -> TargetDetectionResult:
    """
    Detect a likely label/target column.

    Prefers binary columns whose names match common target patterns. When
    multiple candidates exist, ``requires_user_selection`` is set so the caller
    can present a dropdown (e.g. via ``dbutils.widgets`` or Jupyter widgets).
    """
    if user_selection:
        if user_selection not in df.columns:
            raise ValueError(f"Selected target column '{user_selection}' not found in dataframe.")
        return TargetDetectionResult(
            target_column=user_selection,
            candidates=[user_selection],
            requires_user_selection=False,
            reason="User provided an explicit target column.",
        )

    profiles = profiles or [
        ColumnProfile(**_column_stats(df, col, df.count())) for col in df.columns
    ]
    binary_columns = [profile.name for profile in profiles if profile.is_binary]

    named_matches = [
        profile.name
        for profile in profiles
        if profile.is_binary and _name_matches(profile.name, TARGET_NAME_PATTERNS)
    ]

    if len(named_matches) == 1:
        return TargetDetectionResult(
            target_column=named_matches[0],
            candidates=named_matches,
            requires_user_selection=False,
            reason="Single binary column matches common target naming patterns.",
        )

    if len(named_matches) > 1:
        return TargetDetectionResult(
            target_column=None,
            candidates=named_matches,
            requires_user_selection=True,
            reason="Multiple binary columns match target naming patterns.",
        )

    if len(binary_columns) == 1:
        return TargetDetectionResult(
            target_column=binary_columns[0],
            candidates=binary_columns,
            requires_user_selection=False,
            reason="Single binary column detected.",
        )

    if len(binary_columns) > 1:
        return TargetDetectionResult(
            target_column=None,
            candidates=binary_columns,
            requires_user_selection=True,
            reason="Multiple binary columns detected; user selection recommended.",
        )

    return TargetDetectionResult(
        target_column=None,
        candidates=[],
        requires_user_selection=True,
        reason="No binary target column detected.",
    )


def _class_balance(df: DataFrame, target_column: str) -> dict[str, float]:
    total = df.count()
    if total == 0:
        return {}

    rows = (
        df.groupBy(target_column)
        .count()
        .orderBy(F.desc("count"))
        .collect()
    )
    return {str(row[target_column]): (row["count"] / total) * 100 for row in rows}


def _numeric_columns(profiles: list[ColumnProfile]) -> list[str]:
    return [
        profile.name
        for profile in profiles
        if profile.semantic_type in {ColumnSemanticType.NUMERIC, ColumnSemanticType.BOOLEAN}
    ]


def _correlation_matrix(df: DataFrame, numeric_columns: list[str]) -> dict[str, dict[str, float | None]]:
    if len(numeric_columns) < 2:
        return {}

    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.stat import Correlation

    sampled = df
    row_count = df.count()
    if row_count > CORRELATION_SAMPLE_SIZE:
        fraction = CORRELATION_SAMPLE_SIZE / row_count
        sampled = df.sample(withReplacement=False, fraction=fraction, seed=42)

    casted = sampled.select(
        *[F.col(name).cast("double").alias(name) for name in numeric_columns]
    ).dropna()

    if casted.limit(2).count() < 2:
        return {}

    assembler = VectorAssembler(inputCols=numeric_columns, outputCol="_features", handleInvalid="skip")
    vector_df = assembler.transform(casted)
    correlation = Correlation.corr(vector_df, "_features").head()[0]
    if correlation is None:
        return {}

    values = correlation.toArray()
    matrix: dict[str, dict[str, float | None]] = {}
    for i, col_i in enumerate(numeric_columns):
        matrix[col_i] = {}
        for j, col_j in enumerate(numeric_columns):
            matrix[col_i][col_j] = float(values[i, j])
    return matrix


def profile_dataframe(
    df: DataFrame,
    target_column: str | None = None,
) -> DatasetProfile:
    """
    Profile a Spark DataFrame and return a structured dataset report.
    """
    row_count = df.count()
    profiles = [ColumnProfile(**_column_stats(df, column, row_count)) for column in df.columns]
    target = detect_target_column(df, profiles=profiles, user_selection=target_column)

    resolved_target = target.target_column
    class_balance = _class_balance(df, resolved_target) if resolved_target else None
    numeric_columns = _numeric_columns(profiles)

    return DatasetProfile(
        row_count=row_count,
        column_count=len(df.columns),
        columns=profiles,
        target=target,
        null_summary={profile.name: profile.null_count for profile in profiles},
        cardinality_summary={profile.name: profile.distinct_count for profile in profiles},
        class_balance=class_balance,
        correlation_matrix=_correlation_matrix(df, numeric_columns),
        numeric_columns=numeric_columns,
    )


def plot_correlation_heatmap(
    profile: DatasetProfile,
    figsize: tuple[float, float] = (10, 8),
):
    """
    Render a matplotlib heatmap for the numeric correlation matrix.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    columns = profile.numeric_columns
    if len(columns) < 2 or not profile.correlation_matrix:
        raise ValueError("At least two numeric columns are required for a correlation heatmap.")

    matrix = np.array(
        [[profile.correlation_matrix[row][col] for col in columns] for row in columns],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)
    ax.set_title("Numeric Feature Correlation Heatmap")

    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
