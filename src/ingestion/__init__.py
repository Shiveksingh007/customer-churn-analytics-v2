"""Dataset ingestion and profiling for the Customer Churn Analytics Platform."""

from ingestion.auto_profiler import (
    ColumnProfile,
    ColumnSemanticType,
    DatasetProfile,
    TargetDetectionResult,
    detect_target_column,
    plot_correlation_heatmap,
    profile_dataframe,
)
from ingestion.loaders import load_from_kaggle, load_from_upload, load_from_url

__all__ = [
    "ColumnProfile",
    "ColumnSemanticType",
    "DatasetProfile",
    "TargetDetectionResult",
    "detect_target_column",
    "load_from_kaggle",
    "load_from_upload",
    "load_from_url",
    "plot_correlation_heatmap",
    "profile_dataframe",
]
