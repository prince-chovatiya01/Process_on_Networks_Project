"""
Dataset validation and schema integrity checking module.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class ValidationReport:
    dataset_name: str
    total_raw_rows: int
    missing_value_rows: int
    self_loop_rows: int
    duplicate_rows: int
    clean_event_count: int
    unique_nodes: int
    temporal_span: Dict[str, Any]
    duration_semantic: str
    assumptions: List[str]
    is_valid: bool


def validate_event_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
    duration_semantic: str = "instantaneous",
    time_col: str = "timestamp",
    src_col: str = "src",
    dst_col: str = "dst",
    end_time_col: Optional[str] = None
) -> (pd.DataFrame, ValidationReport):
    """
    Validate, sanitize, and produce a validation report for a raw event DataFrame.
    """
    total_raw_rows = len(df)
    assumptions: List[str] = []

    # 1. Missing values check
    cols_to_check = [src_col, dst_col, time_col]
    if end_time_col and end_time_col in df.columns:
        cols_to_check.append(end_time_col)
    
    missing_mask = df[cols_to_check].isnull().any(axis=1)
    missing_count = int(missing_mask.sum())
    clean_df = df[~missing_mask].copy()

    # 2. Self loops check
    self_loop_mask = clean_df[src_col] == clean_df[dst_col]
    self_loop_count = int(self_loop_mask.sum())
    clean_df = clean_df[~self_loop_mask].copy()

    # 3. Duplicate events check
    dup_cols = [src_col, dst_col, time_col]
    if end_time_col and end_time_col in clean_df.columns:
        dup_cols.append(end_time_col)
    dup_mask = clean_df.duplicated(subset=dup_cols, keep="first")
    duplicate_count = int(dup_mask.sum())
    clean_df = clean_df[~dup_mask].copy()

    # 4. Cast types and sort chronologically
    clean_df[time_col] = clean_df[time_col].astype(int)
    if end_time_col and end_time_col in clean_df.columns:
        clean_df[end_time_col] = clean_df[end_time_col].astype(int)
        # Ensure end_time >= start_time
        invalid_duration = (clean_df[end_time_col] < clean_df[time_col]).sum()
        if invalid_duration > 0:
            assumptions.append(f"Filtered {invalid_duration} rows where end_time < start_time")
            clean_df = clean_df[clean_df[end_time_col] >= clean_df[time_col]].copy()

    clean_df = clean_df.sort_values(by=time_col).reset_index(drop=True)

    # 5. Extract unique nodes
    unique_nodes = set(clean_df[src_col].unique()).union(set(clean_df[dst_col].unique()))
    node_count = len(unique_nodes)

    # 6. Temporal span
    t_min = int(clean_df[time_col].min()) if not clean_df.empty else 0
    t_max = int(clean_df[time_col].max()) if not clean_df.empty else 0
    span_info = {
        "t_min": t_min,
        "t_max": t_max,
        "span_seconds": t_max - t_min if not clean_df.empty else 0
    }

    if duration_semantic == "instantaneous":
        assumptions.append(
            "Events represent instantaneous interaction points with no duration recorded in raw source."
        )
    elif duration_semantic == "interval":
        assumptions.append(
            "Events represent active contact intervals with physical duration recorded."
        )

    report = ValidationReport(
        dataset_name=dataset_name,
        total_raw_rows=total_raw_rows,
        missing_value_rows=missing_count,
        self_loop_rows=self_loop_count,
        duplicate_rows=duplicate_count,
        clean_event_count=len(clean_df),
        unique_nodes=node_count,
        temporal_span=span_info,
        duration_semantic=duration_semantic,
        assumptions=assumptions,
        is_valid=(len(clean_df) > 0 and node_count > 1)
    )

    return clean_df, report


def generate_markdown_report(report: ValidationReport) -> str:
    """Format ValidationReport as a clear Markdown document."""
    lines = [
        f"# Dataset Validation Report: {report.dataset_name}",
        "",
        "## Summary Metrics",
        f"- **Raw Rows**: {report.total_raw_rows}",
        f"- **Missing Value Rows Dropped**: {report.missing_value_rows}",
        f"- **Self-loops Dropped**: {report.self_loop_rows}",
        f"- **Duplicate Events Dropped**: {report.duplicate_rows}",
        f"- **Clean Event Count**: {report.clean_event_count}",
        f"- **Unique Nodes**: {report.unique_nodes}",
        f"- **Duration Semantics**: `{report.duration_semantic}`",
        f"- **Temporal Span**: Start={report.temporal_span.get('t_min')}, End={report.temporal_span.get('t_max')} (Delta: {report.temporal_span.get('span_seconds')} units)",
        f"- **Integrity Status**: {'PASSED' if report.is_valid else 'FAILED'}",
        "",
        "## Methodological Assumptions & Disclaimers",
    ]
    for asm in report.assumptions:
        lines.append(f"- {asm}")
    lines.append("")
    return "\n".join(lines)
