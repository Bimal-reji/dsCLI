"""Data cleaning.

A pragmatic cleaning step that is safe to run on any dataset:

* drops duplicate rows (configurable),
* drops columns that are almost entirely missing,
* imputes or drops remaining missing values,
* optionally caps outliers (interquartile range or z-score),
* coerces numeric-looking strings to numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dscli.errors import ConfigError


@dataclass
class CleaningReport:
    """What the cleaning step did, for display and logging."""

    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    duplicates_dropped: int = 0
    columns_dropped: list[str] | None = None
    missing_imputed: int = 0
    rows_dropped_missing: int = 0
    outliers_capped: int = 0
    notes: list[str] | None = None


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns that look numeric (e.g. '1,234' or ' 42 ') to numbers."""
    result = df.copy()
    for col in result.columns:
        if pd.api.types.is_numeric_dtype(result[col]):
            continue
        sample = result[col].dropna().astype(str).head(100)
        if sample.empty:
            continue
        cleaned = sample.str.replace(r"[,\s]", "", regex=True)
        try:
            converted = pd.to_numeric(cleaned)
        except (ValueError, TypeError):
            continue
        if converted.notna().sum() / len(cleaned) >= 0.9:
            result[col] = pd.to_numeric(
                result[col].astype(str).str.replace(r"[,\s]", "", regex=True),
                errors="coerce",
            )
    return result


def _handle_missing(
    df: pd.DataFrame, strategy: str, constant: float, drop_threshold: float
) -> tuple[pd.DataFrame, CleaningReport]:
    """Impute or drop missing values according to ``strategy``."""
    report = CleaningReport(
        rows_before=len(df),
        rows_after=len(df),
        columns_before=len(df.columns),
        columns_after=len(df.columns),
    )
    if df.empty:
        return df, report

    # 1. Drop columns that are almost entirely missing.
    missing_pct = df.isna().mean()
    to_drop = list(missing_pct[missing_pct >= drop_threshold].index)
    if to_drop:
        df = df.drop(columns=to_drop)
        report.columns_dropped = to_drop

    # 2. Handle remaining missing values.
    missing_counts = df.isna().sum()
    total_missing = int(missing_counts.sum())

    if strategy == "drop":
        before = len(df)
        df = df.dropna()
        report.rows_dropped_missing = before - len(df)
        report.notes = [f"Dropped {report.rows_dropped_missing} row(s) with missing values."]
    else:
        if total_missing == 0:
            return df, report

        numeric_cols = df.select_dtypes(include="number").columns
        categorical_cols = df.select_dtypes(exclude="number").columns

        if strategy == "mean":
            for col in numeric_cols:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            for col in numeric_cols:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].median())
        elif strategy == "constant":
            df = df.fillna(constant)
        elif strategy == "most_frequent":
            for col in df.columns:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "")
        else:
            raise ConfigError(
                f"Unknown missing_strategy '{strategy}'. "
                "Use one of: mean, median, most_frequent, constant, drop."
            )
        report.notes = [f"Imputed missing values using strategy '{strategy}'."]

    report.missing_imputed = total_missing - report.rows_dropped_missing
    report.rows_after = len(df)
    report.columns_after = len(df.columns)
    return df, report


def _cap_outliers(df: pd.DataFrame, method: str, threshold: float) -> tuple[pd.DataFrame, int]:
    """Cap extreme values in numeric columns; returns (df, capped_count)."""
    if method is None or df.empty:
        return df, 0
    if method not in ("iqr", "zscore"):
        raise ConfigError(f"Unknown outlier_method '{method}'. Use 'iqr', 'zscore', or null.")

    result = df.copy()
    capped = 0
    for col in result.select_dtypes(include="number").columns:
        series = result[col].dropna()
        if series.empty:
            continue
        if method == "iqr":
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
        else:  # zscore
            mean, std = series.mean(), series.std()
            if std == 0 or np.isnan(std):
                continue
            lower, upper = mean - threshold * std, mean + threshold * std

        mask_low = result[col] < lower
        mask_high = result[col] > upper
        capped += int(mask_low.sum() + mask_high.sum())
        result.loc[mask_low, col] = lower
        result.loc[mask_high, col] = upper
    return result, capped


def clean_dataframe(
    df: pd.DataFrame,
    *,
    drop_duplicates: bool = True,
    missing_strategy: str = "mean",
    missing_constant: float = 0.0,
    drop_high_missing_threshold: float = 0.8,
    outlier_method: str | None = None,
    outlier_threshold: float = 3.0,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean a DataFrame according to the given options.

    Returns the cleaned DataFrame and a :class:`CleaningReport` describing
    what changed.
    """
    report = CleaningReport(
        rows_before=len(df), rows_after=len(df),
        columns_before=len(df.columns), columns_after=len(df.columns),
    )
    if df.empty:
        return df, report

    result = _coerce_numeric(df)

    if drop_duplicates:
        before = len(result)
        result = result.drop_duplicates()
        report.duplicates_dropped = before - len(result)

    result, missing_report = _handle_missing(
        result, missing_strategy, missing_constant, drop_high_missing_threshold
    )
    result, capped = _cap_outliers(result, outlier_method, outlier_threshold)
    report.outliers_capped = capped

    report.rows_after = len(result)
    report.columns_after = len(result.columns)
    report.columns_dropped = missing_report.columns_dropped or report.columns_dropped
    report.missing_imputed = missing_report.missing_imputed
    report.rows_dropped_missing = missing_report.rows_dropped_missing
    report.notes = missing_report.notes
    return result, report
