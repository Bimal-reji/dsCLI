"""Dataset validation.

Validation produces a :class:`ValidationReport` describing problems rather
than raising immediately, so the user sees everything that is wrong at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dscli.errors import ValidationError


@dataclass
class ValidationReport:
    """Result of validating a DataFrame against expectations."""

    rows: int
    columns: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)
    columns_with_missing: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    constant_columns: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def merge(self, other: "ValidationReport") -> "ValidationReport":
        """Combine two reports into one (used when validating train+test)."""
        return ValidationReport(
            rows=self.rows + other.rows,
            columns=max(self.columns, other.columns),
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            missing_columns=self.missing_columns + other.missing_columns,
            columns_with_missing={**self.columns_with_missing, **other.columns_with_missing},
            duplicate_rows=self.duplicate_rows + other.duplicate_rows,
            constant_columns=self.constant_columns + other.constant_columns,
        )


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def validate_dataframe(
    df: pd.DataFrame,
    *,
    required_columns: list[str] | None = None,
    min_rows: int = 1,
    max_missing_pct: float = 1.0,
) -> ValidationReport:
    """Validate a DataFrame, returning a report of problems found.

    Parameters
    ----------
    df:
        The DataFrame to validate.
    required_columns:
        Columns that must be present (e.g. the target column).
    min_rows:
        Minimum acceptable number of rows.
    max_missing_pct:
        Warn when any column exceeds this fraction of missing values (0-1).
    """
    report = ValidationReport(rows=len(df), columns=len(df.columns))

    if len(df) < min_rows:
        report.errors.append(f"Dataset has {len(df)} rows; at least {min_rows} required.")

    if required_columns:
        missing_cols = [c for c in required_columns if c not in df.columns]
        if missing_cols:
            report.missing_columns = missing_cols
            report.errors.append(
                f"Missing required column(s): {', '.join(missing_cols)}. "
                f"Available columns: {', '.join(df.columns)}."
            )

    if df.empty:
        return report

    missing_counts = df.isna().sum()
    report.columns_with_missing = {
        col: int(count) for col, count in missing_counts.items() if count > 0
    }
    for col, count in report.columns_with_missing.items():
        pct = count / len(df)
        if pct > max_missing_pct:
            report.errors.append(
                f"Column '{col}' has {pct:.0%} missing values, exceeding the "
                f"allowed {max_missing_pct:.0%}."
            )
        elif pct > 0.05:
            report.warnings.append(
                f"Column '{col}' has {pct:.1%} missing values ({count} rows)."
            )

    report.duplicate_rows = int(df.duplicated().sum())
    if report.duplicate_rows:
        report.warnings.append(
            f"Found {report.duplicate_rows} duplicate row(s). Run 'dscli data clean' "
            "to remove them."
        )

    constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
    report.constant_columns = constant_cols
    if constant_cols:
        report.warnings.append(
            "Constant column(s) carry no information: "
            + ", ".join(constant_cols)
        )

    report.errors = _dedupe(report.errors)
    report.warnings = _dedupe(report.warnings)
    return report


def validate_target(df: pd.DataFrame, target: str | None) -> tuple[str, str]:
    """Validate the target column and infer the ML task type.

    Returns ``(target_name, task)`` where task is ``"classification"`` or
    ``"regression"``. Raises :class:`ValidationError` with a clear message
    when the target is missing or unusable.
    """
    if not target:
        raise ValidationError(
            "No target column configured. Set 'data.target' in configs/config.yaml "
            "or pass --target."
        )
    if target not in df.columns:
        raise ValidationError(
            f"Target column '{target}' not found in dataset. "
            f"Available columns: {', '.join(df.columns)}."
        )

    series = df[target]
    if series.isna().all():
        raise ValidationError(f"Target column '{target}' is entirely missing values.")

    if pd.api.types.is_numeric_dtype(series):
        unique = series.dropna().nunique()
        if unique == 1:
            raise ValidationError(
                f"Target column '{target}' has only one unique value; "
                "no model can learn from it."
            )
        if unique <= 20 and unique / len(series) < 0.5:
            return target, "classification"
        return target, "regression"

    # Non-numeric target -> classification.
    unique = series.dropna().nunique()
    if unique == 1:
        raise ValidationError(
            f"Target column '{target}' has only one unique value; "
            "no model can learn from it."
        )
    return target, "classification"


def check_class_balance(df: pd.DataFrame, target: str) -> dict[str, int]:
    """Return per-class counts for a classification target."""
    return df[target].value_counts().to_dict()


def is_binary_classification(y: pd.Series) -> bool:
    """True when ``y`` looks like a binary classification target."""
    unique = pd.Series(y).dropna().unique()
    return len(unique) == 2


def encode_labels(y: pd.Series) -> tuple[np.ndarray, list[Any]]:
    """Encode categorical labels to integers; returns (encoded, classes)."""
    classes = list(pd.unique(y.dropna()))
    mapping = {label: i for i, label in enumerate(classes)}
    return np.array([mapping[v] for v in y]), classes
