"""Dataset loading.

Supports CSV, Parquet, Excel, and JSON/JSON-lines files. CSV files are read
with a small auto-detection step for separator and encoding so that files
exported by Excel or other tools "just work".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dscli.errors import DataError
from dscli.utils.io import ensure_readable_file

_SUPPORTED_SUFFIXES = {".csv", ".tsv", ".parquet", ".xlsx", ".xls", ".json", ".jsonl", ".ndjson"}

# Encodings to try for CSV files, in order of likelihood.
_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1"]
_SEPARATORS = [",", ";", "\t", "|"]


def _detect_separator(path: str | Path, encoding: str) -> str:
    """Guess the separator from the first line of a CSV file."""
    with open(path, encoding=encoding) as handle:
        first_line = handle.readline()
    counts = {sep: first_line.count(sep) for sep in _SEPARATORS}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def read_csv_auto(path: str | Path) -> pd.DataFrame:
    """Read a CSV/TSV file, auto-detecting separator and encoding."""
    p = Path(path)
    last_error: Exception | None = None
    for encoding in _ENCODINGS:
        separator = _detect_separator(p, encoding)
        try:
            return pd.read_csv(p, sep=separator, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except pd.errors.ParserError as exc:
            last_error = exc
            # The detected separator failed; fall back to trying the others.
            for sep in _SEPARATORS:
                if sep == separator:
                    continue
                try:
                    return pd.read_csv(p, sep=sep, encoding=encoding)
                except (UnicodeDecodeError, pd.errors.ParserError) as inner:
                    last_error = inner
                    continue
    raise DataError(
        f"Could not parse CSV file '{p}'. Tried separators "
        f"{', '.join(repr(s) for s in _SEPARATORS)} with common encodings. "
        f"Last error: {last_error}"
    )


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a dataset from disk into a pandas DataFrame.

    Raises :class:`DataError` for unsupported formats or parse failures.
    """
    p = ensure_readable_file(path, description="dataset")
    suffix = p.suffix.lower()

    try:
        if suffix == ".csv" or suffix == ".tsv":
            df = read_csv_auto(p)
        elif suffix == ".parquet":
            df = pd.read_parquet(p)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(p)
        elif suffix == ".json":
            df = pd.read_json(p)
        elif suffix in (".jsonl", ".ndjson"):
            df = pd.read_json(p, lines=True)
        else:
            raise DataError(
                f"Unsupported dataset format: '{suffix}'. Supported formats: "
                f"{', '.join(sorted(_SUPPORTED_SUFFIXES))}."
            )
    except DataError:
        raise
    except Exception as exc:
        raise DataError(f"Failed to load dataset '{p}': {exc}") from exc

    if df.empty:
        raise DataError(f"Dataset '{p}' loaded but contains no rows.")

    df.columns = [str(col) for col in df.columns]
    return df


def save_dataframe(df: pd.DataFrame, path: str | Path, overwrite: bool = False) -> Path:
    """Write a DataFrame to CSV, honoring overwrite protection."""
    from dscli.utils.io import check_can_write

    p = check_can_write(path, overwrite, description="dataset")
    df.to_csv(p, index=False)
    return p


def describe_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Compute a compact summary of a dataset for display by ``data info``."""
    missing = int(df.isna().sum().sum())
    total = int(df.size)
    missing_pct = round(100.0 * missing / total, 2) if total else 0.0
    numeric_cols = list(df.select_dtypes(include="number").columns)
    categorical_cols = list(df.select_dtypes(exclude="number").columns)
    duplicates = int(df.duplicated().sum())
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_values": missing,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicates,
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }
