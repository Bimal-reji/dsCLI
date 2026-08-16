"""Tests for the data package: loader, validator, cleaner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dscli.data.cleaner import clean_dataframe
from dscli.data.demo import generate_demo_dataset
from dscli.data.loader import describe_dataset, load_dataframe
from dscli.data.validator import validate_dataframe, validate_target
from dscli.errors import DataError, ValidationError


# -- loader ----------------------------------------------------------------


def test_load_csv(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")
    df = load_dataframe(path)
    assert df.shape == (2, 3)
    assert list(df.columns) == ["a", "b", "c"]


def test_load_csv_autodetects_semicolon(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a;b;c\n1;2;3\n", encoding="utf-8")
    df = load_dataframe(path)
    assert list(df.columns) == ["a", "b", "c"]


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(DataError, match="not found"):
        load_dataframe(tmp_path / "missing.csv")


def test_load_unsupported_format_raises(tmp_path):
    path = tmp_path / "data.xyz"
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(DataError, match="Unsupported dataset format"):
        load_dataframe(path)


def test_load_empty_csv_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("a,b\n", encoding="utf-8")
    with pytest.raises(DataError, match="no rows"):
        load_dataframe(path)


def test_describe_dataset():
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": ["x", "x", "y"]})
    info = describe_dataset(df)
    assert info["rows"] == 3
    assert info["columns"] == 2
    assert info["missing_values"] == 1
    assert info["numeric_columns"] == ["a"]
    assert info["categorical_columns"] == ["b"]


# -- validator -------------------------------------------------------------


def test_validate_dataframe_ok():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    report = validate_dataframe(df, required_columns=["a"])
    assert report.is_valid
    assert report.errors == []


def test_validate_missing_required_column():
    df = pd.DataFrame({"a": [1, 2, 3]})
    report = validate_dataframe(df, required_columns=["target"])
    assert not report.is_valid
    assert "target" in report.errors[0]


def test_validate_warns_about_duplicates_and_constants():
    df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 1, 2]})
    report = validate_dataframe(df)
    assert report.duplicate_rows == 1
    assert report.constant_columns == ["a"]
    assert any("duplicate" in w for w in report.warnings)


def test_validate_target_missing():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValidationError, match="No target column configured"):
        validate_target(df, None)
    with pytest.raises(ValidationError, match="not found"):
        validate_target(df, "nope")


def test_validate_target_task_inference():
    df = pd.DataFrame({"y": [0, 1, 0, 1, 1, 0] * 5})
    assert validate_target(df, "y") == ("y", "classification")
    df2 = pd.DataFrame({"y": np.linspace(0, 100, 100)})
    assert validate_target(df2, "y") == ("y", "regression")
    df3 = pd.DataFrame({"y": ["cat", "dog", "cat", "dog"] * 5})
    assert validate_target(df3, "y") == ("y", "classification")


def test_validate_target_single_class_raises():
    df = pd.DataFrame({"y": [1, 1, 1, 1]})
    with pytest.raises(ValidationError, match="only one unique value"):
        validate_target(df, "y")


# -- cleaner ---------------------------------------------------------------


def test_clean_drops_duplicates_and_imputes_mean():
    df = pd.DataFrame(
        {"a": [1.0, 1.0, np.nan, 4.0], "b": ["x", "x", "y", "z"]}
    )
    cleaned, report = clean_dataframe(df, missing_strategy="mean")
    assert report.duplicates_dropped == 1
    assert cleaned["a"].isna().sum() == 0
    assert cleaned["a"].iloc[1] == pytest.approx(2.5)  # mean of 1, 4 = 2.5
    assert report.rows_after == 3


def test_clean_drop_strategy_removes_rows():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    cleaned, report = clean_dataframe(df, missing_strategy="drop")
    assert report.rows_dropped_missing == 1
    assert len(cleaned) == 2


def test_clean_drops_high_missing_columns():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [np.nan, np.nan], "c": [3.0, 4.0]})
    cleaned, report = clean_dataframe(df, drop_high_missing_threshold=0.5)
    assert "b" not in cleaned.columns
    assert report.columns_dropped == ["b"]


def test_clean_caps_outliers():
    # IQR method: q1=2, q3=4, IQR=2, upper bound = 4 + 3*2 = 10 -> 100 capped.
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 100.0]})
    cleaned, report = clean_dataframe(df, outlier_method="iqr", outlier_threshold=3.0)
    assert report.outliers_capped == 1
    assert cleaned["a"].max() == 10.0


def test_clean_coerces_numeric_strings():
    df = pd.DataFrame({"a": ["1,000", "2,000", "3,000"]})
    cleaned, _ = clean_dataframe(df)
    assert pd.api.types.is_numeric_dtype(cleaned["a"])


# -- demo data -------------------------------------------------------------


def test_demo_classification():
    df = generate_demo_dataset(rows=500, task="classification")
    assert len(df) == 500
    assert "churn" in df.columns
    assert set(df["churn"].unique()) <= {0, 1}


def test_demo_regression():
    df = generate_demo_dataset(rows=200, task="regression")
    assert "price" in df.columns
    assert df["price"].isna().sum() == 0


def test_demo_invalid_task():
    with pytest.raises(DataError, match="classification"):
        generate_demo_dataset(task="bogus")
