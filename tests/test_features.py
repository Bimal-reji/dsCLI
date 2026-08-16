"""Tests for feature preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dscli.errors import ConfigError
from dscli.features.builder import FeatureSpec, build_preprocessor


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0],
            "num2": [10.0, 20.0, 30.0, 40.0],
            "cat1": ["a", "b", "a", "c"],
            "cat2": ["x", "x", "y", "z"],
            "target": [0, 1, 0, 1],
        }
    )


def test_feature_spec_groups_columns():
    df = _sample_df()
    spec = FeatureSpec.from_dataframe(df, exclude=["target"])
    assert spec.numeric_columns == ["num1", "num2"]
    assert spec.categorical_columns == ["cat1", "cat2"]


def test_feature_spec_drops_constant_and_all_missing():
    df = pd.DataFrame(
        {
            "a": [1.0, 1.0, 1.0],
            "b": [np.nan, np.nan, np.nan],
            "c": ["x", "y", "z"],
        }
    )
    spec = FeatureSpec.from_dataframe(df)
    assert "a" in spec.dropped_columns
    assert "b" in spec.dropped_columns
    assert spec.feature_columns == ["c"]


def test_onehot_preprocessor_transforms():
    df = _sample_df()
    spec = FeatureSpec.from_dataframe(df, exclude=["target"])
    pipeline = build_preprocessor(spec, categorical_encoding="onehot")
    X = pipeline.fit_transform(df[spec.feature_columns])
    # 2 numeric + 4 one-hot levels (a,b,c and x,y,z -> 3+3 but only 3 unique in cat1)
    assert X.shape[1] == 2 + 3 + 3


def test_ordinal_preprocessor_transforms():
    df = _sample_df()
    spec = FeatureSpec.from_dataframe(df, exclude=["target"])
    pipeline = build_preprocessor(spec, categorical_encoding="ordinal")
    X = pipeline.fit_transform(df[spec.feature_columns])
    assert X.shape[1] == 2 + 2


def test_preprocessor_reusable_on_new_data():
    df = _sample_df()
    spec = FeatureSpec.from_dataframe(df, exclude=["target"])
    pipeline = build_preprocessor(spec)
    pipeline.fit(df[spec.feature_columns])
    # New row with an unseen category in cat1 must not crash (handle_unknown=ignore).
    new = pd.DataFrame(
        [{"num1": 9.0, "num2": 90.0, "cat1": "zzz", "cat2": "x"}]
    )
    out = pipeline.transform(new[spec.feature_columns])
    assert out.shape[0] == 1


def test_invalid_scaler_raises():
    spec = FeatureSpec(numeric_columns=["a"], categorical_columns=[])
    with pytest.raises(ConfigError, match="scaler"):
        build_preprocessor(spec, scaler="bogus")


def test_invalid_encoding_raises():
    spec = FeatureSpec(numeric_columns=["a"], categorical_columns=["b"])
    with pytest.raises(ConfigError, match="categorical_encoding"):
        build_preprocessor(spec, categorical_encoding="bogus")
