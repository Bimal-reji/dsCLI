"""Feature preprocessing.

Builds a single reusable scikit-learn ``Pipeline`` (a
``ColumnTransformer``) that:

* imputes missing values in numeric and categorical columns,
* scales numeric columns (optional),
* encodes categorical columns (one-hot or ordinal).

The resulting pipeline is a first-class artifact: it is saved with the model
and applied to new data at prediction time, so training and serving use
exactly the same transformations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from dscli.errors import ConfigError

_SCALERS: dict[str, Any] = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


@dataclass
class FeatureSpec:
    """Column grouping discovered from a DataFrame."""

    numeric_columns: list[str]
    categorical_columns: list[str]
    dropped_columns: list[str] = None  # type: ignore[assignment]

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        exclude: list[str] | None = None,
        max_categories: int = 50,
    ) -> "FeatureSpec":
        """Discover numeric/categorical splits, dropping unusable columns.

        Columns with a single unique value, an object dtype with too many
        distinct values, or entirely missing values are dropped and reported.
        """
        exclude = exclude or []
        dropped: list[str] = []

        candidates = [c for c in df.columns if c not in exclude]
        numeric: list[str] = []
        categorical: list[str] = []

        for col in candidates:
            series = df[col]
            if series.isna().all():
                dropped.append(col)
                continue
            if series.nunique(dropna=True) <= 1:
                dropped.append(col)
                continue
            if pd.api.types.is_numeric_dtype(series):
                numeric.append(col)
            else:
                if series.nunique(dropna=True) > max_categories:
                    dropped.append(col)
                else:
                    categorical.append(col)

        return cls(
            numeric_columns=numeric,
            categorical_columns=categorical,
            dropped_columns=dropped,
        )

    @property
    def feature_columns(self) -> list[str]:
        """All columns that survive preprocessing (input to the transformer)."""
        return self.numeric_columns + self.categorical_columns


def build_preprocessor(
    spec: FeatureSpec,
    *,
    scale_numerical: bool = True,
    scaler: str = "standard",
    categorical_encoding: str = "onehot",
    handle_unknown: str = "ignore",
) -> Pipeline:
    """Build the preprocessing pipeline for a :class:`FeatureSpec`."""
    if scaler not in _SCALERS:
        raise ConfigError(
            f"Unknown scaler '{scaler}'. Allowed values: {', '.join(sorted(_SCALERS))}."
        )
    if categorical_encoding not in ("onehot", "ordinal"):
        raise ConfigError(
            f"Unknown categorical_encoding '{categorical_encoding}'. "
            "Allowed values: onehot, ordinal."
        )

    transformers: list[tuple[str, Any, list[str]]] = []

    if spec.numeric_columns:
        numeric_steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median"))
        ]
        if scale_numerical:
            numeric_steps.append(("scaler", _SCALERS[scaler]()))
        transformers.append(("numeric", Pipeline(numeric_steps), spec.numeric_columns))

    if spec.categorical_columns:
        if categorical_encoding == "onehot":
            encoder = OneHotEncoder(
                handle_unknown=handle_unknown,
                sparse_output=False,
                min_frequency=None,
            )
        else:
            # OrdinalEncoder only accepts 'use_encoded_value'/'error' for
            # handle_unknown; map the user-facing 'ignore' to -1.
            if handle_unknown == "ignore":
                encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
            else:
                encoder = OrdinalEncoder(handle_unknown="error")
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", encoder),
                    ]
                ),
                spec.categorical_columns,
            )
        )

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocessor", column_transformer)])
