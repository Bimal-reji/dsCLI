"""Shared fixtures for the dscli test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dscli.config import Config


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """A fresh directory that acts as a project root."""
    root = tmp_path / "project"
    root.mkdir()
    return root


@pytest.fixture
def make_config(project_dir: Path):
    """Factory that builds a Config for the temp project with overrides."""

    def _make(**overrides) -> Config:
        raw = {
            "data": {
                "target": "churn",
                "id_column": "customer_id",
                "train": "data/processed/train.csv",
                "validation": "data/processed/validation.csv",
                "test": "data/processed/test.csv",
            },
            "training": {"cv_folds": 2, "validation_size": 0.2, "test_size": 0.2},
            "model": {"algorithm": "random_forest", "params": {"n_estimators": 50}},
        }
        raw.update(overrides)
        return Config.from_dict(raw, project_root=project_dir)

    return _make


@pytest.fixture
def classification_df() -> pd.DataFrame:
    """A small synthetic classification dataset with a correlated target."""
    rng = np.random.default_rng(7)
    n = 400
    age = rng.integers(18, 80, n).astype(float)
    balance = rng.normal(50_000, 20_000, n).round(2)
    calls = rng.integers(0, 15, n).astype(float)
    region = rng.choice(["north", "south", "east"], n)
    plan = rng.choice(["basic", "premium"], n)

    logits = (
        0.05 * age
        - 0.00002 * balance
        + 0.25 * calls
        + 0.8 * (plan == "premium")
        - 2.0
        + rng.normal(0, 0.5, n)
    )
    prob = 1.0 / (1.0 + np.exp(-logits))
    df = pd.DataFrame(
        {
            "customer_id": [f"C-{i}" for i in range(n)],
            "age": age,
            "balance": balance,
            "calls": calls,
            "region": region,
            "plan": plan,
            "churn": (rng.random(n) < prob).astype(int),
        }
    )
    return df


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """A small synthetic regression dataset."""
    rng = np.random.default_rng(11)
    n = 300
    size = rng.integers(40, 300, n).astype(float)
    bedrooms = rng.integers(1, 6, n).astype(float)
    age = rng.integers(0, 80, n).astype(float)
    district = rng.choice(["center", "suburb", "rural"], n)
    price = (
        50_000
        + 2_500 * size
        + 15_000 * bedrooms
        - 400 * age
        + 20_000 * (district == "center")
        + rng.normal(0, 15_000, n)
    ).clip(0, None)
    return pd.DataFrame(
        {
            "size": size,
            "bedrooms": bedrooms,
            "age": age,
            "district": district,
            "price": price,
        }
    )


@pytest.fixture
def trained_result(classification_df, make_config):
    """A trained TrainingResult on the synthetic classification data."""
    from dscli.models.trainer import train_model

    return train_model(classification_df, make_config())
