"""Synthetic demo dataset generation.

Produces a realistic-looking customer dataset (tenure, charges, contract
type, ...) with a classification target (``churn``) or a regression target
(``price``), plus a few missing values so cleaning and validation steps have
something to do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dscli.errors import DataError


def generate_demo_dataset(
    rows: int = 2000, task: str = "classification", seed: int = 42
) -> pd.DataFrame:
    """Generate a synthetic dataset; returns a pandas DataFrame.

    Parameters
    ----------
    rows:
        Number of rows to generate.
    task:
        ``"classification"`` produces a churn target; ``"regression"``
        produces a continuous ``price`` target.
    seed:
        Random seed for reproducibility.
    """
    if rows < 50:
        raise DataError("Demo dataset needs at least 50 rows.")
    if task not in ("classification", "regression"):
        raise DataError("Demo task must be 'classification' or 'regression'.")

    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 73, rows)
    monthly_charges = rng.uniform(20, 120, rows).round(2)
    age = rng.integers(18, 82, rows)
    support_calls = rng.integers(0, 10, rows)
    income = rng.normal(60_000, 25_000, rows).round(0).clip(15_000, 250_000)
    gender = rng.choice(["Male", "Female"], rows)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], rows, p=[0.55, 0.25, 0.20]
    )
    payment = rng.choice(
        ["Credit card", "Bank transfer", "Electronic check", "Mailed check"], rows
    )

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(1, rows + 1)],
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "age": age,
            "support_calls": support_calls,
            "income": income,
            "gender": gender,
            "contract": contract,
            "payment_method": payment,
        }
    )

    if task == "classification":
        logits = (
            -0.08 * tenure
            + 0.02 * monthly_charges
            + 0.35 * support_calls
            + 0.5 * (contract == "Month-to-month")
            - 1.0 * (contract == "Two year")
            + 0.3 * (gender == "Male")
            - 1.5
            + rng.normal(0, 0.6, rows)
        )
        prob = 1.0 / (1.0 + np.exp(-logits))
        df["churn"] = (rng.random(rows) < prob).astype(int)
    else:
        price = (
            5_000
            + 120.0 * tenure
            + 15.0 * monthly_charges
            + 40.0 * support_calls
            + 0.02 * income
            + 300.0 * (contract == "One year")
            - 200.0 * (contract == "Two year")
            + rng.normal(0, 800, rows)
        ).round(0).clip(0, None)
        df["price"] = price

    # Sprinkle in some missing values so cleaning has work to do.
    missing_mask = rng.random(rows) < 0.03
    df.loc[missing_mask, "monthly_charges"] = np.nan
    missing_mask2 = rng.random(rows) < 0.02
    df.loc[missing_mask2, "support_calls"] = np.nan

    return df
