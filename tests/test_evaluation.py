"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from dscli.evaluation.metrics import (
    compute_metrics,
    extract_feature_importance,
    get_scoring,
)
from dscli.errors import EvaluationError


def test_classification_metrics_perfect():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    metrics = compute_metrics(y_true, y_pred, "classification")
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_classification_metrics_with_proba():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.1, 0.9]])
    metrics = compute_metrics(y_true, y_pred, "classification", y_proba=proba)
    assert "roc_auc" in metrics
    assert 0.5 <= metrics["roc_auc"] <= 1.0


def test_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    metrics = compute_metrics(y_true, y_pred, "regression")
    assert metrics["mae"] == 0.0
    assert metrics["mse"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


def test_regression_metrics_imperfect():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 8.0])
    metrics = compute_metrics(y_true, y_pred, "regression")
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["r2"] < 0.5  # big error on the last point


def test_unknown_task_raises():
    with pytest.raises(EvaluationError):
        compute_metrics(np.array([1]), np.array([1]), "bogus")


def test_get_scoring():
    assert get_scoring("classification") == "roc_auc_ovr"
    assert get_scoring("regression") == "r2"


def test_feature_importance_tree():
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=0)
    model.fit(X, y)
    importance = extract_feature_importance(model, ["a", "b", "c"])
    assert len(importance) == 3
    names = [name for name, _ in importance]
    assert names[0] == "a"  # most important feature
    assert all(isinstance(score, float) for _, score in importance)


def test_feature_importance_linear():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 2))
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression()
    model.fit(X, y)
    importance = extract_feature_importance(model, ["a", "b"])
    assert len(importance) == 2


def test_feature_importance_mismatched_lengths_returns_empty():
    class FakeModel:
        def __init__(self) -> None:
            self.feature_importances_ = np.array([0.5, 0.5])

    assert extract_feature_importance(FakeModel(), ["only_one"]) == []
