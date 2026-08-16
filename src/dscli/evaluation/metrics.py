"""Metrics computation.

Automatically computes the appropriate metrics for the task type:

* **Classification**: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
* **Regression**: MAE, MSE, RMSE, R².

Also provides scoring-string selection for cross-validation and feature
importance extraction from fitted pipelines.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from dscli.errors import EvaluationError

#: Human-friendly labels for metrics, used by Rich tables.
METRIC_LABELS: dict[str, str] = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 Score",
    "roc_auc": "ROC-AUC",
    "mae": "MAE",
    "mse": "MSE",
    "rmse": "RMSE",
    "r2": "R²",
}


def get_scoring(task: str) -> str:
    """Return the default cross-validation scoring for a task."""
    if task == "classification":
        return "roc_auc_ovr"
    if task == "regression":
        return "r2"
    raise EvaluationError(f"Unknown task '{task}'. Use 'classification' or 'regression'.")


def _classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
) -> dict[str, Any]:
    """Compute classification metrics. y_true must already be label-encoded."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = len(np.unique(y_true))

    average = "binary" if n_classes == 2 else "macro"
    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, average=average, zero_division=0)), 4),
    }

    if y_proba is not None and len(np.unique(y_true)) >= 2:
        try:
            if n_classes == 2:
                # roc_auc_score expects scores for the positive class.
                scores = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                metrics["roc_auc"] = round(float(roc_auc_score(y_true, scores)), 4)
            else:
                metrics["roc_auc"] = round(
                    float(roc_auc_score(y_true, y_proba, multi_class="ovr")), 4
                )
        except ValueError as exc:
            # e.g. only one class present in the evaluation set
            raise EvaluationError(f"Could not compute ROC-AUC: {exc}") from exc

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "mse": round(mse, 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    task: str,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute the appropriate metrics for ``task``.

    For classification, ``y_true`` must be numeric label-encoded values and
    ``y_pred`` the corresponding integer predictions. ``y_proba`` (optional)
    enables ROC-AUC.
    """
    if task == "classification":
        return _classification_metrics(y_true, y_pred, y_proba)
    if task == "regression":
        return _regression_metrics(y_true, y_pred)
    raise EvaluationError(f"Unknown task '{task}'. Use 'classification' or 'regression'.")


def extract_feature_importance(
    pipeline: Any, feature_names: list[str]
) -> list[tuple[str, float]]:
    """Extract per-feature importance from a fitted pipeline.

    Works for tree-based models (``feature_importances_``) and linear models
    (absolute coefficients). Returns a list of ``(feature, importance)``
    sorted descending. Falls back to an empty list for models without a
    supported attribute.
    """
    model = getattr(pipeline, "named_steps", {}).get("model", pipeline)
    try:
        if hasattr(model, "feature_importances_"):
            scores = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)
            if coef.ndim == 2:
                scores = np.abs(coef).mean(axis=0)
            else:
                scores = np.abs(coef)
        else:
            return []
    except (AttributeError, ValueError, TypeError):
        return []

    if len(scores) != len(feature_names):
        return []
    ranked = sorted(zip(feature_names, scores), key=lambda t: t[1], reverse=True)
    return [(name, round(float(score), 4)) for name, score in ranked]
