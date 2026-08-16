"""Figure generation.

All figures are saved to ``reports/figures`` as PNG files and work in
headless environments (``matplotlib`` is forced to the Agg backend). Every
plot function returns the path it wrote, so callers can report results
without knowing the internal layout.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import roc_curve, auc  # noqa: E402

from dscli.errors import ReportError

_DPI = 120


def _save(fig: plt.Figure, figure_dir, name: str) -> str:
    """Save a figure, returning the relative path for display."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / name
    try:
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    except Exception as exc:
        raise ReportError(f"Failed to save figure '{path}': {exc}") from exc
    finally:
        plt.close(fig)
    return str(path)


def correlation_heatmap(df: pd.DataFrame, figure_dir) -> str:
    """Correlation heatmap of numeric columns."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        raise ReportError(
            "Need at least two numeric columns to draw a correlation heatmap."
        )
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    return _save(fig, figure_dir, "correlation_heatmap.png")


def target_distribution(df: pd.DataFrame, target: str, figure_dir) -> str:
    """Bar or histogram of the target variable."""
    series = df[target]
    fig, ax = plt.subplots(figsize=(8, 5))
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 20:
        series.dropna().hist(bins=30, ax=ax, color="#4C72B0")
        ax.set_title(f"Target Distribution: {target}")
        ax.set_ylabel("Frequency")
    else:
        counts = series.value_counts()
        sns.barplot(
            x=counts.index.astype(str),
            y=counts.values,
            hue=counts.index.astype(str),
            legend=False,
            ax=ax,
            palette="viridis",
        )
        ax.set_title(f"Target Distribution: {target}")
        ax.set_xlabel(target)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return _save(fig, figure_dir, "target_distribution.png")


def feature_histograms(df: pd.DataFrame, figure_dir, max_features: int = 16) -> str:
    """Histograms for the most variable numeric features."""
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        raise ReportError("No numeric features to plot.")
    top = numeric.var().sort_values(ascending=False).head(max_features).index.tolist()
    n = len(top)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, top):
        numeric[col].dropna().hist(bins=30, ax=ax, color="#55A868")
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Numeric Feature Distributions", fontsize=13)
    fig.tight_layout()
    return _save(fig, figure_dir, "feature_distributions.png")


def confusion_matrix_plot(matrix: list[list[int]], figure_dir) -> str:
    """Confusion matrix heatmap."""
    arr = np.asarray(matrix, dtype=int)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(arr, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return _save(fig, figure_dir, "confusion_matrix.png")


def roc_curve_plot(y_true: np.ndarray, y_proba: np.ndarray, figure_dir) -> str:
    """ROC curve for binary classification."""
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#C44E52", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, figure_dir, "roc_curve.png")


def feature_importance_plot(
    importance: list[tuple[str, float]], figure_dir, top_n: int = 20
) -> str:
    """Horizontal bar chart of feature importances."""
    if not importance:
        raise ReportError("No feature importance available for this model.")
    ranked = sorted(importance, key=lambda t: t[1], reverse=True)[:top_n]
    names = [name for name, _ in ranked][::-1]
    scores = [score for _, score in ranked][::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(names))))
    ax.barh(names, scores, color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    return _save(fig, figure_dir, "feature_importance.png")


def residuals_plot(y_true: np.ndarray, y_pred: np.ndarray, figure_dir) -> str:
    """Residuals vs. predicted for regression models."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.6, color="#8172B3")
    ax.axhline(0, color="gray", linestyle="--", lw=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs. Predicted")
    fig.tight_layout()
    return _save(fig, figure_dir, "residuals.png")


def generate_figures(
    df: pd.DataFrame,
    figure_dir,
    *,
    target: str | None = None,
    metrics: dict | None = None,
    y_true: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    y_proba: np.ndarray | None = None,
    feature_importance: list[tuple[str, float]] | None = None,
    task: str | None = None,
) -> list[str]:
    """Generate every figure relevant to the current context.

    Returns the list of saved figure paths. Missing pieces (e.g. no ROC data)
    are skipped silently.
    """
    saved: list[str] = []
    figure_dir = _as_path(figure_dir)

    def _safe(generator, *args) -> None:
        try:
            saved.append(generator(*args, figure_dir))
        except ReportError:
            pass

    _safe(correlation_heatmap, df)
    if target and target in df.columns:
        _safe(target_distribution, df, target)
    _safe(feature_histograms, df)

    if metrics and "confusion_matrix" in metrics:
        _safe(confusion_matrix_plot, metrics["confusion_matrix"])

    if y_true is not None and y_proba is not None and task == "classification":
        try:
            if len(np.unique(y_true)) == 2:
                _safe(roc_curve_plot, y_true, y_proba)
        except ValueError:
            pass

    if feature_importance:
        _safe(feature_importance_plot, feature_importance)

    if y_true is not None and y_pred is not None and task == "regression":
        _safe(residuals_plot, y_true, y_pred)

    return saved


def _as_path(path) -> Path:
    from pathlib import Path

    return Path(path)
