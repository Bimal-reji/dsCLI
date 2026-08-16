"""Model training pipeline.

``train_model`` is the single entry point for training: it takes a clean
DataFrame plus a :class:`~dscli.config.Config` and returns a fully fitted
pipeline (preprocessing + estimator) together with cross-validation scores
and hold-out metrics.

The hold-out set can be supplied explicitly (``validation_df``, e.g. the
validation split produced by ``dscli split``) or carved out of the training
data internally.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from dscli.config import Config
from dscli.data.validator import encode_labels, validate_target
from dscli.evaluation.metrics import (
    compute_metrics,
    extract_feature_importance,
    get_scoring,
)
from dscli.features.builder import FeatureSpec, build_preprocessor
from dscli.models.registry import create_model


@dataclass
class TrainingResult:
    """Everything produced by a single training run."""

    model_name: str
    task: str
    target: str
    pipeline: Pipeline
    feature_spec: FeatureSpec
    metrics: dict[str, Any] = field(default_factory=dict)
    train_metrics: dict[str, Any] = field(default_factory=dict)
    cv_scores: dict[str, list[float]] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    feature_importance: list[tuple[str, float]] = field(default_factory=list)
    classes: list[Any] | None = None
    label_encoder: dict[Any, int] | None = None
    n_train: int = 0
    n_validation: int = 0
    duration_seconds: float = 0.0

    @property
    def metrics_flat(self) -> dict[str, float]:
        """Metrics without structured values (e.g. confusion matrix)."""
        return {k: v for k, v in self.metrics.items() if isinstance(v, (int, float))}


def _prepare_data(
    df: pd.DataFrame,
    validation_df: pd.DataFrame | None,
    config: Config,
    task: str,
    target: str,
) -> tuple[
    pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.Series, pd.Series
]:
    """Return (X_train, X_val, y_train, y_val, y_train_raw, y_val_raw).

    When ``validation_df`` is given it is used as the hold-out set directly;
    otherwise a validation split is carved from ``df``.
    """
    if validation_df is not None:
        validate_target(validation_df, target)
        X_train = df.drop(columns=[target])
        X_val = validation_df.drop(columns=[target])
        y_train_raw = df[target]
        y_val_raw = validation_df[target]
    else:
        X = df.drop(columns=[target])
        y_raw = df[target]
        stratify = y_raw if config.training.stratify and task == "classification" else None
        if stratify is not None:
            if stratify.nunique() < 2 or stratify.value_counts().min() < 2:
                stratify = None
        X_train, X_val, y_train_raw, y_val_raw = train_test_split(
            X,
            y_raw,
            test_size=config.training.validation_size,
            random_state=config.training.random_state,
            stratify=stratify,
        )

    if task == "classification":
        y_train_encoded, classes = encode_labels(y_train_raw)
        label_map = {label: i for i, label in enumerate(classes)}
        # Unknown labels in the validation set are mapped to -1 and simply
        # never match a prediction (metrics stay well-defined).
        y_val_encoded = np.array([label_map.get(v, -1) for v in y_val_raw])
        return X_train, X_val, y_train_encoded, y_val_encoded, y_train_raw, y_val_raw

    return (
        X_train,
        X_val,
        y_train_raw.to_numpy(dtype=float),
        y_val_raw.to_numpy(dtype=float),
        y_train_raw,
        y_val_raw,
    )


def train_model(
    df: pd.DataFrame,
    config: Config,
    *,
    algorithm: str | None = None,
    model_params: dict[str, Any] | None = None,
    target: str | None = None,
    task: str | None = None,
    validation_df: pd.DataFrame | None = None,
    verbose: bool = False,
) -> TrainingResult:
    """Train a model end-to-end and return a :class:`TrainingResult`.

    The pipeline is built as preprocessor + estimator, cross-validated on the
    training split, then fitted on the full training split and evaluated on
    the hold-out validation split.
    """
    target_name = target or config.data.target
    target_name, inferred_task = validate_target(df, target_name)
    effective_task = task or inferred_task

    if algorithm is None:
        algorithm = config.model.algorithm
    merged_params = dict(config.model.params)
    if model_params:
        merged_params.update(model_params)

    start = time.monotonic()

    X_train, X_val, y_train, y_val, y_train_raw, y_val_raw = _prepare_data(
        df, validation_df, config, effective_task, target_name
    )

    spec = FeatureSpec.from_dataframe(
        X_train,
        exclude=[config.data.id_column] if config.data.id_column else None,
        max_categories=config.features.max_categories,
    )

    preprocessor = build_preprocessor(
        spec,
        scale_numerical=config.features.scale_numerical,
        scaler=config.features.scaler,
        categorical_encoding=config.features.categorical_encoding,
        handle_unknown=config.features.handle_unknown,
    )
    model = create_model(algorithm, effective_task, merged_params)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    # -- cross-validation on the training split ----------------------------
    scoring = config.training.scoring or get_scoring(effective_task)
    cv = _make_cv(config, effective_task, y_train)
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=1,
    )
    cv_scores = {
        "cv_mean": round(float(cv_results["test_score"].mean()), 4),
        "cv_std": round(float(cv_results["test_score"].std()), 4),
        "cv_scores": [round(float(s), 4) for s in cv_results["test_score"]],
    }

    # -- final fit + hold-out evaluation ------------------------------------
    pipeline.fit(X_train, y_train)

    if effective_task == "classification":
        y_pred = pipeline.predict(X_val)
        try:
            y_proba = pipeline.predict_proba(X_val)
        except AttributeError:
            y_proba = None
        classes = list(pd.unique(y_train_raw))
        metrics = compute_metrics(y_val, y_pred, effective_task, y_proba)
        train_pred = pipeline.predict(X_train)
        train_metrics = compute_metrics(y_train, train_pred, effective_task, None)
        label_map = {label: i for i, label in enumerate(classes)}
        metrics["class_distribution"] = {
            str(label): int((y_val_raw == label).sum()) for label in classes
        }
    else:
        y_pred = pipeline.predict(X_val)
        metrics = compute_metrics(y_val, y_pred, effective_task)
        train_pred = pipeline.predict(X_train)
        train_metrics = compute_metrics(y_train, train_pred, effective_task)
        classes = None
        label_map = None

    feature_names = _transformed_feature_names(pipeline, X_train)
    importance = extract_feature_importance(pipeline, feature_names)

    duration = time.monotonic() - start

    return TrainingResult(
        model_name=algorithm,
        task=effective_task,
        target=target_name,
        pipeline=pipeline,
        feature_spec=spec,
        metrics=metrics,
        train_metrics=train_metrics,
        cv_scores=cv_scores,
        feature_names=feature_names,
        feature_importance=importance,
        classes=classes,
        label_encoder=label_map,
        n_train=len(X_train),
        n_validation=len(X_val),
        duration_seconds=round(duration, 2),
    )


def _make_cv(config: Config, task: str, y_train: np.ndarray) -> Any:
    """Build the cross-validation splitter honoring stratify + folds."""
    n_splits = max(2, config.training.cv_folds)
    if task == "classification":
        unique = len(np.unique(y_train))
        counts = np.bincount(y_train.astype(int))
        if unique >= 2 and counts.min() >= n_splits:
            return StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=config.training.random_state
            )
    return KFold(n_splits=n_splits, shuffle=True, random_state=config.training.random_state)


def _transformed_feature_names(pipeline: Pipeline, X_train: pd.DataFrame) -> list[str]:
    """Get the post-transformation feature names from the fitted pipeline."""
    try:
        return list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    except (AttributeError, ValueError):
        return list(X_train.columns)


def predict_with_pipeline(
    pipeline: Pipeline, X: pd.DataFrame, task: str
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run predictions; returns (predictions, probabilities-or-None).

    Probabilities are only returned for classification pipelines that expose
    ``predict_proba``.
    """
    y_pred = pipeline.predict(X)
    proba: np.ndarray | None = None
    if task == "classification":
        try:
            proba = pipeline.predict_proba(X)
        except AttributeError:
            proba = None
    return y_pred, proba
