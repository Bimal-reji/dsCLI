"""Tests for the models package: registry, trainer, persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dscli.errors import ModelError
from dscli.models import available_models, create_model, load_model, save_model, train_model
from dscli.models.registry import get_model_defaults


def test_available_models_always_has_core():
    models = available_models()
    assert "random_forest" in models
    assert "logistic_regression" in models
    assert "gradient_boosting" in models


def test_available_models_by_task():
    assert "ridge" in available_models("regression")
    assert "logistic_regression" in available_models("classification")
    assert "ridge" not in available_models("classification")


def test_create_model_classification_and_regression():
    from sklearn.base import is_classifier, is_regressor

    clf = create_model("random_forest", "classification")
    assert is_classifier(clf)
    assert hasattr(clf, "predict_proba")
    reg = create_model("ridge", "regression")
    assert is_regressor(reg)
    assert reg.get_params()["alpha"] == 1.0


def test_create_model_unknown_raises():
    with pytest.raises(ModelError, match="Unknown model"):
        create_model("knn_magic", "classification")


def test_create_model_task_mismatch_raises():
    with pytest.raises(ModelError, match="does not support"):
        create_model("ridge", "classification")


def test_create_model_params_override_defaults():
    model = create_model("random_forest", "classification", {"n_estimators": 7})
    assert model.n_estimators == 7
    default = get_model_defaults("random_forest")["n_estimators"]
    assert default != 7


def test_train_classification(classification_df, make_config):
    result = train_model(classification_df, make_config())
    assert result.task == "classification"
    assert result.target == "churn"
    assert "accuracy" in result.metrics
    assert result.metrics["accuracy"] > 0.5
    assert result.n_train > 0 and result.n_validation > 0
    assert result.cv_scores["cv_mean"] > 0.5
    assert len(result.feature_names) > 0
    assert result.pipeline is not None
    # categorical columns should be encoded -> more transformed features than raw
    assert len(result.feature_names) > 5


def test_train_regression(regression_df, make_config):
    config = make_config(data={"target": "price", "id_column": None})
    result = train_model(regression_df, config, algorithm="ridge")
    assert result.task == "regression"
    assert "r2" in result.metrics
    assert result.metrics["r2"] > 0.5


def test_train_with_external_validation(classification_df, make_config):
    train = classification_df.iloc[:300].copy()
    val = classification_df.iloc[300:].copy()
    result = train_model(train, make_config(), validation_df=val)
    assert result.n_train == 300
    assert result.n_validation == 100
    assert "accuracy" in result.metrics


def test_train_missing_target_raises(make_config):
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(Exception):
        train_model(df, make_config())


def test_train_explicit_target_override(make_config):
    df = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0] * 5, "b": [0, 1, 0, 1] * 5}
    )
    result = train_model(df, make_config(), target="b")
    assert result.target == "b"
    assert result.task == "classification"


def test_save_and_load_roundtrip(trained_result, tmp_path):
    path = Path(tmp_path) / "model.joblib"
    save_model(trained_result, path)
    assert path.exists()
    assert path.with_suffix(".joblib.json").exists()

    pipeline, metadata = load_model(path)
    assert metadata.model_name == trained_result.model_name
    assert metadata.task == "classification"
    assert metadata.target == "churn"
    assert metadata.feature_names == trained_result.feature_names
    # pipeline still predicts
    X = trained_result.pipeline.named_steps  # noqa: F841
    pred = pipeline.predict(
        pd.DataFrame(
            [{"age": 40.0, "balance": 50_000.0, "calls": 3.0, "region": "north", "plan": "basic"}]
        )
    )
    assert pred.shape == (1,)


def test_load_model_missing_metadata_raises(trained_result, tmp_path):
    from dscli.utils.io import save_model_artifact

    path = Path(tmp_path) / "bare.joblib"
    save_model_artifact(trained_result.pipeline, path)
    with pytest.raises(ModelError, match="Metadata sidecar"):
        load_model(path)


def test_predict_with_pipeline_proba(trained_result):
    from dscli.models.trainer import predict_with_pipeline

    X = pd.DataFrame(
        [{"age": 40.0, "balance": 50_000.0, "calls": 3.0, "region": "north", "plan": "basic"}]
    )
    y_pred, proba = predict_with_pipeline(trained_result.pipeline, X, "classification")
    assert y_pred.shape == (1,)
    assert proba is not None
    assert proba.shape == (1, 2)
    assert np.isclose(proba.sum(axis=1), 1.0).all()
