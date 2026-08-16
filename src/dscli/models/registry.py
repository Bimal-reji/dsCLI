"""Model registry.

Maps human-friendly algorithm names (``random_forest``, ``xgboost``, ...) to
scikit-learn compatible estimators. Optional libraries such as XGBoost and
LightGBM are used only when installed, so the CLI works with a plain
scikit-learn installation too.
"""

from __future__ import annotations

from typing import Any, Callable

from dscli.utils.logging_utils import get_logger
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge

from dscli.errors import ModelError

# Optional dependencies, imported lazily inside the factory functions.
try:  # pragma: no cover - depends on environment
    from xgboost import XGBClassifier, XGBRegressor

    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:  # pragma: no cover - depends on environment
    from lightgbm import LGBMClassifier, LGBMRegressor

    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

#: Default hyperparameters per algorithm. Users can override these via the
#: ``model.params`` config section.
DEFAULT_MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
    },
    "ridge": {
        "alpha": 1.0,
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 1,
        "n_jobs": -1,
    },
    "gradient_boosting": {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 3,
        "subsample": 1.0,
    },
    "xgboost": {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 6,
        "eval_metric": "logloss",
    },
    "lightgbm": {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "verbosity": -1,
    },
}

_ClassifierFactory = Callable[[dict[str, Any]], Any]
_RegressorFactory = Callable[[dict[str, Any]], Any]

_CLASSIFIERS: dict[str, _ClassifierFactory] = {
    "logistic_regression": lambda p: LogisticRegression(**p),
    "random_forest": lambda p: RandomForestClassifier(**p),
    "gradient_boosting": lambda p: GradientBoostingClassifier(**p),
    "xgboost": lambda p: XGBClassifier(**p),
    "lightgbm": lambda p: LGBMClassifier(**p),
}

_REGRESSORS: dict[str, _RegressorFactory] = {
    "ridge": lambda p: Ridge(**p),
    "random_forest": lambda p: RandomForestRegressor(**p),
    "gradient_boosting": lambda p: GradientBoostingRegressor(**p),
    "xgboost": lambda p: XGBRegressor(**p),
    "lightgbm": lambda p: LGBMRegressor(**p),
}

#: Algorithms usable for each task, in a sensible comparison order.
CLASSIFICATION_MODELS = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
]
REGRESSION_MODELS = ["ridge", "random_forest", "gradient_boosting", "xgboost", "lightgbm"]


def available_models(task: str | None = None) -> list[str]:
    """List algorithms, optionally restricted to a task and installed libs."""
    def _installed(names: list[str]) -> list[str]:
        result = []
        for name in names:
            if name == "xgboost" and not _XGB_AVAILABLE:
                continue
            if name == "lightgbm" and not _LGBM_AVAILABLE:
                continue
            result.append(name)
        return result

    if task == "classification":
        return _installed(CLASSIFICATION_MODELS)
    if task == "regression":
        return _installed(REGRESSION_MODELS)
    return _installed(list(dict.fromkeys(CLASSIFICATION_MODELS + REGRESSION_MODELS)))


def is_optional_available(name: str) -> bool:
    """Whether an optional model library is installed (xgboost/lightgbm)."""
    if name == "xgboost":
        return _XGB_AVAILABLE
    if name == "lightgbm":
        return _LGBM_AVAILABLE
    return True


def get_model_defaults(algorithm: str) -> dict[str, Any]:
    """Return the default hyperparameters for an algorithm."""
    if algorithm not in DEFAULT_MODEL_PARAMS:
        raise ModelError(
            f"Unknown model '{algorithm}'. Available models: "
            f"{', '.join(available_models())}."
        )
    return dict(DEFAULT_MODEL_PARAMS[algorithm])


def create_model(algorithm: str, task: str, params: dict[str, Any] | None = None) -> Any:
    """Instantiate an estimator for ``algorithm`` and ``task``.

    ``params`` override the algorithm defaults. Hyperparameters that do not
    apply to the chosen estimator (e.g. a shared ``n_estimators`` config value
    used while comparing several models) are ignored with a logged warning.
    Raises :class:`ModelError` for unknown algorithms or when an optional
    library is missing.
    """
    if algorithm not in DEFAULT_MODEL_PARAMS:
        raise ModelError(
            f"Unknown model '{algorithm}'. Available models: {', '.join(available_models())}."
        )
    if algorithm == "xgboost" and not _XGB_AVAILABLE:
        raise ModelError(
            "XGBoost is not installed. Install it with 'pip install dscli[xgboost]' "
            "or choose another model."
        )
    if algorithm == "lightgbm" and not _LGBM_AVAILABLE:
        raise ModelError(
            "LightGBM is not installed. Install it with 'pip install dscli[lightgbm]' "
            "or choose another model."
        )

    if task == "classification":
        factory = _CLASSIFIERS.get(algorithm)
    elif task == "regression":
        factory = _REGRESSORS.get(algorithm)
    else:
        raise ModelError(f"Unknown task '{task}'. Use 'classification' or 'regression'.")

    if factory is None:
        raise ModelError(
            f"Model '{algorithm}' does not support {task}. "
            f"Available models for {task}: {', '.join(available_models(task))}."
        )

    merged = get_model_defaults(algorithm)
    if params:
        merged.update(params)

    # Drop parameters the estimator does not accept, logging a warning so
    # users notice typos or mismatched shared config values.
    base = factory(get_model_defaults(algorithm))
    valid = {k: v for k, v in merged.items() if k in base.get_params()}
    ignored = [k for k in merged if k not in valid]
    if ignored:
        get_logger().warning(
            "Ignoring hyperparameters not accepted by %s: %s",
            algorithm,
            ", ".join(sorted(ignored)),
        )

    try:
        return factory(valid)
    except TypeError as exc:
        raise ModelError(
            f"Invalid hyperparameters for model '{algorithm}': {exc}"
        ) from exc
