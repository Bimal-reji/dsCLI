"""Model persistence.

Models are saved as joblib artifacts together with a JSON sidecar holding
metadata (task, target, feature names, classes) needed to run predictions and
evaluations later without re-training.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dscli.errors import ModelError
from dscli.models.trainer import TrainingResult
from dscli.utils.io import load_model_artifact, save_model_artifact


def _native(value: Any) -> Any:
    """Convert numpy scalars to native Python types for JSON serialization."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_native(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_native(v) for v in value]
    return value


@dataclass
class ModelMetadata:
    """Metadata describing a saved model artifact."""

    model_name: str
    task: str
    target: str
    feature_names: list[str] = field(default_factory=list)
    classes: list[Any] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    cv_scores: dict[str, Any] = field(default_factory=dict)
    feature_importance: list[list] = field(default_factory=list)
    algorithm: str = ""
    version: int = 1

    @classmethod
    def from_training_result(cls, result: TrainingResult) -> "ModelMetadata":
        return cls(
            model_name=result.model_name,
            task=result.task,
            target=result.target,
            feature_names=list(result.feature_names),
            classes=[_native(c) for c in (result.classes or [])],
            metrics={k: _native(v) for k, v in result.metrics_flat.items()},
            cv_scores={k: _native(v) for k, v in result.cv_scores.items()},
            feature_importance=[
                [name, _native(score)] for name, score in result.feature_importance
            ],
            algorithm=result.model_name,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelMetadata":
        p = Path(path)
        if not p.is_file():
            raise ModelError(f"Model metadata not found: '{p}'.")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelError(f"Invalid model metadata file '{p}': {exc}") from exc
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def save_model(result: TrainingResult, path: str | Path, overwrite: bool = False) -> Path:
    """Save a fitted pipeline plus metadata sidecar; returns the artifact path."""
    model_path = Path(path)
    save_model_artifact(result.pipeline, model_path, overwrite=overwrite)
    metadata_path = model_path.with_suffix(model_path.suffix + ".json")
    metadata = asdict(ModelMetadata.from_training_result(result))
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return model_path


def load_model(path: str | Path) -> tuple[Any, ModelMetadata]:
    """Load a fitted pipeline and its metadata.

    The metadata sidecar is required; raise :class:`ModelError` when missing
    so prediction/evaluation commands can report a clear message.
    """
    model_path = Path(path)
    pipeline = load_model_artifact(model_path)
    metadata_path = model_path.with_suffix(model_path.suffix + ".json")
    if not metadata_path.is_file():
        raise ModelError(
            f"Metadata sidecar not found: '{metadata_path}'. "
            "This model was not saved by dscli and cannot be used for prediction."
        )
    metadata = ModelMetadata.from_file(metadata_path)
    return pipeline, metadata
