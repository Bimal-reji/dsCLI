"""File and artifact I/O helpers.

Centralizes the "do not silently overwrite important files" rule and keeps
JSON/YAML serialization in one place.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import yaml

from dscli.errors import DataError, ModelError

JSON_EXTENSIONS = {".json", ".geojson"}
YAML_EXTENSIONS = {".yaml", ".yml"}
TEXT_EXTENSIONS = {".txt", ".md", ".log", ".csv", ".tsv"}


def ensure_readable_file(path: str | Path, description: str = "file") -> Path:
    """Check that ``path`` exists and is a regular file; return it as a Path."""
    p = Path(path)
    if not p.exists():
        raise DataError(f"{description.capitalize()} not found: '{p}'.")
    if not p.is_file():
        raise DataError(f"Expected {description} to be a file, got a directory: '{p}'.")
    return p


def check_can_write(path: str | Path, overwrite: bool, description: str = "file") -> Path:
    """Ensure ``path`` may be written without clobbering existing data.

    Raises :class:`DataError` when the file already exists and ``overwrite``
    is False.
    """
    p = Path(path)
    if p.exists() and not overwrite:
        raise DataError(
            f"{description.capitalize()} already exists: '{p}'. "
            "Pass --overwrite to replace it."
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path) -> Any:
    """Load a JSON file, raising DataError on parse problems."""
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"Could not read JSON file '{p}': {exc}") from exc


def save_json(data: Any, path: str | Path, overwrite: bool = False) -> Path:
    """Write ``data`` as pretty-printed JSON, honoring overwrite protection."""
    p = check_can_write(path, overwrite, description="file")
    p.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    return p


def load_yaml(path: str | Path) -> Any:
    """Load a YAML file, raising DataError on parse problems."""
    p = Path(path)
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DataError(f"Could not read YAML file '{p}': {exc}") from exc


def save_yaml(data: Any, path: str | Path, overwrite: bool = False) -> Path:
    """Write ``data`` as YAML, honoring overwrite protection."""
    p = check_can_write(path, overwrite, description="file")
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return p


def save_model_artifact(model: Any, path: str | Path, overwrite: bool = False) -> Path:
    """Persist a fitted model with joblib, honoring overwrite protection."""
    p = check_can_write(path, overwrite, description="model")
    try:
        joblib.dump(model, p)
    except Exception as exc:  # joblib can raise several low-level errors
        raise ModelError(f"Failed to save model to '{p}': {exc}") from exc
    return p


def load_model_artifact(path: str | Path) -> Any:
    """Load a joblib model artifact, raising ModelError when it fails."""
    p = ensure_readable_file(path, description="model")
    try:
        return joblib.load(p)
    except Exception as exc:  # joblib raises a broad set of errors
        raise ModelError(
            f"Failed to load model from '{p}'. Is it a valid joblib artifact? ({exc})"
        ) from exc
