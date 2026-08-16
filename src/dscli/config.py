"""Configuration handling for dscli.

Configuration is centralized in a single YAML file (``configs/config.yaml``
by default). CLI arguments can override any value at runtime. Relative paths
in the configuration are always resolved against the *project root* so that
commands work from any subdirectory of a project.
"""

from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from dscli.errors import ConfigError

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "name": "my-project",
    },
    "data": {
        "raw_dir": "data/raw",
        "interim_dir": "data/interim",
        "processed_dir": "data/processed",
        "external_dir": "data/external",
        "train": "data/processed/train.csv",
        "validation": "data/processed/validation.csv",
        "test": "data/processed/test.csv",
        "target": None,
        "id_column": None,
        "drop_columns": [],
    },
    "cleaning": {
        "drop_duplicates": True,
        "missing_strategy": "mean",  # mean | median | most_frequent | constant | drop
        "missing_constant": 0,
        "drop_high_missing_threshold": 0.8,  # drop columns with >=80% missing
        "outlier_method": None,  # iqr | zscore | None
        "outlier_threshold": 3.0,
    },
    "features": {
        "scale_numerical": True,
        "scaler": "standard",  # standard | minmax | robust
        "categorical_encoding": "onehot",  # onehot | ordinal
        "handle_unknown": "ignore",
        "max_categories": 50,
    },
    "training": {
        "test_size": 0.2,
        "validation_size": 0.1,
        "stratify": True,
        "cv_folds": 5,
        "random_state": 42,
        "scoring": None,  # None -> auto-selected from task
    },
    "model": {
        "algorithm": "random_forest",
        "params": {},
    },
    "output": {
        "model_dir": "models",
        "report_dir": "reports",
        "figure_dir": "reports/figures",
        "log_dir": "logs",
        "overwrite": False,
    },
    "experiments": {
        "db": "experiments.db",
    },
}


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``.

    Dictionaries are merged recursively; every other value is replaced.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], dict(value))
        else:
            result[key] = copy.deepcopy(value)
    return result


# ---------------------------------------------------------------------------
# Configuration sections
# ---------------------------------------------------------------------------


@dataclass
class ProjectSection:
    name: str = "my-project"


@dataclass
class DataSection:
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    external_dir: str = "data/external"
    train: str = "data/processed/train.csv"
    validation: str = "data/processed/validation.csv"
    test: str = "data/processed/test.csv"
    target: str | None = None
    id_column: str | None = None
    drop_columns: list[str] = field(default_factory=list)


@dataclass
class CleaningSection:
    drop_duplicates: bool = True
    missing_strategy: str = "mean"
    missing_constant: int = 0
    drop_high_missing_threshold: float = 0.8
    outlier_method: str | None = None
    outlier_threshold: float = 3.0


@dataclass
class FeaturesSection:
    scale_numerical: bool = True
    scaler: str = "standard"
    categorical_encoding: str = "onehot"
    handle_unknown: str = "ignore"
    max_categories: int = 50


@dataclass
class TrainingSection:
    test_size: float = 0.2
    validation_size: float = 0.1
    stratify: bool = True
    cv_folds: int = 5
    random_state: int = 42
    scoring: str | None = None


@dataclass
class ModelSection:
    algorithm: str = "random_forest"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputSection:
    model_dir: str = "models"
    report_dir: str = "reports"
    figure_dir: str = "reports/figures"
    log_dir: str = "logs"
    overwrite: bool = False


@dataclass
class ExperimentSection:
    db: str = "experiments.db"


# Mapping of section name -> dataclass type, used for generic parsing.
_SECTIONS: dict[str, type] = {
    "project": ProjectSection,
    "data": DataSection,
    "cleaning": CleaningSection,
    "features": FeaturesSection,
    "training": TrainingSection,
    "model": ModelSection,
    "output": OutputSection,
    "experiments": ExperimentSection,
}

# Validation rules: section -> field -> allowed values.
_CHOICES: dict[str, dict[str, tuple[Any, ...]]] = {
    "cleaning": {
        "missing_strategy": ("mean", "median", "most_frequent", "constant", "drop"),
        "outlier_method": (None, "iqr", "zscore"),
    },
    "features": {
        "scaler": ("standard", "minmax", "robust"),
        "categorical_encoding": ("onehot", "ordinal"),
        "handle_unknown": ("ignore", "error"),
    },
}


def _from_dict(section_type: type, data: Mapping[str, Any], section_name: str) -> Any:
    """Build a section dataclass, ignoring unknown keys and validating choices."""
    valid_fields = {f for f in section_type.__dataclass_fields__}
    unknown = set(data) - valid_fields
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in config section '{section_name}': "
            f"{', '.join(sorted(unknown))}. Valid keys: {', '.join(sorted(valid_fields))}."
        )

    kwargs: dict[str, Any] = {}
    for field_name, field_def in section_type.__dataclass_fields__.items():
        value = data.get(field_name, field_def.default)
        choices = _CHOICES.get(section_name, {}).get(field_name)
        if choices is not None and value not in choices:
            raise ConfigError(
                f"Invalid value '{value}' for config key '{section_name}.{field_name}'. "
                f"Allowed values: {', '.join(str(c) for c in choices)}."
            )
        kwargs[field_name] = value
    return section_type(**kwargs)


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------


class Config:
    """Immutable-by-convention configuration bundle for a dscli project."""

    def __init__(
        self,
        project: ProjectSection,
        data: DataSection,
        cleaning: CleaningSection,
        features: FeaturesSection,
        training: TrainingSection,
        model: ModelSection,
        output: OutputSection,
        experiments: ExperimentSection,
        project_root: Path,
    ) -> None:
        self.project = project
        self.data = data
        self.cleaning = cleaning
        self.features = features
        self.training = training
        self.model = model
        self.output = output
        self.experiments = experiments
        self.project_root = Path(project_root)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], project_root: Path) -> "Config":
        """Build a Config from a (possibly partial) nested mapping."""
        merged = deep_merge(DEFAULT_CONFIG, dict(raw))
        sections = {
            name: _from_dict(section_type, merged.get(name, {}), name)
            for name, section_type in _SECTIONS.items()
        }
        return cls(project_root=Path(project_root), **sections)

    @classmethod
    def load(cls, path: str | Path, project_root: Path | None = None) -> "Config":
        """Load a Config from a YAML file.

        Raises :class:`ConfigError` if the file does not exist or is malformed.
        """
        config_path = Path(path)
        if not config_path.is_file():
            raise ConfigError(
                f"Configuration file not found: '{config_path}'. "
                "Run 'dscli init' to scaffold a project, or pass an existing file "
                "with --config."
            )
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in configuration file '{config_path}': {exc}") from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ConfigError(
                f"Configuration file '{config_path}' must contain a top-level mapping."
            )
        root = Path(project_root) if project_root is not None else config_path.parent.parent
        return cls.from_dict(raw, project_root=root)

    # -- path helpers ------------------------------------------------------

    def resolve(self, path: str) -> Path:
        """Resolve a config path relative to the project root."""
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()

    def data_dir(self, kind: str) -> Path:
        """Resolve one of the standard data directories (raw/interim/processed/external)."""
        return self.resolve(getattr(self.data, f"{kind}_dir"))

    @property
    def model_dir(self) -> Path:
        return self.resolve(self.output.model_dir)

    @property
    def report_dir(self) -> Path:
        return self.resolve(self.output.report_dir)

    @property
    def figure_dir(self) -> Path:
        return self.resolve(self.output.figure_dir)

    @property
    def log_dir(self) -> Path:
        return self.resolve(self.output.log_dir)

    @property
    def experiments_db(self) -> Path:
        return self.resolve(self.experiments.db)

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the effective configuration back to a nested dict."""
        return {name: asdict(getattr(self, name)) for name in _SECTIONS}

    def dump(self, path: str | Path) -> None:
        """Write the effective configuration to a YAML file."""
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8"
        )

    def with_overrides(self, overrides: Mapping[str, Any]) -> "Config":
        """Return a new Config with CLI overrides deep-merged on top."""
        raw = deep_merge(self.to_dict(), dict(overrides))
        return Config.from_dict(raw, project_root=self.project_root)


def default_config_path(project_root: Path) -> Path:
    """The conventional config location for a project."""
    return Path(project_root) / "configs" / "config.yaml"


def find_project_root(start: str | Path | None = None) -> Path | None:
    """Walk up from ``start`` (default: cwd) looking for a dscli project marker.

    A directory is considered a project root when it contains a
    ``configs/config.yaml`` or a ``dscli.yaml`` file.
    """
    current = Path(start).resolve() if start else Path.cwd().resolve()
    for directory in (current, *current.parents):
        if (directory / "configs" / "config.yaml").is_file():
            return directory
        if (directory / "dscli.yaml").is_file():
            return directory
    return None


def load_project_config(
    config_path: str | Path | None = None, project_dir: str | Path | None = None
) -> tuple[Config, Path]:
    """Load the effective config for the current project.

    Returns the resolved ``(Config, project_root)``. Raises
    :class:`ProjectError` when no project can be found, and
    :class:`ConfigError` when an explicitly given config is invalid.
    """
    from dscli.errors import ProjectError

    if project_dir is not None:
        root = Path(project_dir).resolve()
        if not root.is_dir():
            raise ProjectError(f"Project directory does not exist: '{root}'.")
    else:
        root = find_project_root()
        if root is None:
            raise ProjectError(
                "Not inside a dscli project. Run 'dscli init <name>' to create one, "
                "or pass --project-dir to point at an existing project."
            )

    if config_path is not None:
        config = Config.load(config_path, project_root=root)
    else:
        candidate = default_config_path(root)
        if not candidate.is_file():
            raise ProjectError(
                f"Not inside a dscli project: no configuration file found at "
                f"'{candidate}'. Run 'dscli init' in '{root}' to create one, or "
                f"pass --project-dir/--config to point at an existing project."
            )
        config = Config.load(candidate, project_root=root)

    # Ensure standard directories exist so downstream commands never fail
    # because a folder is missing.
    for directory in (
        config.data_dir("raw"),
        config.data_dir("interim"),
        config.data_dir("processed"),
        config.data_dir("external"),
        config.model_dir,
        config.report_dir,
        config.figure_dir,
        config.log_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return config, root


def ensure_project_init(config: Config) -> None:
    """Create all standard project directories for a freshly initialized project."""
    for directory in (
        config.data_dir("raw"),
        config.data_dir("interim"),
        config.data_dir("processed"),
        config.data_dir("external"),
        config.model_dir,
        config.report_dir,
        config.figure_dir,
        config.log_dir,
        config.resolve("notebooks"),
    ):
        directory.mkdir(parents=True, exist_ok=True)
