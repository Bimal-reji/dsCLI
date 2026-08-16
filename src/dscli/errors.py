"""Exception hierarchy for dscli.

All errors that should be shown to the user as clean, understandable
messages (rather than raw tracebacks) derive from :class:`DScliError`.
"""

from __future__ import annotations


class DScliError(Exception):
    """Base class for all dscli errors.

    The message should be written for a data scientist, not a library
    consumer: explain *what* went wrong and, where useful, *how to fix it*.
    """


class ConfigError(DScliError):
    """Raised when the project configuration is missing or invalid."""


class ProjectError(DScliError):
    """Raised when the current directory is not inside a dscli project."""


class DataError(DScliError):
    """Raised when a dataset cannot be loaded or written."""


class ValidationError(DScliError):
    """Raised when a dataset fails validation."""


class ModelError(DScliError):
    """Raised for problems related to model training or persistence."""


class EvaluationError(DScliError):
    """Raised when metrics cannot be computed."""


class ExperimentError(DScliError):
    """Raised for problems with the experiment tracker."""


class ReportError(DScliError):
    """Raised when report or figure generation fails."""
