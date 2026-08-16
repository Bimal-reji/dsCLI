"""Logging setup for dscli.

Logs go to two places:

* a rotating file in ``logs/dscli.log`` (structured, includes timestamps), and
* the console via Rich's handler when ``--verbose`` is passed.

Commands use :func:`get_logger` and normal ``logging`` calls; user-facing
output is handled separately through Rich in the CLI layer.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

_LOGGER_NAME = "dscli"


def setup_logging(log_dir: Path, verbose: bool = False) -> logging.Logger:
    """Configure the root ``dscli`` logger and return it.

    ``log_dir`` is created if missing. When ``verbose`` is true, log records
    are also streamed to the console at DEBUG level.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Avoid stacking duplicate handlers when setup is called more than once
    # (e.g. from tests that invoke the CLI repeatedly in one process).
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    file_handler = RotatingFileHandler(
        log_dir / "dscli.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    if verbose:
        console_handler = RichHandler(
            level=logging.DEBUG,
            markup=True,
            show_time=False,
            show_path=False,
        )
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str = "dscli") -> logging.Logger:
    """Return a child logger of the ``dscli`` logger."""
    return logging.getLogger(f"{_LOGGER_NAME}.{name}" if name != _LOGGER_NAME else _LOGGER_NAME)


def log_exception(logger: logging.Logger, exc: BaseException) -> None:
    """Record an exception at ERROR level including the traceback."""
    logger.error("Unhandled error: %s: %s", type(exc).__name__, exc, exc_info=sys.exc_info())
