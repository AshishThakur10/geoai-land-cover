"""Logging configuration shared by the library, CLI-style scripts, and dashboard."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.utils.errors import ConfigurationError

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 5_000_000
_BACKUP_COUNT = 3

_configured = False


def normalize_level(level: str) -> str:
    """Validate and normalise a logging level name."""
    candidate = str(level).strip().upper()
    if candidate not in logging.getLevelNamesMapping():
        allowed = ", ".join(
            name
            for name in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
            if name in logging.getLevelNamesMapping()
        )
        raise ConfigurationError(f"Invalid log level {level!r}; expected one of {allowed}")
    return candidate


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    *,
    force: bool = False,
) -> logging.Logger:
    """Configure the root logger with console and optional rotating-file handlers.

    Repeated calls are no-ops unless ``force`` is true, which prevents duplicate
    handlers when Streamlit reruns a script.
    """
    global _configured

    root = logging.getLogger()
    if _configured and not force:
        return root

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    normalized_level = normalize_level(level)
    root.setLevel(normalized_level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _configured = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)


def reset_logging() -> None:
    """Remove handlers configured by :func:`setup_logging` (mainly for tests)."""
    global _configured
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    _configured = False
