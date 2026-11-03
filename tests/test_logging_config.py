from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.utils.errors import ConfigurationError
from src.utils.logging_config import (
    get_logger,
    normalize_level,
    reset_logging,
    setup_logging,
)


def test_normalize_level() -> None:
    assert normalize_level("info") == "INFO"
    assert normalize_level(" debug ") == "DEBUG"
    with pytest.raises(ConfigurationError, match="Invalid log level"):
        normalize_level("verbose")


def test_setup_logging_writes_to_file(tmp_path: Path) -> None:
    log_file = tmp_path / "nested" / "geoai.log"
    setup_logging("DEBUG", log_file, force=True)

    logger = get_logger("tests.logging")
    logger.info("classification pipeline started")

    for handler in logging.getLogger().handlers:
        handler.flush()

    content = log_file.read_text(encoding="utf-8")
    assert "classification pipeline started" in content
    assert "| INFO" in content


def test_setup_logging_is_idempotent(tmp_path: Path) -> None:
    log_file = tmp_path / "geoai.log"
    setup_logging("INFO", log_file, force=True)
    first_count = len(logging.getLogger().handlers)

    setup_logging("INFO", log_file)
    second_count = len(logging.getLogger().handlers)

    assert first_count == second_count == 2


def test_setup_logging_force_replaces_handlers(tmp_path: Path) -> None:
    setup_logging("INFO", tmp_path / "first.log", force=True)
    setup_logging("DEBUG", tmp_path / "second.log", force=True)

    assert len(logging.getLogger().handlers) == 2
    assert logging.getLogger().level == logging.DEBUG


def test_reset_logging_clears_handlers(tmp_path: Path) -> None:
    setup_logging("INFO", tmp_path / "geoai.log", force=True)
    reset_logging()

    assert logging.getLogger().handlers == []
