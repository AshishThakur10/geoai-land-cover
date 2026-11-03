from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _clear_caches():
    from config.settings import get_settings
    from src.utils.logging_config import reset_logging

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
    reset_logging()
