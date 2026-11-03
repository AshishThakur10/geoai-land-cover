"""Small, dependency-free file and path helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from src.utils.errors import DataValidationError

_SIZE_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


def ensure_directory(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it.

    Raises:
        DataValidationError: If the directory cannot be created or is not a directory.
    """
    directory = Path(path)
    if directory.exists() and not directory.is_dir():
        raise DataValidationError(f"Path exists but is not a directory: {directory}")
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DataValidationError(f"Cannot create directory {directory}: {exc}") from exc
    return directory


def ensure_directories(paths: Iterable[str | Path]) -> None:
    """Create multiple directories."""
    for path in paths:
        ensure_directory(path)


def human_readable_size(num_bytes: int | float) -> str:
    """Format a byte count using binary units.

    Raises:
        ValueError: If ``num_bytes`` is negative.
    """
    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")
    size = float(num_bytes)
    for unit in _SIZE_UNITS:
        if size < 1024 or unit == _SIZE_UNITS[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def normalize_suffixes(suffixes: Iterable[str]) -> frozenset[str]:
    """Normalise file suffixes to lowercase with a leading dot."""
    normalized = set()
    for suffix in suffixes:
        value = suffix.strip().lower()
        if not value:
            continue
        normalized.add(value if value.startswith(".") else f".{value}")
    return frozenset(normalized)


def is_supported_file(path: str | Path, suffixes: Iterable[str]) -> bool:
    """Return ``True`` if ``path`` is a file with an allowed suffix."""
    candidate = Path(path)
    return candidate.is_file() and candidate.suffix.lower() in normalize_suffixes(suffixes)


def safe_filename(filename: str, fallback: str = "upload") -> str:
    """Return a filesystem-safe basename derived from an uploaded filename."""
    candidate = Path(str(filename).replace("\\", "/")).name
    cleaned = _UNSAFE_FILENAME.sub("_", candidate).strip("._")
    return cleaned or fallback
