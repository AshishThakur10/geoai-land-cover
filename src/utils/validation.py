"""Reusable validation helpers for files and user input."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.utils.errors import DataValidationError
from src.utils.file_utils import normalize_suffixes


def validate_non_empty_string(value: object, field_name: str) -> str:
    """Return a stripped non-empty string, or raise ``DataValidationError``."""
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def require_supported_suffix(path: str | Path, suffixes: Iterable[str]) -> Path:
    """Validate that ``path`` has one of the allowed suffixes."""
    candidate = Path(path)
    allowed = normalize_suffixes(suffixes)
    if candidate.suffix.lower() not in allowed:
        formatted = ", ".join(sorted(allowed)) or "(none configured)"
        raise DataValidationError(
            f"Unsupported file type for {candidate.name!r}; expected one of {formatted}"
        )
    return candidate


def require_existing_file(path: str | Path, suffixes: Iterable[str] | None = None) -> Path:
    """Validate that ``path`` exists, is a file, and optionally has an allowed suffix."""
    candidate = Path(path)
    if not candidate.exists():
        raise DataValidationError(f"File does not exist: {candidate}")
    if not candidate.is_file():
        raise DataValidationError(f"Path is not a file: {candidate}")
    if suffixes is not None:
        require_supported_suffix(candidate, suffixes)
    return candidate


def validate_uploaded_file(
    filename: str,
    size_bytes: int | None,
    *,
    max_size_mb: int,
    allowed_suffixes: Iterable[str],
) -> None:
    """Validate an uploaded file's name and size before processing it."""
    name = validate_non_empty_string(filename, "filename")
    require_supported_suffix(name, allowed_suffixes)
    if size_bytes is None or size_bytes <= 0:
        raise DataValidationError("Uploaded file is empty")
    max_bytes = max_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise DataValidationError(
            f"Uploaded file is {size_bytes / (1024 * 1024):.1f} MB; "
            f"maximum allowed is {max_size_mb} MB"
        )
