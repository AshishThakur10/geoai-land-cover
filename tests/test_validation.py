from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.errors import DataValidationError
from src.utils.validation import (
    require_existing_file,
    require_supported_suffix,
    validate_non_empty_string,
    validate_uploaded_file,
)


def test_validate_non_empty_string() -> None:
    assert validate_non_empty_string("  study-area ", "name") == "study-area"
    for value in ("", "   ", None, 123):
        with pytest.raises(DataValidationError, match="non-empty string"):
            validate_non_empty_string(value, "name")


def test_require_supported_suffix(tmp_path: Path) -> None:
    assert require_supported_suffix(tmp_path / "aoi.GeoJSON", [".geojson"]) == (
        tmp_path / "aoi.GeoJSON"
    )
    with pytest.raises(DataValidationError, match="Unsupported file type"):
        require_supported_suffix(tmp_path / "aoi.txt", [".geojson"])


def test_require_existing_file(tmp_path: Path) -> None:
    valid = tmp_path / "aoi.geojson"
    valid.write_text("{}", encoding="utf-8")

    assert require_existing_file(valid, [".geojson"]) == valid

    with pytest.raises(DataValidationError, match="does not exist"):
        require_existing_file(tmp_path / "missing.geojson")

    with pytest.raises(DataValidationError, match="not a file"):
        require_existing_file(tmp_path)

    with pytest.raises(DataValidationError, match="Unsupported file type"):
        require_existing_file(valid, [".tif"])


def test_validate_uploaded_file_accepts_boundary_size() -> None:
    validate_uploaded_file(
        "imagery.TIF",
        200 * 1024 * 1024,
        max_size_mb=200,
        allowed_suffixes=[".tif", ".tiff"],
    )


@pytest.mark.parametrize(
    ("filename", "size", "message"),
    [
        ("", 10, "non-empty string"),
        ("imagery.tif", 0, "empty"),
        ("imagery.tif", None, "empty"),
        ("imagery.tif", 201 * 1024 * 1024, "maximum allowed"),
        ("imagery.zip", 10, "Unsupported file type"),
    ],
)
def test_validate_uploaded_file_rejects_invalid_input(
    filename: str, size: int | None, message: str
) -> None:
    with pytest.raises(DataValidationError, match=message):
        validate_uploaded_file(
            filename,
            size,
            max_size_mb=200,
            allowed_suffixes=[".tif", ".tiff"],
        )
