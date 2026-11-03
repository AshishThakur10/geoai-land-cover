from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.errors import DataValidationError
from src.utils.file_utils import (
    ensure_directories,
    ensure_directory,
    human_readable_size,
    is_supported_file,
    normalize_suffixes,
    safe_filename,
)


def test_ensure_directory_creates_nested_paths(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    result = ensure_directory(target)

    assert result == target
    assert target.is_dir()


def test_ensure_directory_rejects_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("data", encoding="utf-8")

    with pytest.raises(DataValidationError, match="not a directory"):
        ensure_directory(target)


def test_ensure_directories(tmp_path: Path) -> None:
    paths = [tmp_path / "one", tmp_path / "two" / "nested"]
    ensure_directories(paths)

    assert all(path.is_dir() for path in paths)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (0, "0 B"),
        (512, "512 B"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (1024**2, "1.0 MB"),
        (1024**3, "1.0 GB"),
    ],
)
def test_human_readable_size(size: int, expected: str) -> None:
    assert human_readable_size(size) == expected


def test_human_readable_size_rejects_negative() -> None:
    with pytest.raises(ValueError):
        human_readable_size(-1)


def test_normalize_suffixes() -> None:
    assert normalize_suffixes(["TIF", ".tiff", " GeoJSON "]) == frozenset(
        {".tif", ".tiff", ".geojson"}
    )


def test_is_supported_file(tmp_path: Path) -> None:
    raster = tmp_path / "image.TIF"
    raster.write_bytes(b"fake")

    assert is_supported_file(raster, [".tif", ".tiff"])
    assert not is_supported_file(raster, [".geojson"])
    assert not is_supported_file(tmp_path / "missing.tif", [".tif"])


def test_safe_filename() -> None:
    assert safe_filename("../evil name.tif") == "evil_name.tif"
    assert safe_filename("folder\\layer.geojson") == "layer.geojson"
    assert safe_filename("..", fallback="upload") == "upload"
    assert safe_filename("", fallback="upload") == "upload"
