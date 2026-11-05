from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from src.data.raster_loader import (
    assert_same_grid,
    count_valid_pixels,
    ensure_georeferenced,
    open_raster,
    read_all_bands,
    read_band,
    read_metadata,
    valid_data_mask,
    validate_raster_path,
)
from src.utils.errors import DataValidationError, RasterError
from tests.fixtures import default_transform, write_synthetic_raster


def test_read_metadata(tmp_path: Path) -> None:
    path = write_synthetic_raster(tmp_path / "image.tif", width=20, height=10, count=4, nodata=0)

    metadata = read_metadata(path)

    assert metadata.path == path
    assert metadata.driver == "GTiff"
    assert metadata.width == 20
    assert metadata.height == 10
    assert metadata.count == 4
    assert metadata.dtypes == ("uint16", "uint16", "uint16", "uint16")
    assert metadata.crs == CRS.from_epsg(32633)
    assert metadata.transform == default_transform()
    assert metadata.nodata == 0
    assert metadata.descriptions == ("Blue", "Green", "Red", "NIR")
    assert metadata.resolutions == (10.0, 10.0)
    assert metadata.is_georeferenced
    assert metadata.crs_wkt is not None

    serialised = metadata.to_dict()
    assert serialised["crs"] == "EPSG:32633"
    assert serialised["width"] == 20
    assert serialised["bounds"] == list(metadata.bounds)
    assert serialised["is_georeferenced"] is True


def test_read_band_and_all_bands(tmp_path: Path) -> None:
    path = write_synthetic_raster(tmp_path / "image.tif", width=8, height=6, count=3, nodata=0)

    band = read_band(path, band_index=2)
    assert band.shape == (6, 8)
    assert np.ma.isMaskedArray(band)
    assert bool(band.mask[0, 0])
    assert bool(band.mask[-1, -1])
    assert not bool(band.mask[1, 1])

    all_bands = read_all_bands(path)
    assert all_bands.shape == (3, 6, 8)

    unmasked = read_band(path, band_index=1, masked=False)
    assert not np.ma.isMaskedArray(unmasked)
    assert unmasked[0, 0] == 0


def test_read_band_out_of_range(tmp_path: Path) -> None:
    path = write_synthetic_raster(tmp_path / "image.tif", count=2)

    with pytest.raises(RasterError, match="out of range"):
        read_band(path, band_index=3)


def test_validate_raster_path_rejects_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="does not exist"):
        validate_raster_path(tmp_path / "missing.tif")

    text_file = tmp_path / "image.txt"
    text_file.write_text("not a raster", encoding="utf-8")
    with pytest.raises(DataValidationError, match="Unsupported file type"):
        validate_raster_path(text_file)


def test_corrupt_raster_raises_raster_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.tif"
    path.write_bytes(b"this is not a geotiff")

    with pytest.raises(RasterError, match="Unable to open raster"):
        read_metadata(path)


def test_open_raster_closes_dataset(tmp_path: Path) -> None:
    path = write_synthetic_raster(tmp_path / "image.tif")

    with open_raster(path) as dataset:
        assert not dataset.closed

    assert dataset.closed


def test_valid_data_mask_handles_nodata_nan_and_masks() -> None:
    values = np.array([[1.0, np.nan], [0.0, 5.0]])
    mask = valid_data_mask(values, nodata=0)
    assert mask.tolist() == [[True, False], [False, True]]

    masked = np.ma.array([[1, 2], [3, 4]], mask=[[False, True], [False, False]])
    assert valid_data_mask(masked).tolist() == [[True, False], [True, True]]

    nan_nodata = np.array([[1.0, np.nan], [2.0, 3.0]])
    assert valid_data_mask(nan_nodata, nodata=float("nan")).tolist() == [
        [True, False],
        [True, True],
    ]


def test_count_valid_pixels(tmp_path: Path) -> None:
    path = write_synthetic_raster(tmp_path / "image.tif", width=10, height=10, count=1, nodata=0)
    band = read_band(path)

    assert count_valid_pixels(band) == 98


def test_ensure_georeferenced(tmp_path: Path) -> None:
    path = write_synthetic_raster(tmp_path / "no_crs.tif", crs=None)
    metadata = read_metadata(path)

    assert metadata.crs is None
    assert not metadata.is_georeferenced

    with pytest.raises(RasterError, match="no CRS"):
        ensure_georeferenced(metadata)


def test_assert_same_grid(tmp_path: Path) -> None:
    first = read_metadata(write_synthetic_raster(tmp_path / "first.tif"))
    second = read_metadata(write_synthetic_raster(tmp_path / "second.tif"))
    assert_same_grid(first, second)

    shifted = read_metadata(
        write_synthetic_raster(
            tmp_path / "shifted.tif",
            transform=from_origin(500_010.0, 5_000_000.0, 10.0, 10.0),
        )
    )
    with pytest.raises(DataValidationError, match="Transform mismatch"):
        assert_same_grid(first, shifted)

    smaller = read_metadata(write_synthetic_raster(tmp_path / "smaller.tif", width=10, height=10))
    with pytest.raises(DataValidationError, match="Dimension mismatch"):
        assert_same_grid(first, smaller)
