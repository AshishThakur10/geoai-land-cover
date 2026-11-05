from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from src.preprocessing.image_stacking import (
    RasterStack,
    save_stack,
    stack_aligned_rasters,
    stack_band_paths,
)
from src.preprocessing.raster_alignment import TargetGrid, align_raster_to_grid
from src.utils.errors import DataValidationError
from tests.fixtures import RASTER_MINX, write_synthetic_raster


def _aligned_bands(tmp_path: Path, *, nodata: float | None = 0):
    target = write_synthetic_raster(tmp_path / "target.tif", width=40, height=40, nodata=nodata)
    grid = TargetGrid.from_raster(target)
    paths = {
        "B02": (write_synthetic_raster(tmp_path / "B02.tif", nodata=nodata), 1),
        "B03": (write_synthetic_raster(tmp_path / "B03.tif", nodata=nodata), 2),
    }
    return {
        name: align_raster_to_grid(
            path,
            grid,
            band_index=band_index,
            resampling="nearest",
            name=name,
        )
        for name, (path, band_index) in paths.items()
    }


def test_stack_aligned_rasters_preserves_order_and_mask(tmp_path: Path) -> None:
    aligned = _aligned_bands(tmp_path)

    stack = stack_aligned_rasters(aligned)

    assert stack.band_names == ("B02", "B03")
    assert stack.data.shape == (2, 40, 40)
    assert stack.data.dtype == np.dtype("float32")
    assert stack.valid_mask.sum() == 1598
    assert stack.valid_pixel_count == 1598
    assert not stack.valid_mask[0, 0]
    assert stack.to_dict()["crs"] == "EPSG:32633"
    assert stack.to_masked().mask[0, 0, 0]
    assert not stack.to_masked().mask[0, 1, 1]


def test_stack_respects_custom_band_order(tmp_path: Path) -> None:
    aligned = _aligned_bands(tmp_path)

    stack = stack_aligned_rasters(aligned, band_order=("B03", "B02"))

    assert stack.band_names == ("B03", "B02")
    assert stack.data[0, 0, 1] == aligned["B03"].data[0, 1]


def test_stack_rejects_invalid_band_specifications(tmp_path: Path) -> None:
    aligned = _aligned_bands(tmp_path)

    with pytest.raises(DataValidationError, match="At least one"):
        stack_aligned_rasters({})
    with pytest.raises(DataValidationError, match="unknown bands"):
        stack_aligned_rasters(aligned, band_order=("B02", "B08"))
    with pytest.raises(DataValidationError, match="duplicate"):
        stack_aligned_rasters(aligned, band_order=("B02", "B02"))
    with pytest.raises(DataValidationError, match="must not be empty"):
        stack_aligned_rasters(aligned, band_order=())


def test_stack_rejects_mismatched_grids(tmp_path: Path) -> None:
    aligned = _aligned_bands(tmp_path)
    small = write_synthetic_raster(
        tmp_path / "small.tif",
        width=10,
        height=10,
        transform=from_origin(RASTER_MINX, 5_000_000.0, 10.0, 10.0),
    )
    small_grid = TargetGrid.from_raster(small)
    aligned["B04"] = align_raster_to_grid(small, small_grid, name="B04")

    with pytest.raises(DataValidationError, match="not on the same grid"):
        stack_aligned_rasters(aligned)


def test_stack_band_paths_aligns_and_stacks(tmp_path: Path) -> None:
    paths = {
        "B02": write_synthetic_raster(tmp_path / "B02.tif", width=40, height=40),
        "B03": write_synthetic_raster(tmp_path / "B03.tif", width=40, height=40),
    }

    stack = stack_band_paths(paths, reference_band="B02", resampling="nearest")

    assert stack.band_names == ("B02", "B03")
    assert stack.width == 40
    assert stack.height == 40


def test_save_stack_writes_multiband_geotiff(tmp_path: Path) -> None:
    aligned = _aligned_bands(tmp_path)
    stack = stack_aligned_rasters(aligned)
    output = tmp_path / "nested" / "stack.tif"

    result = save_stack(stack, output)

    assert result == output.resolve()
    with rasterio.open(output) as dataset:
        assert dataset.count == 2
        assert dataset.dtypes == ("float32", "float32")
        assert dataset.crs == CRS.from_epsg(32633)
        assert dataset.transform == from_origin(RASTER_MINX, 5_000_000.0, 10.0, 10.0)
        assert dataset.descriptions == ("B02", "B03")
        assert np.isnan(dataset.nodata)
        assert np.isnan(dataset.read(1, masked=False)[0, 0])


def test_save_stack_rejects_bad_descriptions_and_suffix(tmp_path: Path) -> None:
    stack = stack_aligned_rasters(_aligned_bands(tmp_path))

    with pytest.raises(DataValidationError, match="band descriptions"):
        save_stack(stack, tmp_path / "stack.tif", band_descriptions=("only-one",))
    with pytest.raises(DataValidationError, match="Unsupported output type"):
        save_stack(stack, tmp_path / "stack.png")


def test_raster_stack_validates_shapes(tmp_path: Path) -> None:
    aligned = _aligned_bands(tmp_path)
    stack = stack_aligned_rasters(aligned)

    with pytest.raises(DataValidationError, match="must be 3D"):
        RasterStack(
            data=np.zeros((40, 40), dtype=np.float32),
            band_names=("B02",),
            transform=stack.transform,
            crs=stack.crs,
            nodata=np.nan,
            valid_mask=np.ones((40, 40), dtype=bool),
        )
    with pytest.raises(DataValidationError, match="band names"):
        RasterStack(
            data=stack.data,
            band_names=("B02",),
            transform=stack.transform,
            crs=stack.crs,
            nodata=np.nan,
            valid_mask=stack.valid_mask,
        )
    with pytest.raises(DataValidationError, match="Valid mask shape"):
        RasterStack(
            data=stack.data,
            band_names=stack.band_names,
            transform=stack.transform,
            crs=stack.crs,
            nodata=np.nan,
            valid_mask=np.ones((2, 2), dtype=bool),
        )
    with pytest.raises(DataValidationError, match="at least one band"):
        RasterStack(
            data=np.zeros((0, 40, 40), dtype=np.float32),
            band_names=(),
            transform=stack.transform,
            crs=stack.crs,
            nodata=np.nan,
            valid_mask=np.ones((40, 40), dtype=bool),
        )
