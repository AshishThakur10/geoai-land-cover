from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from src.preprocessing.raster_alignment import (
    TargetGrid,
    align_bands,
    align_raster_to_grid,
    resolve_resampling,
    write_aligned_raster,
)
from src.utils.errors import DataValidationError, RasterError
from tests.fixtures import (
    RASTER_CRS,
    RASTER_MAXY,
    RASTER_MINX,
    write_synthetic_raster,
)


def _reference_grid(tmp_path: Path) -> TargetGrid:
    reference = write_synthetic_raster(tmp_path / "reference.tif", width=40, height=40)
    return TargetGrid.from_raster(reference)


def test_target_grid_from_raster(tmp_path: Path) -> None:
    grid = _reference_grid(tmp_path)

    assert grid.crs == CRS.from_epsg(32633)
    assert (grid.width, grid.height) == (40, 40)
    assert grid.resolution == (10.0, 10.0)
    assert grid.bounds.left == RASTER_MINX
    assert grid.bounds.top == RASTER_MAXY
    assert grid.bounds.right == RASTER_MINX + 400
    assert grid.bounds.bottom == RASTER_MAXY - 400


def test_target_grid_from_bounds_snaps_outward() -> None:
    grid = TargetGrid.from_bounds(
        (500_053.0, 4_999_653.0, 500_147.0, 4_999_747.0),
        RASTER_CRS,
        10.0,
    )

    assert (grid.width, grid.height) == (10, 10)
    assert grid.transform == from_origin(500_050.0, 4_999_750.0, 10.0, 10.0)
    assert grid.to_dict()["crs"] == "EPSG:32633"


def test_target_grid_rejects_invalid_dimensions_and_bounds() -> None:
    with pytest.raises(DataValidationError, match="must be positive"):
        TargetGrid(CRS.from_epsg(32633), from_origin(0, 0, 1, 1), 0, 10)
    with pytest.raises(DataValidationError, match="positive and finite"):
        TargetGrid.from_bounds((0, 0, 1, 1), "EPSG:32633", 0)
    with pytest.raises(DataValidationError, match="Invalid bounds"):
        TargetGrid.from_bounds((1, 1, 0, 0), "EPSG:32633", 10)
    with pytest.raises(DataValidationError, match="finite"):
        TargetGrid.from_bounds((0, 0, np.inf, 1), "EPSG:32633", 10)
    with pytest.raises(DataValidationError, match="Invalid CRS"):
        TargetGrid.from_bounds((0, 0, 1, 1), "not-a-crs", 10)


def test_align_same_grid_preserves_values_and_masks_nodata(tmp_path: Path) -> None:
    source = write_synthetic_raster(tmp_path / "source.tif", width=40, height=40, nodata=0)
    grid = _reference_grid(tmp_path)

    aligned = align_raster_to_grid(source, grid, resampling="nearest")

    assert aligned.grid == grid
    assert aligned.data.shape == (40, 40)
    assert aligned.data.dtype == np.dtype("float32")
    assert aligned.resampling == "nearest"
    assert aligned.valid_mask.sum() == 1598
    assert bool(np.ma.getmaskarray(aligned.data)[0, 0])
    assert bool(np.ma.getmaskarray(aligned.data)[-1, -1])
    assert aligned.data[0, 1] == pytest.approx(2.0)


def test_align_resamples_to_reference_resolution(tmp_path: Path) -> None:
    source = write_synthetic_raster(
        tmp_path / "coarse.tif",
        width=20,
        height=20,
        transform=from_origin(RASTER_MINX, RASTER_MAXY, 20.0, 20.0),
        nodata=None,
    )
    grid = _reference_grid(tmp_path)

    aligned = align_raster_to_grid(source, grid, resampling=Resampling.bilinear)

    assert aligned.data.shape == (40, 40)
    assert aligned.valid_mask.all()
    assert aligned.data[0, 0] == pytest.approx(1.0, abs=0.75)
    assert aligned.to_dict()["source"] == str(source)


def test_align_reprojects_geographic_source(tmp_path: Path) -> None:
    source = write_synthetic_raster(
        tmp_path / "wgs84.tif",
        width=20,
        height=20,
        crs="EPSG:4326",
        transform=from_origin(12.0, 45.0, 0.0001, 0.0001),
        nodata=None,
    )
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    center_x, center_y = transformer.transform(12.001, 44.999)
    grid = TargetGrid.from_bounds(
        (center_x - 100, center_y - 100, center_x + 100, center_y + 100),
        "EPSG:32633",
        10.0,
    )

    aligned = align_raster_to_grid(source, grid, resampling="bilinear")

    assert aligned.crs == CRS.from_epsg(32633)
    assert aligned.data.shape == (grid.height, grid.width)
    assert aligned.valid_pixel_count > 0


def test_align_bands_uses_reference_grid(tmp_path: Path) -> None:
    paths = {
        "blue": write_synthetic_raster(tmp_path / "blue.tif", width=40, height=40),
        "red": write_synthetic_raster(tmp_path / "red.tif", width=40, height=40),
    }
    grid = _reference_grid(tmp_path)

    aligned = align_bands(paths, reference_band="blue", resampling="nearest")

    assert set(aligned) == {"blue", "red"}
    assert all(raster.grid == grid for raster in aligned.values())
    assert aligned["red"].name == "red"


def test_align_bands_rejects_unknown_reference(tmp_path: Path) -> None:
    paths = {"blue": write_synthetic_raster(tmp_path / "blue.tif")}

    with pytest.raises(DataValidationError, match="Reference band"):
        align_bands(paths, reference_band="red")


def test_align_bands_requires_at_least_one_band() -> None:
    with pytest.raises(DataValidationError, match="At least one"):
        align_bands({})


def test_align_rejects_invalid_inputs(tmp_path: Path) -> None:
    source = write_synthetic_raster(tmp_path / "source.tif")
    no_crs = write_synthetic_raster(tmp_path / "no_crs.tif", crs=None)
    grid = _reference_grid(tmp_path)

    with pytest.raises(DataValidationError, match="Unsupported resampling"):
        resolve_resampling("quantum")
    with pytest.raises(RasterError, match="no CRS"):
        align_raster_to_grid(no_crs, grid)
    with pytest.raises(DataValidationError, match="out of range"):
        align_raster_to_grid(source, grid, band_index=99)
    with pytest.raises(DataValidationError, match="does not exist"):
        align_raster_to_grid(tmp_path / "missing.tif", grid)


def test_write_aligned_raster_preserves_georeferencing(tmp_path: Path) -> None:
    source = write_synthetic_raster(tmp_path / "source.tif", nodata=0)
    grid = _reference_grid(tmp_path)
    aligned = align_raster_to_grid(source, grid, resampling="nearest")
    output = tmp_path / "nested" / "aligned.tif"

    result = write_aligned_raster(aligned, output)

    assert result == output.resolve()
    with rasterio.open(output) as dataset:
        assert dataset.count == 1
        assert dataset.dtypes == ("float32",)
        assert dataset.crs == CRS.from_epsg(32633)
        assert dataset.transform == grid.transform
        assert np.isnan(dataset.nodata)
        assert dataset.descriptions == ("source",)
        assert np.isnan(dataset.read(1, masked=False)[0, 0])


def test_write_aligned_raster_rejects_unsupported_suffix(tmp_path: Path) -> None:
    source = write_synthetic_raster(tmp_path / "source.tif")
    grid = _reference_grid(tmp_path)
    aligned = align_raster_to_grid(source, grid)

    with pytest.raises(DataValidationError, match="Unsupported output type"):
        write_aligned_raster(aligned, tmp_path / "aligned.png")
