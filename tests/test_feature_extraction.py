from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from src.features.feature_extraction import (
    FeatureStack,
    build_feature_stack,
    save_feature_stack,
)
from src.preprocessing.raster_alignment import TargetGrid, align_raster_to_grid
from src.utils.errors import DataValidationError
from tests.fixtures import RASTER_MINX, write_synthetic_raster

CORE_BAND_INDICES = (1, 2, 3, 4)


def _aligned_sentinel_bands(
    tmp_path: Path,
    *,
    include_swir: bool = True,
    band_index: int = 1,
):
    target = write_synthetic_raster(tmp_path / "target.tif", count=4, nodata=0)
    grid = TargetGrid.from_raster(target)
    bands = {}
    for key, index in zip(("B02", "B03", "B04", "B08"), CORE_BAND_INDICES, strict=True):
        path = write_synthetic_raster(tmp_path / f"{key}.tif", count=4, nodata=0)
        bands[key] = align_raster_to_grid(
            path,
            grid,
            band_index=index,
            resampling="nearest",
            name=key,
        )
    if include_swir:
        for key, index in (("B11", 1), ("B12", 2)):
            path = write_synthetic_raster(tmp_path / f"{key}.tif", count=2, nodata=0)
            bands[key] = align_raster_to_grid(
                path,
                grid,
                band_index=index,
                resampling="nearest",
                name=key,
            )
    return bands


def test_build_feature_stack_with_core_bands(tmp_path: Path) -> None:
    bands = _aligned_sentinel_bands(tmp_path, include_swir=False)

    features = build_feature_stack(bands)

    assert features.feature_names == (
        "B02",
        "B03",
        "B04",
        "B08",
        "NDVI",
        "NDWI",
    )
    assert features.data.shape == (6, 40, 40)
    assert features.data.dtype == np.dtype("float32")
    assert features.valid_pixel_count == 1598
    assert features.total_pixel_count == 1600
    assert features.crs == CRS.from_epsg(32633)
    assert features.transform == bands["B02"].transform


def test_build_feature_stack_includes_swir_and_ndbi(tmp_path: Path) -> None:
    bands = _aligned_sentinel_bands(tmp_path, include_swir=True)

    features = build_feature_stack(bands)

    assert features.feature_names == (
        "B02",
        "B03",
        "B04",
        "B08",
        "B11",
        "B12",
        "NDVI",
        "NDWI",
        "NDBI",
    )
    assert features.band_names == ("B02", "B03", "B04", "B08", "B11", "B12")


def test_build_feature_stack_can_skip_indices(tmp_path: Path) -> None:
    bands = _aligned_sentinel_bands(tmp_path, include_swir=False)

    features = build_feature_stack(bands, include_indices=False)

    assert features.feature_names == ("B02", "B03", "B04", "B08")
    assert features.count == 4


def test_feature_sample_matrix_contains_only_valid_pixels(tmp_path: Path) -> None:
    bands = _aligned_sentinel_bands(tmp_path)

    features = build_feature_stack(bands)
    matrix, rows, cols = features.as_sample_matrix()

    assert matrix.shape == (features.valid_pixel_count, features.count)
    assert rows.shape == cols.shape == (features.valid_pixel_count,)
    assert np.isfinite(matrix).all()
    assert not features.valid_mask[0, 0]
    assert features.valid_mask[rows, cols].all()


def test_feature_statistics_are_computed_from_valid_pixels(tmp_path: Path) -> None:
    bands = _aligned_sentinel_bands(tmp_path)
    features = build_feature_stack(bands)

    stats = features.statistics()

    assert set(stats) == set(features.feature_names)
    assert stats["NDVI"]["count"] == features.valid_pixel_count
    assert stats["NDVI"]["min"] >= -1.0
    assert stats["NDVI"]["max"] <= 1.0
    assert stats["B02"]["mean"] > 0


def test_feature_stack_metadata_and_masked_view(tmp_path: Path) -> None:
    features = build_feature_stack(_aligned_sentinel_bands(tmp_path))

    metadata = features.to_dict()
    masked = features.to_masked()

    assert metadata["count"] == 9
    assert metadata["crs"] == "EPSG:32633"
    assert metadata["valid_pixels"] == 1598
    assert masked.mask[0, 0, 0]
    assert not masked.mask[0, 1, 1]


def test_build_feature_stack_resolves_aliases(tmp_path: Path) -> None:
    bands = _aligned_sentinel_bands(tmp_path, include_swir=False)
    aliased = {
        "blue": bands["B02"],
        "green": bands["B03"],
        "red": bands["B04"],
        "nir": bands["B08"],
    }

    features = build_feature_stack(aliased)

    assert features.band_names == ("B02", "B03", "B04", "B08")


def test_build_feature_stack_rejects_invalid_band_inputs(tmp_path: Path) -> None:
    bands = _aligned_sentinel_bands(tmp_path, include_swir=False)

    with pytest.raises(DataValidationError, match="Missing required"):
        build_feature_stack({"B02": bands["B02"], "B03": bands["B03"], "B04": bands["B04"]})
    with pytest.raises(DataValidationError, match="more than once"):
        build_feature_stack({**bands, "red": bands["B04"]})
    with pytest.raises(DataValidationError, match="Unknown Sentinel-2 band"):
        build_feature_stack({**bands, "B01": bands["B02"]})
    with pytest.raises(DataValidationError, match="requires B11"):
        build_feature_stack(bands, include_ndbi=True)


def test_build_feature_stack_rejects_mismatched_grids(tmp_path: Path) -> None:
    bands = _aligned_sentinel_bands(tmp_path, include_swir=False)
    small = write_synthetic_raster(
        tmp_path / "small.tif",
        width=10,
        height=10,
        transform=Affine(10.0, 0.0, RASTER_MINX, 0.0, -10.0, 5_000_000.0),
    )
    small_grid = TargetGrid.from_raster(small)
    bands["B08"] = align_raster_to_grid(small, small_grid, name="B08")

    with pytest.raises(DataValidationError, match="not on the same grid"):
        build_feature_stack(bands)


def test_save_feature_stack_writes_georeferenced_multiband_file(tmp_path: Path) -> None:
    features = build_feature_stack(_aligned_sentinel_bands(tmp_path))
    output = tmp_path / "nested" / "features.tif"

    result = save_feature_stack(features, output)

    assert result == output.resolve()
    with rasterio.open(output) as dataset:
        assert dataset.count == 9
        assert dataset.crs == CRS.from_epsg(32633)
        assert dataset.dtypes == ("float32",) * 9
        assert dataset.descriptions == features.feature_names
        assert np.isnan(dataset.nodata)
        assert np.isnan(dataset.read(1, masked=False)[0, 0])


def test_feature_stack_validates_internal_shapes() -> None:
    transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
    crs = CRS.from_epsg(32633)
    data = np.ones((2, 4, 4), dtype=np.float32)

    with pytest.raises(DataValidationError, match="must be 3D"):
        FeatureStack(
            data=np.ones((4, 4), dtype=np.float32),
            feature_names=("B02",),
            transform=transform,
            crs=crs,
            valid_mask=np.ones((4, 4), dtype=bool),
            band_names=("B02",),
        )
    with pytest.raises(DataValidationError, match="names"):
        FeatureStack(
            data=data,
            feature_names=("B02",),
            transform=transform,
            crs=crs,
            valid_mask=np.ones((4, 4), dtype=bool),
            band_names=("B02",),
        )
    with pytest.raises(DataValidationError, match="at least one feature"):
        FeatureStack(
            data=np.zeros((0, 4, 4), dtype=np.float32),
            feature_names=(),
            transform=transform,
            crs=crs,
            valid_mask=np.ones((4, 4), dtype=bool),
            band_names=(),
        )
    with pytest.raises(DataValidationError, match="Valid mask shape"):
        FeatureStack(
            data=data,
            feature_names=("B02", "B03"),
            transform=transform,
            crs=crs,
            valid_mask=np.ones((2, 2), dtype=bool),
            band_names=("B02",),
        )
