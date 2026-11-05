from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from pyproj import CRS
from rasterio.transform import from_origin
from shapely import box

from src.data.vector_loader import load_vector
from src.preprocessing.raster_clipping import clip_raster_to_aoi
from src.utils.errors import DataValidationError, VectorError
from tests.fixtures import (
    RASTER_CRS,
    synthetic_aoi_polygon,
    write_synthetic_aoi_geojson,
    write_synthetic_raster,
)


def test_clip_preserves_crs_transform_and_dimensions(tmp_path: Path) -> None:
    raster = write_synthetic_raster(tmp_path / "raster.tif", width=40, height=40)
    aoi = write_synthetic_aoi_geojson(tmp_path / "aoi.geojson")

    result = clip_raster_to_aoi(raster, aoi)

    assert result.crs == CRS.from_epsg(32633)
    assert result.data.shape == (4, 10, 10)
    assert result.transform == from_origin(500_050.0, 4_999_950.0, 10.0, 10.0)
    assert result.height == 10
    assert result.width == 10
    assert result.count == 4
    assert result.source_metadata.path == raster
    assert result.valid_pixel_count == result.total_pixel_count == 100


def test_clip_writes_georeferenced_output_with_nodata(tmp_path: Path) -> None:
    raster = write_synthetic_raster(tmp_path / "raster.tif", width=40, height=40, nodata=0)
    aoi = write_synthetic_aoi_geojson(tmp_path / "aoi.geojson")
    output = tmp_path / "nested" / "clipped.tif"

    result = clip_raster_to_aoi(raster, aoi, output)

    assert result.output_path == output.resolve()
    assert output.exists()

    with rasterio.open(output) as dataset:
        assert dataset.crs == CRS.from_epsg(32633)
        assert dataset.nodata == 0
        assert dataset.width == 10
        assert dataset.height == 10
        assert dataset.count == 4
        assert dataset.transform == from_origin(500_050.0, 4_999_950.0, 10.0, 10.0)
        assert dataset.descriptions == ("Blue", "Green", "Red", "NIR")
        assert dataset.read(1).shape == (10, 10)


def test_clip_marks_pixels_outside_aoi_as_nodata(tmp_path: Path) -> None:
    raster = write_synthetic_raster(tmp_path / "raster.tif", width=40, height=40, nodata=0)
    aoi = write_synthetic_aoi_geojson(tmp_path / "aoi.geojson")

    result = clip_raster_to_aoi(raster, aoi, crop=False)

    assert result.width == 40
    assert result.height == 40
    assert result.valid_pixel_count < result.total_pixel_count
    assert bool(result.data.mask[0, 0, 0])
    assert not bool(result.data.mask[0, 5, 5])
    assert np.count_nonzero(~np.ma.getmaskarray(result.data[0])) > 0


def test_clip_without_source_nodata_uses_internal_mask(tmp_path: Path) -> None:
    raster = write_synthetic_raster(tmp_path / "raster.tif", width=40, height=40, nodata=None)
    aoi = write_synthetic_aoi_geojson(tmp_path / "aoi.geojson")
    output = tmp_path / "clipped_masked.tif"

    result = clip_raster_to_aoi(raster, aoi, output, crop=False)

    assert result.nodata is None
    with rasterio.open(output) as dataset:
        assert dataset.nodata is None
        band = dataset.read(1, masked=True)
        assert bool(band.mask[0, 0])
        assert not bool(band.mask[5, 5])


def test_clip_accepts_geodataframe_and_repairs_geometry(tmp_path: Path) -> None:
    raster = write_synthetic_raster(tmp_path / "raster.tif", width=40, height=40)
    frame = gpd.GeoDataFrame(
        geometry=[synthetic_aoi_polygon()],
        crs=RASTER_CRS,
    )

    result = clip_raster_to_aoi(raster, frame)
    assert result.data.shape == (4, 10, 10)


def test_clip_with_aoi_path_reprojects_automatically(tmp_path: Path) -> None:
    raster = write_synthetic_raster(tmp_path / "raster.tif", width=40, height=40)
    aoi = write_synthetic_aoi_geojson(tmp_path / "aoi.geojson")

    result = clip_raster_to_aoi(raster, aoi)
    assert result.crs == CRS.from_epsg(32633)


def test_clip_rejects_non_overlapping_aoi(tmp_path: Path) -> None:
    raster = write_synthetic_raster(tmp_path / "raster.tif")
    far_away = gpd.GeoDataFrame(geometry=[box(0, 0, 0.01, 0.01)], crs="EPSG:4326")

    with pytest.raises(DataValidationError, match="does not overlap"):
        clip_raster_to_aoi(raster, far_away)


def test_clip_rejects_output_matching_input(tmp_path: Path) -> None:
    raster = write_synthetic_raster(tmp_path / "raster.tif")
    aoi = write_synthetic_aoi_geojson(tmp_path / "aoi.geojson")

    with pytest.raises(DataValidationError, match="must differ"):
        clip_raster_to_aoi(raster, aoi, raster)


def test_clip_rejects_unsupported_output_suffix(tmp_path: Path) -> None:
    raster = write_synthetic_raster(tmp_path / "raster.tif")
    aoi = write_synthetic_aoi_geojson(tmp_path / "aoi.geojson")

    with pytest.raises(DataValidationError, match="Unsupported output type"):
        clip_raster_to_aoi(raster, aoi, tmp_path / "clipped.png")


def test_clip_result_filled_uses_nodata(tmp_path: Path) -> None:
    raster = write_synthetic_raster(tmp_path / "raster.tif", width=40, height=40, nodata=0)
    aoi = write_synthetic_aoi_geojson(tmp_path / "aoi.geojson")

    result = clip_raster_to_aoi(raster, aoi, crop=False)
    filled = result.filled()

    assert filled.shape == (4, 40, 40)
    assert filled.dtype == np.dtype("uint16")
    assert filled[0, 0, 0] == 0


def test_clip_invalid_aoi_file_is_reported(tmp_path: Path) -> None:
    raster = write_synthetic_raster(tmp_path / "raster.tif")
    invalid = tmp_path / "invalid.geojson"
    invalid.write_text("not geojson", encoding="utf-8")

    with pytest.raises(VectorError, match="Unable to read vector file"):
        clip_raster_to_aoi(raster, invalid)


def test_loaded_aoi_geometry_is_inside_raster(tmp_path: Path) -> None:
    aoi_path = write_synthetic_aoi_geojson(tmp_path / "aoi.geojson")
    frame = load_vector(aoi_path)
    minx, miny, maxx, maxy = frame.total_bounds

    assert minx >= 500_000 and maxx <= 500_400
    assert miny >= 4_999_600 and maxy <= 5_000_000
