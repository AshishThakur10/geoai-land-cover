from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from pyproj import CRS
from shapely import box

from src.data.dataset_validator import (
    align_aoi_to_raster,
    aoi_intersects_raster,
    validate_raster_vector_compatibility,
)
from src.data.raster_loader import read_metadata
from src.data.vector_loader import load_vector
from src.utils.errors import DataValidationError
from tests.fixtures import (
    RASTER_CRS,
    write_synthetic_aoi_geojson,
    write_synthetic_aoi_wgs84,
    write_synthetic_raster,
)


def test_align_aoi_reprojects_to_raster_crs(tmp_path: Path) -> None:
    raster = read_metadata(write_synthetic_raster(tmp_path / "raster.tif"))
    aoi = load_vector(write_synthetic_aoi_wgs84(tmp_path / "aoi.geojson"))

    aligned = align_aoi_to_raster(aoi, raster)

    assert aligned.crs == CRS.from_epsg(32633)
    assert aligned.geometry.iloc[0].is_valid


def test_validate_compatibility_returns_aligned_aoi(tmp_path: Path) -> None:
    raster = read_metadata(write_synthetic_raster(tmp_path / "raster.tif"))
    aoi = load_vector(write_synthetic_aoi_geojson(tmp_path / "aoi.geojson"))

    aligned = validate_raster_vector_compatibility(raster, aoi)

    assert aligned.crs == raster.crs
    assert len(aligned) == 1


def test_non_overlapping_aoi_is_rejected(tmp_path: Path) -> None:
    raster = read_metadata(write_synthetic_raster(tmp_path / "raster.tif"))
    far_away = gpd.GeoDataFrame(
        geometry=[box(0, 0, 0.01, 0.01)],
        crs="EPSG:4326",
    )

    with pytest.raises(DataValidationError, match="does not overlap"):
        validate_raster_vector_compatibility(raster, far_away)


def test_aoi_without_crs_is_rejected(tmp_path: Path) -> None:
    raster = read_metadata(write_synthetic_raster(tmp_path / "raster.tif"))
    aoi = gpd.GeoDataFrame(geometry=[box(500_000, 4_999_000, 500_100, 4_999_100)])

    with pytest.raises(DataValidationError, match="AOI has no CRS"):
        validate_raster_vector_compatibility(raster, aoi)


def test_raster_without_crs_is_rejected(tmp_path: Path) -> None:
    raster = read_metadata(write_synthetic_raster(tmp_path / "raster.tif", crs=None))
    aoi = gpd.GeoDataFrame(
        geometry=[box(500_000, 4_999_000, 500_100, 4_999_100)],
        crs=RASTER_CRS,
    )

    with pytest.raises(DataValidationError, match="Raster has no CRS"):
        validate_raster_vector_compatibility(raster, aoi)


def test_aoi_intersects_raster(tmp_path: Path) -> None:
    raster = read_metadata(write_synthetic_raster(tmp_path / "raster.tif"))
    overlapping = load_vector(write_synthetic_aoi_geojson(tmp_path / "aoi.geojson"))
    far_away = gpd.GeoDataFrame(geometry=[box(0, 0, 0.01, 0.01)], crs="EPSG:4326")

    assert aoi_intersects_raster(overlapping, raster)
    assert not aoi_intersects_raster(far_away, raster)
