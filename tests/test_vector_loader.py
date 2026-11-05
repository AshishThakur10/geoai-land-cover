from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from pyproj import CRS
from shapely import Polygon, box

from src.data.vector_loader import (
    geodesic_area_km2,
    load_aoi,
    load_vector,
    read_vector_metadata,
    reproject_vector,
    validate_geometries,
    validate_vector_path,
    vector_bounds,
)
from src.utils.errors import DataValidationError, VectorError
from tests.fixtures import (
    RASTER_CRS,
    write_synthetic_aoi_geojson,
    write_synthetic_aoi_wgs84,
    write_synthetic_shapefile,
)


def test_load_geojson(tmp_path: Path) -> None:
    path = write_synthetic_aoi_geojson(tmp_path / "aoi.geojson")
    frame = load_vector(path)

    assert len(frame) == 1
    assert frame.crs == CRS.from_epsg(32633)
    assert frame.geometry.iloc[0].is_valid


def test_load_shapefile(tmp_path: Path) -> None:
    path = write_synthetic_shapefile(tmp_path / "aoi.shp")
    frame = load_vector(path)

    assert len(frame) == 1
    assert frame.crs == CRS.from_epsg(32633)


def test_validate_vector_path_rejects_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="does not exist"):
        validate_vector_path(tmp_path / "missing.geojson")

    unsupported = tmp_path / "aoi.txt"
    unsupported.write_text("{}", encoding="utf-8")
    with pytest.raises(DataValidationError, match="Unsupported file type"):
        validate_vector_path(unsupported)


def test_load_vector_raises_for_invalid_content(tmp_path: Path) -> None:
    path = tmp_path / "broken.geojson"
    path.write_text("not geojson", encoding="utf-8")

    with pytest.raises(VectorError, match="Unable to read vector file"):
        load_vector(path)


def test_reproject_vector(tmp_path: Path) -> None:
    path = write_synthetic_aoi_wgs84(tmp_path / "aoi_wgs84.geojson")
    frame = load_vector(path)
    assert frame.crs == CRS.from_epsg(4326)

    reprojected = reproject_vector(frame, RASTER_CRS)

    assert reprojected.crs == CRS.from_epsg(32633)
    assert reprojected.geometry.iloc[0].is_valid
    minx, miny, maxx, maxy = reprojected.total_bounds
    assert minx == pytest.approx(500_050, abs=1.0)
    assert maxy == pytest.approx(4_999_950, abs=1.0)


def test_reproject_vector_requires_source_crs() -> None:
    frame = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)])

    with pytest.raises(VectorError, match="without a source CRS"):
        reproject_vector(frame, RASTER_CRS)


def test_reproject_vector_rejects_invalid_target_crs() -> None:
    frame = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:32633")

    with pytest.raises(VectorError, match="Invalid target CRS"):
        reproject_vector(frame, "not-a-crs")


def test_validate_geometries_repairs_bowtie() -> None:
    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
    frame = gpd.GeoDataFrame({"id": [1]}, geometry=[bowtie], crs="EPSG:32633")
    assert not bool(frame.geometry.is_valid.iloc[0])

    repaired = validate_geometries(frame)

    assert len(repaired) == 1
    assert bool(repaired.geometry.is_valid.iloc[0])


def test_validate_geometries_drops_null_and_empty() -> None:
    frame = gpd.GeoDataFrame(
        {"id": [1, 2, 3]},
        geometry=[None, Polygon(), box(0, 0, 1, 1)],
        crs="EPSG:32633",
    )

    result = validate_geometries(frame)

    assert len(result) == 1
    assert result["id"].tolist() == [3]


def test_validate_geometries_raises_when_nothing_usable() -> None:
    frame = gpd.GeoDataFrame({"id": [1]}, geometry=[None], crs="EPSG:32633")

    with pytest.raises(VectorError, match="no usable geometries"):
        validate_geometries(frame)


def test_load_aoi_reprojects_to_target_crs(tmp_path: Path) -> None:
    path = write_synthetic_aoi_wgs84(tmp_path / "aoi_wgs84.geojson")

    frame = load_aoi(path, target_crs=RASTER_CRS)

    assert frame.crs == CRS.from_epsg(32633)


def test_geodesic_area_is_in_square_kilometres() -> None:
    square_1km = box(500_000, 4_999_000, 501_000, 5_000_000)
    frame = gpd.GeoDataFrame(geometry=[square_1km], crs=RASTER_CRS)

    area = geodesic_area_km2(frame)

    assert area == pytest.approx(1.0, abs=0.01)


def test_geodesic_area_requires_crs() -> None:
    frame = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)])

    with pytest.raises(VectorError, match="without a CRS"):
        geodesic_area_km2(frame)


def test_read_vector_metadata(tmp_path: Path) -> None:
    path = write_synthetic_aoi_geojson(tmp_path / "aoi.geojson")

    metadata = read_vector_metadata(path)

    assert metadata.feature_count == 1
    assert metadata.crs == CRS.from_epsg(32633)
    assert metadata.geometry_types == ("Polygon",)
    assert metadata.is_georeferenced
    assert metadata.bounds is not None
    assert metadata.to_dict()["crs"] == "EPSG:32633"


def test_vector_bounds(tmp_path: Path) -> None:
    path = write_synthetic_aoi_geojson(tmp_path / "aoi.geojson")
    frame = load_vector(path)

    minx, miny, maxx, maxy = vector_bounds(frame)

    assert (minx, maxx) == pytest.approx((500_050.0, 500_150.0))
    assert (miny, maxy) == pytest.approx((4_999_850.0, 4_999_950.0))
