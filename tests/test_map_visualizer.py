from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely import box

from src.utils.errors import DataValidationError, RasterError, VectorError
from src.visualization.map_visualizer import (
    boundary_map,
    change_map,
    classification_map,
    continuous_raster_map,
)
from tests.fixtures import (
    RASTER_CRS,
    RASTER_MAXY,
    RASTER_MINX,
    write_synthetic_aoi_wgs84,
    write_synthetic_raster,
    write_synthetic_worldcover,
)


def _write_change(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.array([[0, 1, 255], [1, 0, 255]], dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=3,
        height=2,
        count=1,
        dtype="uint8",
        crs=RASTER_CRS,
        transform=from_origin(RASTER_MINX, RASTER_MAXY, 10.0, 10.0),
        nodata=255,
    ) as dataset:
        dataset.write(values, 1)
    return path


def test_classification_map_renders_colours_and_legend(tmp_path: Path) -> None:
    path = write_synthetic_worldcover(tmp_path / "classified.tif")

    raster_map = classification_map(path)
    html = raster_map.get_root().render()

    assert "Land-cover classes" in html
    assert "Tree Cover" in html
    assert "Cropland" in html
    assert "Land cover" in html


def test_classification_map_rejects_unknown_codes(tmp_path: Path) -> None:
    path = write_synthetic_worldcover(
        tmp_path / "unknown.tif",
        codes=np.array([[10, 99]], dtype=np.uint8),
        width=2,
        height=1,
        nodata=None,
    )

    with pytest.raises(DataValidationError, match="unknown class codes"):
        classification_map(path)


def test_classification_map_requires_georeferencing(tmp_path: Path) -> None:
    path = write_synthetic_worldcover(tmp_path / "plain.tif", crs=None)

    with pytest.raises(RasterError, match="no CRS"):
        classification_map(path)


def test_continuous_map_renders_band(tmp_path: Path) -> None:
    path = write_synthetic_raster(tmp_path / "stack.tif")

    raster_map = continuous_raster_map(path, band_index=2, name="NIR preview")
    html = raster_map.get_root().render()

    assert "NIR preview" in html
    assert "OpenStreetMap" in html


def test_change_map_renders_binary_classes(tmp_path: Path) -> None:
    path = _write_change(tmp_path / "change.tif")

    raster_map = change_map(path)
    html = raster_map.get_root().render()

    assert "Change detection" in html
    assert "Unchanged" in html
    assert "Changed" in html


def test_boundary_map_renders_geojson_layer(tmp_path: Path) -> None:
    path = write_synthetic_aoi_wgs84(tmp_path / "aoi.geojson")
    frame = gpd.read_file(path)

    raster_map = boundary_map(frame)
    html = raster_map.get_root().render()

    assert "Study area" in html
    assert "synthetic-aoi-wgs84" in html


def test_boundary_map_rejects_missing_crs() -> None:
    frame = gpd.GeoDataFrame(
        {"name": ["no-crs"]},
        geometry=[box(0, 0, 1, 1)],
        crs=None,
    )
    with pytest.raises(VectorError, match="without a CRS"):
        boundary_map(frame)
