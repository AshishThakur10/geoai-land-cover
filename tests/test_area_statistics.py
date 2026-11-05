from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from pyproj import Geod
from rasterio.transform import from_origin

from src.analysis.area_statistics import (
    compute_area_statistics,
    save_area_statistics_csv,
)
from src.utils.errors import DataValidationError, RasterError
from tests.fixtures import RASTER_MAXY, RASTER_MINX


def _write_classification(
    path: Path,
    codes: np.ndarray,
    *,
    crs: str | None = "EPSG:32633",
    transform: Affine | None = None,
    nodata: float | None = 0,
    dtype: str = "uint8",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=codes.shape[1],
        height=codes.shape[0],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform or from_origin(RASTER_MINX, RASTER_MAXY, 10.0, 10.0),
        nodata=nodata,
    ) as dataset:
        dataset.write(codes.astype(dtype), 1)
    return path


def _geodesic_cell_area_km2(
    longitude: float,
    latitude: float,
    width: float,
    height: float,
) -> float:
    geod = Geod(ellps="WGS84")
    area, _ = geod.polygon_area_perimeter(
        [longitude, longitude + width, longitude + width, longitude],
        [latitude, latitude, latitude + height, latitude + height],
    )
    return abs(area) / 1_000_000.0


def test_projected_area_statistics_use_pixel_area(tmp_path: Path) -> None:
    codes = np.array(
        [
            [10, 10, 40, 40, 0],
            [10, 10, 40, 40, 0],
            [30, 30, 30, 40, 0],
            [30, 30, 30, 40, 0],
        ],
        dtype=np.uint8,
    )
    path = _write_classification(tmp_path / "classification.tif", codes)

    result = compute_area_statistics(path, window_size=2)

    assert result.area_method == "projected"
    assert result.pixel_area_km2 == pytest.approx(0.0001)
    assert result.crs.to_epsg() == 32633
    assert result.total_pixels == 20
    assert result.valid_pixels == 16
    assert result.nodata_pixels == 4
    assert result.total_area_km2 == pytest.approx(0.0016)
    assert [item.code for item in result.classes] == [10, 30, 40]
    assert result.classes[0].name == "Tree Cover"
    assert result.classes[1].name == "Grassland"
    assert result.classes[2].name == "Cropland"
    assert [item.pixel_count for item in result.classes] == [4, 6, 6]
    assert [item.area_km2 for item in result.classes] == pytest.approx(
        [0.0004, 0.0006, 0.0006]
    )
    assert [item.percentage for item in result.classes] == pytest.approx(
        [25.0, 37.5, 37.5]
    )
    assert result.to_dict()["classes"][0]["color"] == "#006400"


def test_geographic_area_statistics_use_geodesic_cells(tmp_path: Path) -> None:
    transform = from_origin(10.0, 20.0, 0.01, 0.01)
    codes = np.array([[10, 10], [30, 0]], dtype=np.uint8)
    path = _write_classification(
        tmp_path / "geographic.tif",
        codes,
        crs="EPSG:4326",
        transform=transform,
    )

    result = compute_area_statistics(path)

    row0_area = _geodesic_cell_area_km2(10.005, 19.99, 0.01, 0.01)
    row1_area = _geodesic_cell_area_km2(10.005, 19.98, 0.01, 0.01)
    assert result.area_method == "geodesic"
    assert result.pixel_area_km2 is None
    assert result.classes[0].area_km2 == pytest.approx(2 * row0_area, rel=1e-6)
    assert result.classes[1].area_km2 == pytest.approx(row1_area, rel=1e-6)
    assert result.total_area_km2 > 1.0
    assert result.total_area_km2 < 5.0


def test_area_statistics_support_no_nodata_and_nan_nodata(tmp_path: Path) -> None:
    codes = np.array([[10, 40], [30, 10]], dtype=np.uint8)
    no_nodata = _write_classification(
        tmp_path / "no_nodata.tif",
        codes,
        nodata=None,
    )
    nan_nodata = _write_classification(
        tmp_path / "nan_nodata.tif",
        codes,
        nodata=0,
    )

    without_nodata = compute_area_statistics(no_nodata, nodata=None)
    with_nan = compute_area_statistics(nan_nodata, nodata=np.nan)

    assert without_nodata.nodata is None
    assert without_nodata.nodata_pixels == 0
    assert without_nodata.valid_pixels == 4
    assert with_nan.nodata is not None
    assert np.isnan(with_nan.nodata)
    assert with_nan.valid_pixels == 4


def test_geographic_statistics_skip_all_nodata_rows(tmp_path: Path) -> None:
    path = _write_classification(
        tmp_path / "geographic_gap.tif",
        np.array([[10, 10], [0, 0]], dtype=np.uint8),
        crs="EPSG:4326",
        transform=from_origin(10.0, 20.0, 0.01, 0.01),
    )

    result = compute_area_statistics(path)

    assert result.valid_pixels == 2
    assert result.nodata_pixels == 2
    assert result.classes[0].code == 10
    assert result.classes[0].area_km2 == pytest.approx(
        2 * _geodesic_cell_area_km2(10.005, 19.99, 0.01, 0.01),
        rel=1e-6,
    )


def test_area_statistics_csv_contains_distribution(tmp_path: Path) -> None:
    codes = np.array([[10, 10, 40, 0]], dtype=np.uint8)
    path = _write_classification(tmp_path / "classification.tif", codes)
    statistics = compute_area_statistics(path)
    destination = tmp_path / "nested" / "area_statistics.csv"

    saved = save_area_statistics_csv(statistics, destination)

    assert saved == destination.resolve()
    with destination.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["code"] for row in rows] == ["10", "40"]
    assert rows[0]["land_cover_type"] == "Tree Cover"
    assert rows[0]["pixel_count"] == "2"
    assert rows[0]["area_km2"] == "0.000200"
    assert rows[0]["percentage"] == "66.666667"


def test_area_statistics_reject_invalid_rasters(tmp_path: Path) -> None:
    valid_codes = np.array([[10, 40], [30, 0]], dtype=np.uint8)
    path = _write_classification(tmp_path / "classification.tif", valid_codes)

    with pytest.raises(DataValidationError, match="Band index"):
        compute_area_statistics(path, band_index=2)
    with pytest.raises(DataValidationError, match="window_size"):
        compute_area_statistics(path, window_size=0)
    with pytest.raises(DataValidationError, match="nodata must be numeric"):
        compute_area_statistics(path, nodata=True)
    with pytest.raises(DataValidationError, match="no valid land-cover"):
        compute_area_statistics(
            _write_classification(tmp_path / "empty.tif", np.zeros((2, 2), dtype=np.uint8))
        )
    with pytest.raises(DataValidationError, match="unknown class codes"):
        compute_area_statistics(
            _write_classification(
                tmp_path / "unknown.tif",
                np.array([[10, 99]], dtype=np.uint8),
            )
        )
    with pytest.raises(DataValidationError, match="integer dtype"):
        compute_area_statistics(
            _write_classification(
                tmp_path / "float.tif",
                np.array([[10.0, 40.0]], dtype=np.float32),
                dtype="float32",
                nodata=0.0,
            )
        )
    with pytest.raises(DataValidationError, match="overlaps"):
        compute_area_statistics(
            _write_classification(
                tmp_path / "invalid_nodata.tif",
                valid_codes,
                nodata=10,
            )
        )
    with pytest.raises(RasterError, match="CRS"):
        compute_area_statistics(
            _write_classification(
                tmp_path / "no_crs.tif",
                valid_codes,
                crs=None,
            )
        )


def test_area_statistics_reject_rotated_geographic_grid(tmp_path: Path) -> None:
    path = _write_classification(
        tmp_path / "rotated.tif",
        np.array([[10, 10], [30, 0]], dtype=np.uint8),
        crs="EPSG:4326",
        transform=Affine(0.01, 0.001, 10.0, 0.0, -0.01, 20.0),
    )

    with pytest.raises(DataValidationError, match="Rotated geographic"):
        compute_area_statistics(path)


def test_save_area_statistics_rejects_non_csv(tmp_path: Path) -> None:
    path = _write_classification(
        tmp_path / "classification.tif",
        np.array([[10, 40]], dtype=np.uint8),
    )
    statistics = compute_area_statistics(path)

    with pytest.raises(DataValidationError, match="Unsupported statistics"):
        save_area_statistics_csv(statistics, tmp_path / "area.txt")
