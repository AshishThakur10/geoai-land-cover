from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from pyproj import Geod
from rasterio.transform import from_origin

from src.analysis.change_detection import (
    CHANGE_NODATA,
    CHANGED_VALUE,
    UNCHANGED_VALUE,
    detect_changes,
)
from src.analysis.transition_matrix import (
    CHANGE_CAVEAT,
    save_transition_matrix_csv,
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


def _read_change(path: Path) -> np.ndarray:
    with rasterio.open(path) as dataset:
        return dataset.read(1)


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


def _known_change_rasters(tmp_path: Path) -> tuple[Path, Path]:
    earlier = _write_classification(
        tmp_path / "earlier.tif",
        np.array(
            [
                [10, 10, 40, 40],
                [10, 30, 40, 0],
                [30, 30, 30, 0],
            ],
            dtype=np.uint8,
        ),
    )
    later = _write_classification(
        tmp_path / "later.tif",
        np.array(
            [
                [10, 50, 40, 40],
                [10, 30, 60, 0],
                [30, 30, 80, 0],
            ],
            dtype=np.uint8,
        ),
    )
    return earlier, later


def test_detect_changes_builds_expected_matrix_and_raster(tmp_path: Path) -> None:
    earlier, later = _known_change_rasters(tmp_path)
    output = tmp_path / "outputs" / "change.tif"

    result = detect_changes(earlier, later, output, window_size=2)

    assert result.output_path == output.resolve()
    assert result.alignment == "strict"
    assert result.total_pixels == 12
    assert result.valid_pixels == 10
    assert result.nodata_pixels == 2
    assert result.unchanged_pixels == 7
    assert result.changed_pixels == 3
    assert result.changed_percentage == pytest.approx(30.0)
    assert result.total_area_km2 == pytest.approx(0.0010)
    assert result.changed_area_km2 == pytest.approx(0.0003)
    assert result.matrix.classes == (10, 30, 40, 50, 60, 80)
    assert result.matrix.matrix_pixels[0][0] == 2
    assert result.matrix.matrix_pixels[2][2] == 2
    assert [
        (item.from_code, item.to_code, item.pixel_count)
        for item in result.matrix.changed_transitions
    ] == [(10, 50, 1), (30, 80, 1), (40, 60, 1)]
    assert result.to_dict()["scientific_caveat"] == CHANGE_CAVEAT

    expected = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, CHANGE_NODATA],
            [0, 0, 1, CHANGE_NODATA],
        ],
        dtype=np.uint8,
    )
    change = _read_change(output)
    assert np.array_equal(change, expected)
    with rasterio.open(output) as dataset:
        assert dataset.crs.to_epsg() == 32633
        assert dataset.transform == from_origin(RASTER_MINX, RASTER_MAXY, 10.0, 10.0)
        assert dataset.width == 4
        assert dataset.height == 3
        assert dataset.nodata == CHANGE_NODATA
        assert dataset.dtypes == ("uint8",)
        assert dataset.descriptions == ("Land-cover change (0 unchanged, 1 changed)",)
        assert dataset.tags()["CHANGED_VALUE"] == "1"
        assert dataset.tags()["UNCHANGED_VALUE"] == "0"
        assert dataset.tags()["SCIENTIFIC_CAVEAT"] == CHANGE_CAVEAT
        assert json.loads(dataset.tags()["CHANGE_LEGEND"])[1]["value"] == CHANGED_VALUE
        assert dataset.colormap(1)[UNCHANGED_VALUE] == (217, 217, 217, 255)
        assert dataset.colormap(1)[CHANGED_VALUE] == (220, 30, 30, 255)


def test_change_detection_is_window_size_independent(tmp_path: Path) -> None:
    earlier, later = _known_change_rasters(tmp_path)
    small = detect_changes(
        earlier,
        later,
        tmp_path / "small.tif",
        window_size=1,
    )
    large = detect_changes(
        earlier,
        later,
        tmp_path / "large.tif",
        window_size=50,
    )

    assert small.matrix.matrix_pixels == large.matrix.matrix_pixels
    assert small.matrix.matrix_areas_km2 == large.matrix.matrix_areas_km2
    assert np.array_equal(
        _read_change(small.output_path),
        _read_change(large.output_path),
    )


def test_geographic_change_areas_use_geodesic_cells(tmp_path: Path) -> None:
    transform = from_origin(10.0, 20.0, 0.01, 0.01)
    earlier = _write_classification(
        tmp_path / "earlier_geo.tif",
        np.array([[10, 10], [30, 30]], dtype=np.uint8),
        crs="EPSG:4326",
        transform=transform,
    )
    later = _write_classification(
        tmp_path / "later_geo.tif",
        np.array([[10, 40], [30, 80]], dtype=np.uint8),
        crs="EPSG:4326",
        transform=transform,
    )

    result = detect_changes(earlier, later, tmp_path / "change_geo.tif")

    row0_area = _geodesic_cell_area_km2(10.005, 19.99, 0.01, 0.01)
    row1_area = _geodesic_cell_area_km2(10.005, 19.98, 0.01, 0.01)
    assert result.matrix.area_method == "geodesic"
    assert result.matrix.pixel_area_km2 is None
    assert result.total_area_km2 == pytest.approx(2 * row0_area + 2 * row1_area)
    assert result.changed_area_km2 == pytest.approx(row0_area + row1_area)
    transition_areas = {
        (item.from_code, item.to_code): item.area_km2
        for item in result.matrix.changed_transitions
    }
    assert transition_areas[(10, 40)] == pytest.approx(row0_area)
    assert transition_areas[(30, 80)] == pytest.approx(row1_area)


def test_strict_alignment_rejects_misaligned_inputs(tmp_path: Path) -> None:
    earlier = _write_classification(
        tmp_path / "earlier.tif",
        np.full((3, 3), 10, dtype=np.uint8),
    )
    later = _write_classification(
        tmp_path / "later_shifted.tif",
        np.full((3, 3), 10, dtype=np.uint8),
        transform=from_origin(RASTER_MINX + 5.0, RASTER_MAXY, 10.0, 10.0),
    )

    with pytest.raises(DataValidationError, match="alignment='reproject'"):
        detect_changes(earlier, later, tmp_path / "change.tif")


def test_reproject_alignment_warps_later_onto_earlier_grid(tmp_path: Path) -> None:
    earlier = _write_classification(
        tmp_path / "earlier.tif",
        np.full((4, 4), 10, dtype=np.uint8),
    )
    later = _write_classification(
        tmp_path / "later_shifted.tif",
        np.full((4, 4), 50, dtype=np.uint8),
        transform=from_origin(RASTER_MINX + 10.0, RASTER_MAXY, 10.0, 10.0),
    )

    result = detect_changes(
        earlier,
        later,
        tmp_path / "change.tif",
        alignment="reproject",
        window_size=2,
    )

    assert result.alignment == "reproject"
    assert result.crs.to_epsg() == 32633
    assert result.width == 4
    assert result.height == 4
    assert result.matrix.classes == (10, 50)
    assert result.valid_pixels == 12
    assert result.changed_pixels == 12
    assert result.changed_area_km2 == pytest.approx(0.0012)
    change = _read_change(result.output_path)
    assert np.all(change[:, 0] == CHANGE_NODATA)
    assert np.all(change[:, 1:] == CHANGED_VALUE)


def test_reproject_alignment_propagates_source_nodata(tmp_path: Path) -> None:
    earlier = _write_classification(
        tmp_path / "earlier.tif",
        np.full((3, 3), 10, dtype=np.uint8),
    )
    later = _write_classification(
        tmp_path / "later.tif",
        np.array(
            [
                [50, 50, 50],
                [50, 0, 50],
                [50, 50, 50],
            ],
            dtype=np.uint8,
        ),
    )

    result = detect_changes(
        earlier,
        later,
        tmp_path / "change.tif",
        alignment="reproject",
        window_size=1,
    )

    assert result.valid_pixels == 8
    assert result.changed_pixels == 8
    assert result.nodata_pixels == 1
    assert _read_change(result.output_path)[1, 1] == CHANGE_NODATA


def test_transition_matrix_csv_contains_all_cells(tmp_path: Path) -> None:
    earlier, later = _known_change_rasters(tmp_path)
    result = detect_changes(earlier, later, tmp_path / "change.tif")
    destination = tmp_path / "tables" / "transitions.csv"

    saved = save_transition_matrix_csv(result.matrix, destination)

    assert saved == destination.resolve()
    with destination.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 36
    changed = [
        row
        for row in rows
        if row["from_code"] == "10" and row["to_code"] == "50"
    ]
    assert changed[0]["pixel_count"] == "1"
    assert changed[0]["area_km2"] == "0.000100"
    assert changed[0]["percentage_of_valid"] == "10.000000"
    assert changed[0]["percentage_of_changed"] == "33.333333"
    assert changed[0]["changed"] == "yes"
    unchanged = [
        row
        for row in rows
        if row["from_code"] == "10" and row["to_code"] == "10"
    ]
    assert unchanged[0]["changed"] == "no"


def test_change_detection_rejects_invalid_inputs(tmp_path: Path) -> None:
    earlier, later = _known_change_rasters(tmp_path)
    output = tmp_path / "change.tif"

    with pytest.raises(DataValidationError, match="alignment mode"):
        detect_changes(earlier, later, output, alignment="guess")
    with pytest.raises(DataValidationError, match="earlier_band_index"):
        detect_changes(earlier, later, output, earlier_band_index=0)
    with pytest.raises(DataValidationError, match="window_size"):
        detect_changes(earlier, later, output, window_size=0)
    with pytest.raises(DataValidationError, match="Unsupported change raster"):
        detect_changes(earlier, later, tmp_path / "change.png")
    with pytest.raises(DataValidationError, match="band index"):
        detect_changes(earlier, later, output, earlier_band_index=2)
    with pytest.raises(DataValidationError, match="integer dtype"):
        detect_changes(
            _write_classification(
                tmp_path / "float.tif",
                np.array([[10.0, 40.0]], dtype=np.float32),
                dtype="float32",
                nodata=0.0,
            ),
            later,
            output,
        )
    with pytest.raises(DataValidationError, match="unknown class codes"):
        detect_changes(
            _write_classification(
                tmp_path / "unknown.tif",
                np.array([[10, 99, 10, 10]] * 3, dtype=np.uint8),
            ),
            later,
            output,
        )
    with pytest.raises(DataValidationError, match="overlaps"):
        detect_changes(earlier, later, output, earlier_nodata=10)
    with pytest.raises(DataValidationError, match="no pixels valid in both"):
        detect_changes(
            _write_classification(
                tmp_path / "empty_earlier.tif",
                np.zeros((3, 4), dtype=np.uint8),
            ),
            later,
            output,
        )
    with pytest.raises(RasterError, match="CRS"):
        detect_changes(
            _write_classification(
                tmp_path / "no_crs.tif",
                np.full((3, 4), 10, dtype=np.uint8),
                crs=None,
            ),
            later,
            output,
        )


def test_save_transition_matrix_rejects_non_csv(tmp_path: Path) -> None:
    earlier, later = _known_change_rasters(tmp_path)
    result = detect_changes(earlier, later, tmp_path / "change.tif")

    with pytest.raises(DataValidationError, match="Unsupported transition matrix"):
        save_transition_matrix_csv(result.matrix, tmp_path / "transitions.txt")
