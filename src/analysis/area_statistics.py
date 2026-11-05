"""Area statistics for classified land-cover rasters."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from affine import Affine
from pyproj import Geod
from rasterio.crs import CRS
from rasterio.windows import Window

from config.land_cover_classes import (
    get_class_color,
    get_class_name,
    is_valid_class_code,
)
from src.data.data_metadata import RasterMetadata
from src.data.raster_loader import ensure_georeferenced, open_raster, read_metadata
from src.utils.errors import DataValidationError, RasterError
from src.utils.file_utils import ensure_directory
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_CSV_SUFFIXES: frozenset[str] = frozenset({".csv"})
WGS84_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class ClassArea:
    """Pixel count, area, and percentage for one land-cover class."""

    code: int
    name: str
    color: str
    pixel_count: int
    area_km2: float
    percentage: float

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "name": self.name,
            "color": self.color,
            "pixel_count": self.pixel_count,
            "area_km2": self.area_km2,
            "percentage": self.percentage,
        }


@dataclass(frozen=True)
class AreaStatistics:
    """Area distribution for a classified raster."""

    source_path: Path
    crs: CRS
    transform: Affine
    width: int
    height: int
    nodata: int | float | None
    area_method: str
    pixel_area_km2: float | None
    total_pixels: int
    valid_pixels: int
    nodata_pixels: int
    classes: tuple[ClassArea, ...]

    @property
    def total_area_km2(self) -> float:
        return float(sum(item.area_km2 for item in self.classes))

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": str(self.source_path),
            "crs": self.crs.to_string(),
            "transform": list(self.transform)[:6],
            "width": self.width,
            "height": self.height,
            "nodata": self.nodata,
            "area_method": self.area_method,
            "pixel_area_km2": self.pixel_area_km2,
            "total_pixels": self.total_pixels,
            "valid_pixels": self.valid_pixels,
            "nodata_pixels": self.nodata_pixels,
            "total_area_km2": self.total_area_km2,
            "classes": [item.to_dict() for item in self.classes],
        }


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(f"{name} must be an integer, got {value!r}")
    if value < 1:
        raise DataValidationError(f"{name} must be >= 1, got {value}")
    return value


def _normalise_nodata(value: object) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise DataValidationError(f"nodata must be numeric, got {value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if math.isnan(numeric):
            return numeric
        if not math.isfinite(numeric):
            raise DataValidationError(f"nodata must be finite, got {value!r}")
        return int(numeric) if numeric.is_integer() else numeric
    raise DataValidationError(f"nodata must be numeric, got {value!r}")


def _projected_pixel_area_km2(crs: CRS, transform: Affine) -> float:
    try:
        factor = float(crs.linear_units_factor[1])
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise DataValidationError(f"Unable to determine CRS linear units: {crs}") from exc
    if not math.isfinite(factor) or factor <= 0:
        raise DataValidationError(f"Invalid linear unit conversion factor for CRS {crs}")
    determinant = abs(transform.a * transform.e - transform.b * transform.d)
    pixel_area_m2 = determinant * factor * factor
    if not math.isfinite(pixel_area_m2) or pixel_area_m2 <= 0:
        raise DataValidationError("Raster transform does not define a positive pixel area")
    return pixel_area_m2 / 1_000_000.0


def _geodesic_row_area_km2(transform: Affine, row: int, width: int) -> float:
    half_width = abs(transform.a) / 2.0
    half_height = abs(transform.e) / 2.0
    center_x = transform.c + (width / 2.0) * transform.a
    center_y = transform.f + (row + 0.5) * transform.e
    lons = (
        center_x - half_width,
        center_x + half_width,
        center_x + half_width,
        center_x - half_width,
    )
    lats = (
        center_y - half_height,
        center_y - half_height,
        center_y + half_height,
        center_y + half_height,
    )
    area, _ = WGS84_GEOD.polygon_area_perimeter(lons, lats)
    return abs(area) / 1_000_000.0


def _area_plan(
    metadata: RasterMetadata,
    *,
    width: int,
    height: int,
) -> tuple[str, float | None]:
    crs = metadata.crs
    transform = metadata.transform
    if crs is None:
        raise RasterError("Classification raster has no CRS")
    if crs.is_projected:
        return "projected", _projected_pixel_area_km2(crs, transform)
    if crs.is_geographic:
        if not np.isclose(transform.b, 0.0) or not np.isclose(transform.d, 0.0):
            raise DataValidationError(
                "Rotated geographic grids are not supported for area statistics; "
                "reproject to a projected CRS"
            )
        if width <= 0 or height <= 0:
            raise DataValidationError("Classification raster dimensions must be positive")
        return "geodesic", None
    raise DataValidationError(
        f"Unsupported CRS for area statistics: {crs.to_string()}; "
        "use a projected or geographic CRS"
    )


def _valid_values(values: np.ndarray, nodata: int | float | None) -> np.ndarray:
    if nodata is None:
        return np.ones(values.shape, dtype=bool)
    if isinstance(nodata, float) and math.isnan(nodata):
        return np.isfinite(values)
    return values != nodata


def compute_area_statistics(
    classification_path: str | Path,
    *,
    band_index: int = 1,
    nodata: int | float | None = None,
    window_size: int = 1024,
) -> AreaStatistics:
    """Compute per-class pixel counts, areas, and percentages window by window.

    Areas use projected pixel dimensions when the CRS is projected. Geographic
    rasters use WGS84 geodesic cell areas, never raw degrees.
    """
    band = _positive_integer(band_index, "band_index")
    size = _positive_integer(window_size, "window_size")
    metadata = read_metadata(classification_path)
    ensure_georeferenced(metadata)
    if band > metadata.count:
        raise DataValidationError(
            f"Band index {band} is out of range for {metadata.path.name} "
            f"({metadata.count} band(s))"
        )

    selected_dtype = np.dtype(metadata.dtypes[band - 1])
    if not np.issubdtype(selected_dtype, np.integer):
        raise DataValidationError(
            f"Classification band must use an integer dtype, got {selected_dtype}"
        )

    crs = metadata.crs
    assert crs is not None
    area_method, pixel_area_km2 = _area_plan(
        metadata,
        width=metadata.width,
        height=metadata.height,
    )
    counts: dict[int, int] = {}
    areas: dict[int, float] = {}
    unknown: dict[int, int] = {}

    with open_raster(classification_path) as source:
        resolved_nodata = _normalise_nodata(source.nodata if nodata is None else nodata)
        if (
            isinstance(resolved_nodata, int)
            and is_valid_class_code(resolved_nodata)
        ):
            raise DataValidationError(
                f"NoData value {resolved_nodata} overlaps a land-cover class code"
            )

        for row_off in range(0, metadata.height, size):
            for col_off in range(0, metadata.width, size):
                height = min(size, metadata.height - row_off)
                width = min(size, metadata.width - col_off)
                window = Window(col_off, row_off, width, height)
                values = source.read(band, window=window)
                valid = _valid_values(values, resolved_nodata)

                if area_method == "geodesic":
                    for local_row in range(height):
                        row_values = values[local_row][valid[local_row]]
                        if row_values.size == 0:
                            continue
                        row_area = _geodesic_row_area_km2(
                            metadata.transform,
                            row_off + local_row,
                            metadata.width,
                        )
                        unique, row_counts = np.unique(row_values, return_counts=True)
                        for code, count in zip(unique, row_counts, strict=True):
                            code_int = int(code)
                            if not is_valid_class_code(code_int):
                                unknown[code_int] = unknown.get(code_int, 0) + int(count)
                                continue
                            counts[code_int] = counts.get(code_int, 0) + int(count)
                            areas[code_int] = areas.get(code_int, 0.0) + (
                                float(count) * row_area
                            )
                else:
                    selected = values[valid]
                    if selected.size:
                        unique, window_counts = np.unique(selected, return_counts=True)
                        for code, count in zip(unique, window_counts, strict=True):
                            code_int = int(code)
                            if not is_valid_class_code(code_int):
                                unknown[code_int] = unknown.get(code_int, 0) + int(count)
                                continue
                            counts[code_int] = counts.get(code_int, 0) + int(count)

    if unknown:
        details = ", ".join(
            f"{code}: {count}" for code, count in sorted(unknown.items())
        )
        raise DataValidationError(f"Classification contains unknown class codes: {details}")

    valid_pixels = sum(counts.values())
    if valid_pixels == 0:
        raise DataValidationError(
            "Classification raster contains no valid land-cover class pixels"
        )

    if area_method == "projected":
        assert pixel_area_km2 is not None
        areas = {code: count * pixel_area_km2 for code, count in counts.items()}

    class_areas = tuple(
        ClassArea(
            code=code,
            name=get_class_name(code),
            color=get_class_color(code),
            pixel_count=count,
            area_km2=float(areas[code]),
            percentage=(count / valid_pixels) * 100.0,
        )
        for code, count in sorted(counts.items())
    )
    total_pixels = metadata.width * metadata.height
    statistics = AreaStatistics(
        source_path=metadata.path,
        crs=crs,
        transform=metadata.transform,
        width=metadata.width,
        height=metadata.height,
        nodata=resolved_nodata,
        area_method=area_method,
        pixel_area_km2=pixel_area_km2,
        total_pixels=total_pixels,
        valid_pixels=valid_pixels,
        nodata_pixels=total_pixels - valid_pixels,
        classes=class_areas,
    )
    logger.info(
        "Computed %s area statistics for %d classes in %s",
        area_method,
        len(class_areas),
        metadata.path,
    )
    return statistics


def save_area_statistics_csv(
    statistics: AreaStatistics,
    output_path: str | Path,
) -> Path:
    """Write area statistics to a CSV table."""
    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() not in SUPPORTED_CSV_SUFFIXES:
        raise DataValidationError(
            f"Unsupported statistics type {destination.suffix!r}; expected .csv"
        )
    ensure_directory(destination.parent)

    try:
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["code", "land_cover_type", "pixel_count", "area_km2", "percentage"]
            )
            for item in statistics.classes:
                writer.writerow(
                    [
                        item.code,
                        item.name,
                        item.pixel_count,
                        f"{item.area_km2:.6f}",
                        f"{item.percentage:.6f}",
                    ]
                )
    except OSError as exc:
        raise RasterError(f"Unable to write area statistics {destination}: {exc}") from exc

    logger.info("Wrote area statistics to %s", destination)
    return destination
