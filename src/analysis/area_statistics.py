"""Area statistics for classified land-cover rasters."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from affine import Affine
from rasterio.crs import CRS
from rasterio.windows import Window

from config.land_cover_classes import get_class_color, get_class_name, is_valid_class_code
from src.analysis.area_utils import (
    AreaPlan,
    resolve_area_plan,
    valid_values,
    validate_class_nodata,
)
from src.data.raster_loader import ensure_georeferenced, open_raster, read_metadata
from src.utils.errors import DataValidationError, RasterError
from src.utils.file_utils import ensure_directory
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_CSV_SUFFIXES: frozenset[str] = frozenset({".csv"})


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
    plan: AreaPlan = resolve_area_plan(metadata)
    area_method = plan.area_method
    pixel_area_km2 = plan.pixel_area_km2
    counts: dict[int, int] = {}
    areas: dict[int, float] = {}
    unknown: dict[int, int] = {}

    with open_raster(classification_path) as source:
        resolved_nodata = validate_class_nodata(
            source.nodata if nodata is None else nodata
        )

        for row_off in range(0, metadata.height, size):
            for col_off in range(0, metadata.width, size):
                height = min(size, metadata.height - row_off)
                width = min(size, metadata.width - col_off)
                window = Window(col_off, row_off, width, height)
                values = source.read(band, window=window)
                valid = valid_values(values, resolved_nodata)

                if area_method == "geodesic":
                    for local_row in range(height):
                        row_values = values[local_row][valid[local_row]]
                        if row_values.size == 0:
                            continue
                        row_area = plan.row_pixel_area_km2(
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
