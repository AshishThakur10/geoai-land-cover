"""Shared area, NoData, and validity calculations for classified rasters."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from affine import Affine
from pyproj import Geod
from rasterio.crs import CRS

from config.land_cover_classes import is_valid_class_code
from src.data.data_metadata import RasterMetadata
from src.utils.errors import DataValidationError, RasterError

WGS84_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class AreaPlan:
    """Area method and per-pixel area for a classified raster grid."""

    area_method: str
    pixel_area_km2: float | None

    def row_pixel_area_km2(
        self,
        transform: Affine,
        row: int,
        width: int,
    ) -> float:
        """Return the area of one pixel in ``row`` of the full raster."""
        if self.area_method == "projected":
            assert self.pixel_area_km2 is not None
            return self.pixel_area_km2
        return geodesic_row_area_km2(transform, row, width)


def normalise_nodata(value: object) -> int | float | None:
    """Normalise a NoData value and reject booleans and non-finite numbers."""
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


def validate_class_nodata(
    value: int | float | None,
    *,
    label: str = "nodata",
) -> int | float | None:
    """Reject NoData values that overlap a recognised land-cover class."""
    resolved = normalise_nodata(value)
    if isinstance(resolved, int) and is_valid_class_code(resolved):
        raise DataValidationError(
            f"{label} value {resolved} overlaps a land-cover class code"
        )
    return resolved


def valid_values(
    values: np.ndarray,
    nodata: int | float | None,
) -> np.ndarray:
    """Return a mask where values are valid for classification analysis."""
    if nodata is None:
        return np.ones(values.shape, dtype=bool)
    if isinstance(nodata, float) and math.isnan(nodata):
        return np.isfinite(values)
    return values != nodata


def projected_pixel_area_km2(crs: CRS, transform: Affine) -> float:
    """Calculate projected pixel area, converting CRS units to square metres."""
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


def geodesic_row_area_km2(transform: Affine, row: int, width: int) -> float:
    """Calculate one row's WGS84 geodesic cell area in square kilometres."""
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


def resolve_area_plan(metadata: RasterMetadata) -> AreaPlan:
    """Choose projected or geodesic area calculations for a raster grid."""
    crs = metadata.crs
    transform = metadata.transform
    if crs is None:
        raise RasterError("Classification raster has no CRS")
    if crs.is_projected:
        return AreaPlan(
            area_method="projected",
            pixel_area_km2=projected_pixel_area_km2(crs, transform),
        )
    if crs.is_geographic:
        if not np.isclose(transform.b, 0.0) or not np.isclose(transform.d, 0.0):
            raise DataValidationError(
                "Rotated geographic grids are not supported for area statistics; "
                "reproject to a projected CRS"
            )
        if metadata.width <= 0 or metadata.height <= 0:
            raise DataValidationError("Classification raster dimensions must be positive")
        return AreaPlan(area_method="geodesic", pixel_area_km2=None)
    raise DataValidationError(
        f"Unsupported CRS for area statistics: {crs.to_string()}; "
        "use a projected or geographic CRS"
    )
