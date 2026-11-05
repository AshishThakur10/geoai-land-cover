"""Vector loading, geometry validation/repair, reprojection, and area helpers."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from pyproj import CRS, Geod
from shapely import make_valid

from src.data.data_metadata import VectorMetadata
from src.utils.errors import VectorError
from src.utils.logging_config import get_logger
from src.utils.validation import require_existing_file, require_supported_suffix

logger = get_logger(__name__)

SUPPORTED_VECTOR_SUFFIXES: frozenset[str] = frozenset({".geojson", ".json", ".shp", ".gpkg"})
WGS84_CRS = "EPSG:4326"


def validate_vector_path(path: str | Path) -> Path:
    """Validate that ``path`` is an existing supported vector file."""
    candidate = require_existing_file(path)
    require_supported_suffix(candidate, SUPPORTED_VECTOR_SUFFIXES)
    return candidate


def load_vector(path: str | Path) -> gpd.GeoDataFrame:
    """Load a GeoJSON, Shapefile, or GeoPackage into a GeoDataFrame.

    Raises:
        DataValidationError: If the path is invalid.
        VectorError: If the file cannot be read or contains no features.
    """
    candidate = validate_vector_path(path)
    try:
        frame = gpd.read_file(candidate)
    except Exception as exc:
        raise VectorError(f"Unable to read vector file {candidate}: {exc}") from exc

    if frame.empty:
        raise VectorError(f"Vector file contains no features: {candidate}")
    return frame


def validate_geometries(
    frame: gpd.GeoDataFrame,
    *,
    repair: bool = True,
    drop_invalid: bool = True,
) -> gpd.GeoDataFrame:
    """Remove null/empty geometries and repair invalid geometries.

    Args:
        frame: Input GeoDataFrame (not modified).
        repair: Attempt ``shapely.make_valid`` on invalid geometries.
        drop_invalid: Drop geometries that remain invalid after repair. When
            ``False``, a :class:`VectorError` is raised instead.

    Raises:
        VectorError: If no usable geometries remain or invalid geometries are
            retained and ``drop_invalid`` is false.
    """
    if not isinstance(frame, gpd.GeoDataFrame):
        raise VectorError("Expected a GeoDataFrame")

    result = frame.copy()
    geometry_name = result.geometry.name

    empty = result.geometry.isna() | result.geometry.is_empty
    if empty.any():
        logger.warning("Dropping %d null or empty geometries", int(empty.sum()))
        result = result.loc[~empty].copy()

    if result.empty:
        raise VectorError("Vector data contains no usable geometries")

    invalid = ~result.geometry.is_valid
    if repair and invalid.any():
        logger.info("Repairing %d invalid geometries", int(invalid.sum()))
        result.loc[invalid, geometry_name] = result.loc[invalid, geometry_name].apply(make_valid)
        invalid = ~result.geometry.is_valid

    if invalid.any():
        if not drop_invalid:
            raise VectorError(f"{int(invalid.sum())} geometries are invalid after repair")
        logger.warning(
            "Dropping %d invalid geometries that could not be repaired",
            int(invalid.sum()),
        )
        result = result.loc[~invalid].copy()

    empty = result.geometry.isna() | result.geometry.is_empty
    if empty.any():
        result = result.loc[~empty].copy()

    if result.empty:
        raise VectorError("Vector data contains no usable geometries")

    return result.reset_index(drop=True)


def reproject_vector(frame: gpd.GeoDataFrame, target_crs: str | CRS) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame to ``target_crs``.

    Raises:
        VectorError: If the source CRS is missing or the target CRS is invalid.
    """
    if frame.crs is None:
        raise VectorError("Cannot reproject vector data without a source CRS")

    try:
        target = CRS.from_user_input(target_crs)
    except Exception as exc:
        raise VectorError(f"Invalid target CRS {target_crs!r}: {exc}") from exc

    source = CRS.from_user_input(frame.crs)
    if source == target:
        return frame.copy()

    try:
        return frame.to_crs(target)
    except Exception as exc:
        raise VectorError(f"Unable to reproject vector data to {target}: {exc}") from exc


def load_aoi(
    path: str | Path,
    *,
    target_crs: str | CRS | None = None,
    repair: bool = True,
) -> gpd.GeoDataFrame:
    """Load and validate an AOI, optionally reprojecting it to ``target_crs``."""
    frame = load_vector(path)
    frame = validate_geometries(frame, repair=repair)
    if target_crs is not None:
        frame = reproject_vector(frame, target_crs)
    return frame


def read_vector_metadata(path: str | Path) -> VectorMetadata:
    """Read vector metadata without keeping a file handle open."""
    candidate = validate_vector_path(path)
    frame = load_vector(candidate)
    driver = None
    try:
        import pyogrio

        driver = pyogrio.read_info(candidate).get("driver")
    except Exception:  # pragma: no cover - driver lookup is best effort
        logger.debug("Could not determine vector driver for %s", candidate)

    return VectorMetadata(
        path=candidate,
        driver=str(driver) if driver is not None else None,
        feature_count=len(frame),
        crs=frame.crs,
        geometry_types=tuple(sorted(frame.geom_type.dropna().unique().tolist())),
        bounds=vector_bounds(frame),
    )


def geodesic_area_km2(frame: gpd.GeoDataFrame) -> float:
    """Compute the total absolute geodesic area of all features in square kilometres.

    The calculation uses the WGS84 ellipsoid and is therefore independent of the
    source CRS units. It is suitable for approximate AOI reporting, not for
    cadastral or legally binding measurements.

    Raises:
        VectorError: If the layer has no CRS.
    """
    if frame.crs is None:
        raise VectorError("Cannot calculate area for a layer without a CRS")

    geographic = frame if frame.crs.is_geographic else frame.to_crs(WGS84_CRS)
    geod = Geod(ellps="WGS84")
    total_area = 0.0
    for geometry in geographic.geometry:
        if geometry is None or geometry.is_empty:
            continue
        area, _ = geod.geometry_area_perimeter(geometry)
        total_area += abs(area)
    return total_area / 1_000_000.0


def vector_bounds(frame: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    """Return ``(minx, miny, maxx, maxy)`` for a GeoDataFrame."""
    minx, miny, maxx, maxy = frame.total_bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def require_vector_crs(frame: gpd.GeoDataFrame, *, context: str = "vector data") -> CRS:
    """Return the layer CRS or raise :class:`VectorError`."""
    if frame.crs is None:
        raise VectorError(f"{context} has no CRS")
    return CRS.from_user_input(frame.crs)


__all__ = [
    "SUPPORTED_VECTOR_SUFFIXES",
    "WGS84_CRS",
    "geodesic_area_km2",
    "load_aoi",
    "load_vector",
    "read_vector_metadata",
    "reproject_vector",
    "require_vector_crs",
    "validate_geometries",
    "validate_vector_path",
    "vector_bounds",
]
