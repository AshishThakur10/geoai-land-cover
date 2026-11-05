"""Validation of raster/vector compatibility and CRS alignment."""

from __future__ import annotations

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import box

from src.data.data_metadata import RasterMetadata
from src.data.vector_loader import reproject_vector, validate_geometries
from src.utils.errors import DataValidationError


def align_aoi_to_raster(
    aoi: gpd.GeoDataFrame,
    raster_metadata: RasterMetadata,
    *,
    repair: bool = True,
) -> gpd.GeoDataFrame:
    """Reproject an AOI to the raster CRS and validate its geometries.

    Raises:
        DataValidationError: If either dataset lacks a CRS or the AOI cannot be
            represented in the raster CRS.
    """
    if raster_metadata.crs is None:
        raise DataValidationError(f"Raster has no CRS: {raster_metadata.path}")
    if aoi.crs is None:
        raise DataValidationError("AOI has no CRS; provide a georeferenced boundary")

    try:
        aligned = reproject_vector(aoi, raster_metadata.crs)
    except Exception as exc:
        raise DataValidationError(
            f"Unable to reproject AOI to the raster CRS {raster_metadata.crs}: {exc}"
        ) from exc

    return validate_geometries(aligned, repair=repair)


def aoi_intersects_raster(aoi: gpd.GeoDataFrame, raster_metadata: RasterMetadata) -> bool:
    """Return ``True`` if any AOI geometry intersects the raster bounds."""
    if raster_metadata.crs is None or aoi.crs is None:
        return False
    aligned = reproject_vector(aoi, CRS.from_user_input(raster_metadata.crs))
    raster_bounds = box(
        raster_metadata.bounds.left,
        raster_metadata.bounds.bottom,
        raster_metadata.bounds.right,
        raster_metadata.bounds.top,
    )
    return bool(aligned.geometry.intersects(raster_bounds).any())


def validate_raster_vector_compatibility(
    raster_metadata: RasterMetadata,
    aoi: gpd.GeoDataFrame,
    *,
    repair: bool = True,
) -> gpd.GeoDataFrame:
    """Validate that an AOI overlaps a raster and return it in the raster CRS.

    Raises:
        DataValidationError: If the raster or AOI is not georeferenced, the AOI
            has no usable geometries, or the AOI does not intersect the raster.
    """
    aligned = align_aoi_to_raster(aoi, raster_metadata, repair=repair)

    raster_bounds = box(
        raster_metadata.bounds.left,
        raster_metadata.bounds.bottom,
        raster_metadata.bounds.right,
        raster_metadata.bounds.top,
    )
    if not bool(aligned.geometry.intersects(raster_bounds).any()):
        raise DataValidationError(
            f"AOI does not overlap the raster extent {tuple(raster_metadata.bounds)}"
        )

    return aligned
