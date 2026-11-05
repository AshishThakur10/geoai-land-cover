"""Typed metadata records for raster and vector datasets.

These records decouple downstream processing from open file handles and provide
a consistent, serialisable description of geospatial inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.crs import CRS


@dataclass(frozen=True)
class RasterMetadata:
    """Immutable metadata for a raster dataset."""

    path: Path
    driver: str
    width: int
    height: int
    count: int
    dtypes: tuple[str, ...]
    crs: CRS | None
    transform: Affine
    bounds: BoundingBox
    nodata: float | None
    descriptions: tuple[str | None, ...]
    resolutions: tuple[float, float]

    @property
    def crs_wkt(self) -> str | None:
        """Return the CRS as WKT, or ``None`` when the raster is not georeferenced."""
        return self.crs.to_wkt() if self.crs is not None else None

    @property
    def is_georeferenced(self) -> bool:
        """Return ``True`` when the raster has a CRS and a non-identity transform."""
        return self.crs is not None and self.transform != Affine.identity()

    @property
    def dtype(self) -> str:
        """Return the dtype of the first band (common case for multispectral stacks)."""
        return self.dtypes[0]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable metadata dictionary."""
        return {
            "path": str(self.path),
            "driver": self.driver,
            "width": self.width,
            "height": self.height,
            "count": self.count,
            "dtypes": list(self.dtypes),
            "crs": self.crs.to_string() if self.crs is not None else None,
            "crs_wkt": self.crs_wkt,
            "transform": list(self.transform)[:6],
            "bounds": [self.bounds.left, self.bounds.bottom, self.bounds.right, self.bounds.top],
            "nodata": self.nodata,
            "descriptions": list(self.descriptions),
            "resolutions": list(self.resolutions),
            "is_georeferenced": self.is_georeferenced,
        }


@dataclass(frozen=True)
class VectorMetadata:
    """Immutable metadata for a vector dataset."""

    path: Path
    driver: str | None
    feature_count: int
    crs: CRS | None
    geometry_types: tuple[str, ...]
    bounds: tuple[float, float, float, float] | None

    @property
    def crs_wkt(self) -> str | None:
        """Return the CRS as WKT, or ``None`` when the layer has no CRS."""
        return self.crs.to_wkt() if self.crs is not None else None

    @property
    def is_georeferenced(self) -> bool:
        """Return ``True`` when the vector layer declares a CRS."""
        return self.crs is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable metadata dictionary."""
        return {
            "path": str(self.path),
            "driver": self.driver,
            "feature_count": self.feature_count,
            "crs": self.crs.to_string() if self.crs is not None else None,
            "crs_wkt": self.crs_wkt,
            "geometry_types": list(self.geometry_types),
            "bounds": list(self.bounds) if self.bounds is not None else None,
            "is_georeferenced": self.is_georeferenced,
        }
