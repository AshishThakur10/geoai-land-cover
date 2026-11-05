"""Synthetic geospatial fixtures.

All fake datasets are clearly synthetic arrays written to temporary paths. They
are used only to exercise file I/O, CRS handling, clipping, and visualization;
they are never presented as real satellite observations or model results.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely import Polygon, box
from shapely import transform as shapely_transform

RASTER_CRS = "EPSG:32633"
RASTER_MINX = 500_000.0
RASTER_MAXY = 5_000_000.0
RASTER_RESOLUTION = 10.0
BAND_NAMES = ("Blue", "Green", "Red", "NIR")


def default_transform() -> Affine:
    return from_origin(RASTER_MINX, RASTER_MAXY, RASTER_RESOLUTION, RASTER_RESOLUTION)


def write_synthetic_raster(
    path: Path,
    *,
    width: int = 40,
    height: int = 40,
    count: int = 4,
    crs: str | None = RASTER_CRS,
    transform: Affine | None = None,
    nodata: float | None = 0,
    dtype: str = "uint16",
) -> Path:
    """Write a deterministic synthetic GeoTIFF and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raster_transform = transform or default_transform()
    base = np.arange(1, width * height + 1, dtype=np.float64).reshape(height, width)
    bands = np.stack([base + index * 100 for index in range(count)]).astype(dtype)

    if nodata is not None:
        bands[:, 0, 0] = nodata
        bands[:, -1, -1] = nodata

    descriptions = (
        BAND_NAMES
        if count == len(BAND_NAMES)
        else tuple(f"Band {index + 1}" for index in range(count))
    )

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=raster_transform,
        nodata=nodata,
    ) as dataset:
        dataset.write(bands)
        for index, description in enumerate(descriptions, start=1):
            dataset.set_band_description(index, description)

    return path


def synthetic_aoi_polygon(
    *,
    offset: float = 50.0,
    width: float = 100.0,
    height: float = 100.0,
) -> Polygon:
    """Return a rectangular AOI inside the default synthetic raster."""
    minx = RASTER_MINX + offset
    maxy = RASTER_MAXY - offset
    return box(minx, maxy - height, minx + width, maxy)


def write_synthetic_aoi_geojson(
    path: Path,
    *,
    offset: float = 50.0,
    width: float = 100.0,
    height: float = 100.0,
    crs: str = RASTER_CRS,
    name: str = "synthetic-aoi",
) -> Path:
    """Write a synthetic rectangular AOI as GeoJSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    polygon = synthetic_aoi_polygon(offset=offset, width=width, height=height)
    frame = gpd.GeoDataFrame({"name": [name]}, geometry=[polygon], crs=crs)
    frame.to_file(path, driver="GeoJSON")
    return path


def write_synthetic_aoi_wgs84(
    path: Path,
    *,
    offset: float = 50.0,
    width: float = 100.0,
    height: float = 100.0,
) -> Path:
    """Write the synthetic AOI reprojected to EPSG:4326."""
    transformer = Transformer.from_crs(RASTER_CRS, "EPSG:4326", always_xy=True)

    def project(coordinates: np.ndarray) -> np.ndarray:
        longitude, latitude = transformer.transform(coordinates[:, 0], coordinates[:, 1])
        return np.column_stack([longitude, latitude])

    polygon = shapely_transform(
        synthetic_aoi_polygon(offset=offset, width=width, height=height),
        project,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = gpd.GeoDataFrame({"name": ["synthetic-aoi-wgs84"]}, geometry=[polygon], crs="EPSG:4326")
    frame.to_file(path, driver="GeoJSON")
    return path


def write_synthetic_shapefile(
    path: Path,
    *,
    offset: float = 50.0,
    width: float = 100.0,
    height: float = 100.0,
) -> Path:
    """Write the synthetic AOI as an ESRI Shapefile."""
    path.parent.mkdir(parents=True, exist_ok=True)
    polygon = synthetic_aoi_polygon(offset=offset, width=width, height=height)
    frame = gpd.GeoDataFrame({"name": ["synthetic-aoi"]}, geometry=[polygon], crs=RASTER_CRS)
    frame.to_file(path, driver="ESRI Shapefile")
    return path


def synthetic_worldcover_codes(width: int = 40, height: int = 40) -> np.ndarray:
    """Return a deterministic four-class WorldCover-code array.

    The classes are synthetic, not a real land-cover observation.
    """
    codes = np.zeros((height, width), dtype=np.uint8)
    mid_row = height // 2
    mid_col = width // 2
    codes[:mid_row, :mid_col] = 10
    codes[:mid_row, mid_col:] = 40
    codes[mid_row:, :mid_col] = 30
    codes[mid_row:, mid_col:] = 80
    return codes


def write_synthetic_worldcover(
    path: Path,
    *,
    codes: np.ndarray | None = None,
    width: int = 40,
    height: int = 40,
    crs: str | None = RASTER_CRS,
    transform: Affine | None = None,
    nodata: float | None = 0,
    dtype: str = "uint8",
) -> Path:
    """Write a synthetic single-band WorldCover-code GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = (
        synthetic_worldcover_codes(width=width, height=height)
        if codes is None
        else np.asarray(codes)
    )
    if labels.shape != (height, width):
        raise ValueError(
            f"codes shape {labels.shape} does not match requested {height}x{width}"
        )

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform or default_transform(),
        nodata=nodata,
    ) as dataset:
        dataset.write(labels.astype(dtype), 1)
        dataset.set_band_description(1, "Synthetic WorldCover labels")

    return path
