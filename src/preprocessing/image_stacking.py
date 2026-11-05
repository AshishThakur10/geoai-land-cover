"""Build and export multispectral band stacks on a shared grid."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError

from src.preprocessing.raster_alignment import AlignedRaster, align_bands
from src.utils.errors import DataValidationError, RasterError
from src.utils.file_utils import ensure_directory
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_OUTPUT_SUFFIXES: frozenset[str] = frozenset({".tif", ".tiff"})


@dataclass(frozen=True)
class RasterStack:
    """A multi-band ``float32`` stack with ``NaN`` NoData and a shared 2D mask."""

    data: np.ndarray
    band_names: tuple[str, ...]
    transform: Affine
    crs: CRS
    nodata: float
    valid_mask: np.ndarray

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise DataValidationError(f"Stack data must be 3D, got shape {self.data.shape}")
        if self.data.shape[0] != len(self.band_names):
            raise DataValidationError(
                f"Stack has {self.data.shape[0]} arrays but {len(self.band_names)} band names"
            )
        if self.data.shape[0] == 0:
            raise DataValidationError("Stack must contain at least one band")
        if self.valid_mask.shape != self.data.shape[1:]:
            raise DataValidationError(
                f"Valid mask shape {self.valid_mask.shape} does not match "
                f"spatial shape {self.data.shape[1:]}"
            )

    @property
    def count(self) -> int:
        return int(self.data.shape[0])

    @property
    def height(self) -> int:
        return int(self.data.shape[1])

    @property
    def width(self) -> int:
        return int(self.data.shape[2])

    @property
    def valid_pixel_count(self) -> int:
        return int(np.count_nonzero(self.valid_mask))

    def to_masked(self) -> np.ma.MaskedArray:
        """Return the stack as a masked array with invalid pixels masked."""
        mask = np.broadcast_to(~self.valid_mask, self.data.shape).copy()
        return np.ma.array(self.data, mask=mask)

    def to_dict(self) -> dict[str, object]:
        return {
            "band_names": list(self.band_names),
            "count": self.count,
            "width": self.width,
            "height": self.height,
            "crs": self.crs.to_string(),
            "transform": list(self.transform)[:6],
            "nodata": self.nodata,
            "valid_pixels": self.valid_pixel_count,
        }


def _check_same_grid(rasters: Mapping[str, AlignedRaster]) -> None:
    first_name, first = next(iter(rasters.items()))
    for name, raster in rasters.items():
        if raster.grid != first.grid:
            raise DataValidationError(
                f"Band {name!r} is not on the same grid as {first_name!r}"
            )


def stack_aligned_rasters(
    rasters: Mapping[str, AlignedRaster],
    *,
    band_order: Sequence[str] | None = None,
) -> RasterStack:
    """Stack aligned bands into a 3D array.

    Args:
        rasters: Mapping of band name to aligned raster.
        band_order: Optional ordering. Defaults to the mapping's insertion order.

    Raises:
        DataValidationError: If the mapping is empty, names are missing or
            duplicated, or the rasters do not share a grid.
    """
    if not rasters:
        raise DataValidationError("At least one aligned raster is required")

    names = tuple(band_order) if band_order is not None else tuple(rasters)
    if not names:
        raise DataValidationError("Band order must not be empty")
    if len(set(names)) != len(names):
        raise DataValidationError("Band order contains duplicate names")
    missing = [name for name in names if name not in rasters]
    if missing:
        raise DataValidationError(f"Band order references unknown bands: {missing}")

    _check_same_grid(rasters)
    first = rasters[names[0]]

    arrays = []
    for name in names:
        raster = rasters[name]
        values = np.ma.filled(np.ma.asarray(raster.data, dtype=np.float32), np.nan)
        arrays.append(np.asarray(values, dtype=np.float32))

    data = np.stack(arrays)
    valid_mask = np.all(np.isfinite(data), axis=0)

    return RasterStack(
        data=data,
        band_names=names,
        transform=first.transform,
        crs=first.crs,
        nodata=np.nan,
        valid_mask=valid_mask,
    )


def stack_band_paths(
    band_paths: Mapping[str, str | Path],
    *,
    reference_band: str | None = None,
    resampling: str | Resampling = "bilinear",
    band_order: Sequence[str] | None = None,
) -> RasterStack:
    """Align band paths to a reference grid and stack them."""
    aligned = align_bands(
        band_paths,
        reference_band=reference_band,
        resampling=resampling,
    )
    return stack_aligned_rasters(aligned, band_order=band_order)


def save_stack(
    stack: RasterStack,
    output_path: str | Path,
    *,
    band_descriptions: Sequence[str] | None = None,
) -> Path:
    """Write a stack to a multi-band Float32 GeoTIFF with ``NaN`` NoData."""
    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() not in SUPPORTED_OUTPUT_SUFFIXES:
        raise DataValidationError(
            f"Unsupported output type {destination.suffix!r}; expected .tif or .tiff"
        )
    ensure_directory(destination.parent)

    descriptions = tuple(band_descriptions) if band_descriptions is not None else stack.band_names
    if len(descriptions) != stack.count:
        raise DataValidationError(
            f"Expected {stack.count} band descriptions, got {len(descriptions)}"
        )

    profile = {
        "driver": "GTiff",
        "width": stack.width,
        "height": stack.height,
        "count": stack.count,
        "dtype": "float32",
        "crs": stack.crs,
        "transform": stack.transform,
        "nodata": np.nan,
        "compress": "deflate",
    }
    try:
        with rasterio.open(destination, "w", **profile) as dataset:
            dataset.write(stack.data.astype(np.float32))
            for index, description in enumerate(descriptions, start=1):
                dataset.set_band_description(index, description)
    except (RasterioIOError, OSError, ValueError) as exc:
        raise RasterError(f"Unable to write raster stack {destination}: {exc}") from exc

    logger.info("Wrote %d-band stack to %s", stack.count, destination)
    return destination
