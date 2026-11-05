"""Raster loading, metadata inspection, and NoData helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.io import DatasetReader

from src.data.data_metadata import RasterMetadata
from src.utils.errors import DataValidationError, RasterError
from src.utils.validation import require_existing_file, require_supported_suffix

SUPPORTED_RASTER_SUFFIXES: frozenset[str] = frozenset({".tif", ".tiff"})


def validate_raster_path(path: str | Path) -> Path:
    """Validate that ``path`` is an existing GeoTIFF.

    Raises:
        DataValidationError: If the file is missing or has an unsupported suffix.
    """
    candidate = require_existing_file(path)
    require_supported_suffix(candidate, SUPPORTED_RASTER_SUFFIXES)
    return candidate


@contextmanager
def open_raster(path: str | Path) -> Iterator[DatasetReader]:
    """Open a GeoTIFF for reading and always close it.

    Raises:
        DataValidationError: If the path is invalid.
        RasterError: If GDAL cannot open or read the file.
    """
    candidate = validate_raster_path(path)
    try:
        dataset = rasterio.open(candidate)
    except (RasterioIOError, OSError) as exc:
        raise RasterError(f"Unable to open raster {candidate}: {exc}") from exc

    try:
        yield dataset
    finally:
        dataset.close()


def _metadata_from_dataset(dataset: DatasetReader, path: Path) -> RasterMetadata:
    return RasterMetadata(
        path=path,
        driver=dataset.driver,
        width=dataset.width,
        height=dataset.height,
        count=dataset.count,
        dtypes=tuple(dataset.dtypes),
        crs=dataset.crs,
        transform=dataset.transform,
        bounds=dataset.bounds,
        nodata=dataset.nodata,
        descriptions=tuple(dataset.descriptions),
        resolutions=tuple(dataset.res),
    )


def read_metadata(path: str | Path) -> RasterMetadata:
    """Read raster metadata without loading pixel data into memory.

    Raises:
        RasterError: If required metadata cannot be read or the raster is empty.
    """
    candidate = validate_raster_path(path)
    with open_raster(candidate) as dataset:
        metadata = _metadata_from_dataset(dataset, candidate)

    if metadata.width <= 0 or metadata.height <= 0:
        raise RasterError(f"Raster has invalid dimensions: {metadata.width}x{metadata.height}")
    if metadata.count <= 0:
        raise RasterError("Raster contains no bands")

    return metadata


def read_band(
    path: str | Path,
    band_index: int = 1,
    *,
    masked: bool = True,
) -> np.ndarray | np.ma.MaskedArray:
    """Read a single band from a raster.

    Args:
        path: GeoTIFF path.
        band_index: 1-based band index.
        masked: Return a masked array when a NoData value is defined.

    Raises:
        RasterError: If the band index is out of range or reading fails.
    """
    candidate = validate_raster_path(path)
    with open_raster(candidate) as dataset:
        if band_index < 1 or band_index > dataset.count:
            raise RasterError(
                f"Band index {band_index} is out of range for {candidate.name} "
                f"({dataset.count} band(s))"
            )
        try:
            return dataset.read(band_index, masked=masked)
        except (RasterioIOError, OSError, ValueError) as exc:
            raise RasterError(f"Unable to read band {band_index} from {candidate}: {exc}") from exc


def read_all_bands(
    path: str | Path,
    *,
    masked: bool = True,
) -> np.ndarray | np.ma.MaskedArray:
    """Read every band as an array with shape ``(count, height, width)``."""
    candidate = validate_raster_path(path)
    with open_raster(candidate) as dataset:
        try:
            return dataset.read(masked=masked)
        except (RasterioIOError, OSError, ValueError) as exc:
            raise RasterError(f"Unable to read bands from {candidate}: {exc}") from exc


def valid_data_mask(
    data: np.ndarray | np.ma.MaskedArray,
    nodata: float | None = None,
) -> np.ndarray:
    """Return a boolean mask where ``True`` marks valid pixels.

    Masked values, non-finite floats, and explicit NoData values are invalid.
    """
    values = np.ma.getdata(data)
    invalid = np.ma.getmaskarray(data).astype(bool, copy=True)

    if np.issubdtype(values.dtype, np.floating):
        invalid |= ~np.isfinite(values)

    if nodata is not None:
        if isinstance(nodata, float) and np.isnan(nodata):
            if np.issubdtype(values.dtype, np.floating):
                invalid |= np.isnan(values)
        else:
            invalid |= values == nodata

    return ~invalid


def count_valid_pixels(
    data: np.ndarray | np.ma.MaskedArray,
    nodata: float | None = None,
) -> int:
    """Count pixels that are not masked, non-finite, or equal to NoData."""
    return int(np.count_nonzero(valid_data_mask(data, nodata=nodata)))


def ensure_georeferenced(metadata: RasterMetadata) -> None:
    """Raise :class:`RasterError` when a raster cannot be used spatially."""
    if metadata.crs is None:
        raise RasterError(f"Raster has no CRS: {metadata.path}")
    if not metadata.is_georeferenced:
        raise RasterError(
            f"Raster has an identity transform and cannot be georeferenced: {metadata.path}"
        )


def assert_same_grid(reference: RasterMetadata, other: RasterMetadata) -> None:
    """Validate that two rasters share CRS, transform, and dimensions.

    Raises:
        DataValidationError: If the grids are not aligned.
    """
    if reference.crs != other.crs:
        raise DataValidationError(
            f"CRS mismatch: {reference.path.name} uses {reference.crs}, "
            f"{other.path.name} uses {other.crs}"
        )
    if reference.transform != other.transform:
        raise DataValidationError(
            f"Transform mismatch between {reference.path.name} and {other.path.name}"
        )
    if (reference.width, reference.height) != (other.width, other.height):
        raise DataValidationError(
            f"Dimension mismatch: {reference.width}x{reference.height} vs "
            f"{other.width}x{other.height}"
        )
