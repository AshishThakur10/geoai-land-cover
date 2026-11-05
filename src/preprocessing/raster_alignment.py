"""Align and resample rasters to a shared target grid.

Bands are converted to ``float32`` during alignment so that masked/NoData
pixels can be represented as ``NaN`` without colliding with valid integer
values. Georeferencing (CRS, transform, dimensions) is always taken from the
target grid.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.transform import from_origin
from rasterio.warp import reproject

from src.data.data_metadata import RasterMetadata
from src.data.raster_loader import open_raster, read_metadata, validate_raster_path
from src.utils.errors import DataValidationError, RasterError
from src.utils.file_utils import ensure_directory
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_OUTPUT_SUFFIXES: frozenset[str] = frozenset({".tif", ".tiff"})
SUPPORTED_RESAMPLING: frozenset[str] = frozenset(
    {
        "nearest",
        "bilinear",
        "cubic",
        "cubic_spline",
        "lanczos",
        "average",
        "mode",
        "max",
        "min",
        "med",
        "q1",
        "q3",
        "sum",
        "rms",
    }
)


def resolve_resampling(value: str | Resampling) -> Resampling:
    """Resolve a resampling name or enum to :class:`rasterio.enums.Resampling`."""
    if isinstance(value, Resampling):
        return value
    name = str(value).strip().lower()
    if name not in SUPPORTED_RESAMPLING:
        allowed = ", ".join(sorted(SUPPORTED_RESAMPLING))
        raise DataValidationError(f"Unsupported resampling method {value!r}; expected {allowed}")
    return Resampling[name]


@dataclass(frozen=True)
class TargetGrid:
    """A common raster grid (CRS, transform, width, height)."""

    crs: CRS
    transform: Affine
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise DataValidationError(
                f"Target grid must be positive, got {self.width}x{self.height}"
            )

    @property
    def bounds(self) -> BoundingBox:
        left, top = self.transform.c, self.transform.f
        right = left + self.transform.a * self.width
        bottom = top + self.transform.e * self.height
        return BoundingBox(left, bottom, right, top)

    @property
    def resolution(self) -> tuple[float, float]:
        return abs(self.transform.a), abs(self.transform.e)

    @classmethod
    def from_raster(cls, path: str | Path) -> TargetGrid:
        """Build a grid from an existing georeferenced raster."""
        metadata = read_metadata(path)
        if metadata.crs is None:
            raise RasterError(f"Raster has no CRS and cannot define a target grid: {metadata.path}")
        return cls(
            crs=metadata.crs,
            transform=metadata.transform,
            width=metadata.width,
            height=metadata.height,
        )

    @classmethod
    def from_bounds(
        cls,
        bounds: tuple[float, float, float, float] | BoundingBox,
        crs: str | CRS,
        resolution: float,
    ) -> TargetGrid:
        """Build a grid covering ``bounds``, snapped outward to ``resolution``.

        ``resolution`` is expressed in the units of ``crs`` (metres for a
        projected CRS). For geographic CRS values the units are degrees and the
        caller is responsible for choosing a meaningful value.
        """
        if resolution <= 0 or not np.isfinite(resolution):
            raise DataValidationError(f"Resolution must be positive and finite, got {resolution}")

        west, south, east, north = (float(value) for value in bounds)
        if not all(np.isfinite(value) for value in (west, south, east, north)):
            raise DataValidationError("Bounds must contain finite values")
        if east <= west or north <= south:
            raise DataValidationError(f"Invalid bounds: {(west, south, east, north)}")

        try:
            target_crs = CRS.from_user_input(crs)
        except Exception as exc:
            raise DataValidationError(f"Invalid CRS {crs!r}: {exc}") from exc

        snapped_west = np.floor(west / resolution) * resolution
        snapped_south = np.floor(south / resolution) * resolution
        snapped_east = np.ceil(east / resolution) * resolution
        snapped_north = np.ceil(north / resolution) * resolution

        width = int(round((snapped_east - snapped_west) / resolution))
        height = int(round((snapped_north - snapped_south) / resolution))

        return cls(
            crs=target_crs,
            transform=from_origin(snapped_west, snapped_north, resolution, resolution),
            width=width,
            height=height,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "crs": self.crs.to_string(),
            "transform": list(self.transform)[:6],
            "width": self.width,
            "height": self.height,
            "resolution": list(self.resolution),
            "bounds": list(self.bounds),
        }


@dataclass(frozen=True)
class AlignedRaster:
    """A single band aligned to a :class:`TargetGrid`."""

    name: str
    data: np.ma.MaskedArray
    grid: TargetGrid
    source_metadata: RasterMetadata
    resampling: str

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise DataValidationError(f"Aligned data must be 2D, got shape {self.data.shape}")
        if self.data.shape != (self.grid.height, self.grid.width):
            raise DataValidationError(
                f"Aligned data shape {self.data.shape} does not match grid "
                f"{(self.grid.height, self.grid.width)}"
            )

    @property
    def transform(self) -> Affine:
        return self.grid.transform

    @property
    def crs(self) -> CRS:
        return self.grid.crs

    @property
    def width(self) -> int:
        return self.grid.width

    @property
    def height(self) -> int:
        return self.grid.height

    @property
    def valid_mask(self) -> np.ndarray:
        return np.isfinite(np.ma.getdata(self.data)) & ~np.ma.getmaskarray(self.data)

    @property
    def valid_pixel_count(self) -> int:
        return int(np.count_nonzero(self.valid_mask))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source": str(self.source_metadata.path),
            "resampling": self.resampling,
            "grid": self.grid.to_dict(),
            "valid_pixels": int(np.count_nonzero(self.valid_mask)),
        }


def align_raster_to_grid(
    source_path: str | Path,
    grid: TargetGrid,
    *,
    band_index: int = 1,
    resampling: str | Resampling = "bilinear",
    name: str | None = None,
) -> AlignedRaster:
    """Reproject/resample one band of a raster onto ``grid``.

    Masked and NoData pixels propagate as ``NaN`` in a ``float32`` masked array.

    Raises:
        RasterError: If the source cannot be read or resampled.
        DataValidationError: If the band index or resampling method is invalid.
    """
    candidate = validate_raster_path(source_path)
    metadata = read_metadata(candidate)
    if metadata.crs is None:
        raise RasterError(f"Raster has no CRS and cannot be aligned: {candidate}")
    method = resolve_resampling(resampling)

    with open_raster(candidate) as source:
        if band_index < 1 or band_index > source.count:
            raise DataValidationError(
                f"Band index {band_index} is out of range for {candidate.name} "
                f"({source.count} band(s))"
            )
        source_data = source.read(band_index, masked=True)
        source_transform = source.transform
        source_crs = source.crs

    source_values = np.ma.filled(np.ma.asarray(source_data, dtype=np.float32), np.nan)
    destination = np.full((grid.height, grid.width), np.nan, dtype=np.float32)

    try:
        reproject(
            source=np.asarray(source_values, dtype=np.float32),
            destination=destination,
            src_transform=source_transform,
            src_crs=source_crs,
            src_nodata=np.nan,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=np.nan,
            resampling=method,
        )
    except (RasterioIOError, ValueError, RuntimeError) as exc:
        raise RasterError(f"Unable to align {candidate} to the target grid: {exc}") from exc

    aligned = np.ma.masked_invalid(destination)
    return AlignedRaster(
        name=name or candidate.stem,
        data=aligned,
        grid=grid,
        source_metadata=metadata,
        resampling=method.name,
    )


def align_bands(
    band_paths: Mapping[str, str | Path],
    *,
    reference_band: str | None = None,
    grid: TargetGrid | None = None,
    resampling: str | Resampling = "bilinear",
) -> dict[str, AlignedRaster]:
    """Align a mapping of named band paths to one shared grid.

    If ``grid`` is omitted, it is derived from ``reference_band`` (or the first
    band when no reference is specified).
    """
    if not band_paths:
        raise DataValidationError("At least one band path is required")

    if grid is None:
        selected_reference = reference_band or next(iter(band_paths))
        if selected_reference not in band_paths:
            raise DataValidationError(f"Reference band {selected_reference!r} is not in band_paths")
        grid = TargetGrid.from_raster(band_paths[selected_reference])

    method = resolve_resampling(resampling)
    aligned: dict[str, AlignedRaster] = {}
    for name, path in band_paths.items():
        aligned[name] = align_raster_to_grid(
            path,
            grid,
            resampling=method,
            name=name,
        )
    return aligned


def write_aligned_raster(
    aligned: AlignedRaster,
    output_path: str | Path,
) -> Path:
    """Write an aligned band to a Float32 GeoTIFF with ``NaN`` NoData."""
    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() not in SUPPORTED_OUTPUT_SUFFIXES:
        raise DataValidationError(
            f"Unsupported output type {destination.suffix!r}; expected .tif or .tiff"
        )
    ensure_directory(destination.parent)

    profile = {
        "driver": "GTiff",
        "width": aligned.width,
        "height": aligned.height,
        "count": 1,
        "dtype": "float32",
        "crs": aligned.crs,
        "transform": aligned.transform,
        "nodata": np.nan,
        "compress": "deflate",
    }
    try:
        with rasterio.open(destination, "w", **profile) as dataset:
            dataset.write(aligned.data.filled(np.nan), 1)
            dataset.set_band_description(1, aligned.name)
    except (RasterioIOError, OSError, ValueError) as exc:
        raise RasterError(f"Unable to write aligned raster {destination}: {exc}") from exc

    logger.info("Wrote aligned band %s to %s", aligned.name, destination)
    return destination
