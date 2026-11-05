"""Clip raster bands to an AOI while preserving georeferencing and NoData."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.errors import RasterioIOError
from rasterio.mask import mask as raster_mask

from src.data.data_metadata import RasterMetadata
from src.data.dataset_validator import validate_raster_vector_compatibility
from src.data.raster_loader import open_raster, read_metadata, valid_data_mask, validate_raster_path
from src.data.vector_loader import load_aoi, validate_geometries
from src.utils.errors import DataValidationError, RasterError
from src.utils.file_utils import ensure_directory
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_OUTPUT_SUFFIXES: frozenset[str] = frozenset({".tif", ".tiff"})


@dataclass(frozen=True)
class ClipResult:
    """Result of clipping a raster to an AOI.

    ``data`` retains the source band count and is masked where pixels fall
    outside the AOI or are NoData in the source.
    """

    data: np.ma.MaskedArray
    transform: Affine
    crs: CRS
    nodata: float | None
    source_metadata: RasterMetadata
    output_path: Path | None = None

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
    def valid_mask(self) -> np.ndarray:
        """Return a 2D mask that is ``True`` only where every band is valid."""
        mask = valid_data_mask(self.data, nodata=self.nodata)
        if mask.ndim == 3:
            return mask.all(axis=0)
        return mask

    @property
    def valid_pixel_count(self) -> int:
        return int(np.count_nonzero(self.valid_mask))

    @property
    def total_pixel_count(self) -> int:
        return self.height * self.width

    def filled(self) -> np.ndarray:
        """Return the clipped data with masked pixels replaced by NoData (or 0)."""
        fill_value = self.nodata if self.nodata is not None else 0
        return self.data.filled(fill_value)


def _prepare_aoi(
    aoi: str | Path | gpd.GeoDataFrame,
    metadata: RasterMetadata,
) -> gpd.GeoDataFrame:
    if isinstance(aoi, gpd.GeoDataFrame):
        aligned = validate_raster_vector_compatibility(metadata, validate_geometries(aoi))
    else:
        frame = load_aoi(aoi)
        aligned = validate_raster_vector_compatibility(metadata, frame)

    if aligned.empty:
        raise DataValidationError("AOI contains no geometries after validation")
    return aligned


def _write_clipped(
    output_path: Path,
    data: np.ma.MaskedArray,
    transform: Affine,
    crs: CRS,
    nodata: float | None,
    source_metadata: RasterMetadata,
) -> None:
    if output_path.suffix.lower() not in SUPPORTED_OUTPUT_SUFFIXES:
        raise DataValidationError(
            f"Unsupported output type {output_path.suffix!r}; expected .tif or .tiff"
        )

    ensure_directory(output_path.parent)

    profile = {
        "driver": "GTiff",
        "width": int(data.shape[2]),
        "height": int(data.shape[1]),
        "count": int(data.shape[0]),
        "dtype": source_metadata.dtype,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }
    if nodata is not None:
        profile["nodata"] = nodata

    try:
        with rasterio.open(output_path, "w", **profile) as destination:
            if nodata is not None:
                destination.write(data.filled(nodata))
            else:
                destination.write(data.filled(0))
                valid = ~np.ma.getmaskarray(data)
                if valid.ndim == 3:
                    valid = valid.all(axis=0)
                if not bool(valid.all()):
                    destination.write_mask(valid)
            for index, description in enumerate(source_metadata.descriptions, start=1):
                if description:
                    destination.set_band_description(index, description)
    except (RasterioIOError, OSError, ValueError) as exc:
        raise RasterError(f"Unable to write clipped raster {output_path}: {exc}") from exc


def clip_raster_to_aoi(
    raster_path: str | Path,
    aoi: str | Path | gpd.GeoDataFrame,
    output_path: str | Path | None = None,
    *,
    nodata: float | None = None,
    all_touched: bool = False,
    crop: bool = True,
) -> ClipResult:
    """Clip all bands of a raster to an AOI.

    Args:
        raster_path: Input GeoTIFF.
        aoi: GeoDataFrame or path to a GeoJSON/Shapefile/GeoPackage boundary.
        output_path: Optional GeoTIFF destination.
        nodata: Override the source NoData value. If omitted, the source value
            is preserved. When neither is defined, the output uses an internal
            mask instead of assigning a misleading NoData value.
        all_touched: Include pixels touched by the AOI boundary.
        crop: Crop the output to the AOI bounding box.

    Raises:
        DataValidationError: For invalid paths, missing CRS, non-overlapping AOI,
            or an invalid output path.
        RasterError: If reading, masking, or writing fails.
    """
    raster_candidate = validate_raster_path(raster_path)
    metadata = read_metadata(raster_candidate)
    if metadata.crs is None:
        raise RasterError(f"Raster has no CRS and cannot be clipped spatially: {raster_candidate}")

    aligned_aoi = _prepare_aoi(aoi, metadata)
    shapes = [
        geometry.__geo_interface__
        for geometry in aligned_aoi.geometry
        if geometry is not None and not geometry.is_empty
    ]
    if not shapes:
        raise DataValidationError("AOI contains no non-empty geometries")

    output_candidate: Path | None = None
    if output_path is not None:
        output_candidate = Path(output_path).expanduser().resolve()
        if output_candidate == raster_candidate.resolve():
            raise DataValidationError("Output path must differ from the input raster path")
        if output_candidate.suffix.lower() not in SUPPORTED_OUTPUT_SUFFIXES:
            raise DataValidationError(
                f"Unsupported output type {output_candidate.suffix!r}; expected .tif or .tiff"
            )

    effective_nodata = nodata if nodata is not None else metadata.nodata

    with open_raster(raster_candidate) as source:
        try:
            clipped, clipped_transform = raster_mask(
                source,
                shapes,
                crop=crop,
                all_touched=all_touched,
                filled=False,
            )
        except (ValueError, IndexError, RasterioIOError, OSError) as exc:
            raise RasterError(f"Unable to clip raster {raster_candidate}: {exc}") from exc
        crs = source.crs
        if crs is None:  # pragma: no cover - guarded by metadata check above
            raise RasterError("Raster CRS disappeared while reading")

    if output_candidate is not None:
        _write_clipped(
            output_candidate,
            clipped,
            clipped_transform,
            crs,
            effective_nodata,
            metadata,
        )
        logger.info(
            "Clipped %s to %dx%d pixels and wrote %s",
            raster_candidate.name,
            clipped.shape[2],
            clipped.shape[1],
            output_candidate,
        )

    return ClipResult(
        data=clipped,
        transform=clipped_transform,
        crs=crs,
        nodata=effective_nodata,
        source_metadata=metadata,
        output_path=output_candidate,
    )
