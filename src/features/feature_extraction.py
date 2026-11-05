"""Assemble Sentinel-2 bands and spectral indices into a model feature stack."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from affine import Affine
from rasterio.crs import CRS

from config.sentinel2_bands import (
    CORE_FEATURE_BANDS,
    OPTIONAL_FEATURE_BANDS,
    core_bands_present,
    resolve_band_key,
)
from src.features.spectral_indices import compute_spectral_indices
from src.preprocessing.image_stacking import RasterStack, save_stack
from src.preprocessing.raster_alignment import AlignedRaster
from src.utils.errors import DataValidationError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

BAND_FEATURE_ORDER: tuple[str, ...] = CORE_FEATURE_BANDS + OPTIONAL_FEATURE_BANDS


@dataclass(frozen=True)
class FeatureStack:
    """A feature cube with one shared 2D validity mask.

    ``data`` has shape ``(n_features, height, width)`` and ``float32`` dtype.
    Invalid pixels are stored as ``NaN``; ``valid_mask`` is ``True`` only where
    every feature is finite.
    """

    data: np.ndarray
    feature_names: tuple[str, ...]
    transform: Affine
    crs: CRS
    valid_mask: np.ndarray
    band_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise DataValidationError(f"Feature data must be 3D, got shape {self.data.shape}")
        if self.data.shape[0] != len(self.feature_names):
            raise DataValidationError(
                f"Feature data has {self.data.shape[0]} layers but "
                f"{len(self.feature_names)} names"
            )
        if not self.feature_names:
            raise DataValidationError("Feature stack must contain at least one feature")
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

    @property
    def total_pixel_count(self) -> int:
        return self.height * self.width

    def to_masked(self) -> np.ma.MaskedArray:
        """Return features with all pixels that are invalid anywhere masked."""
        mask = np.broadcast_to(~self.valid_mask, self.data.shape).copy()
        return np.ma.array(self.data, mask=mask)

    def as_sample_matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(X, rows, cols)`` for spatially valid pixels only.

        ``X`` has shape ``(n_valid_pixels, n_features)`` and is safe to pass to
        scikit-learn estimators.
        """
        rows, cols = np.nonzero(self.valid_mask)
        features = self.data[:, rows, cols].T.astype(np.float32, copy=False)
        return features, rows, cols

    def statistics(self) -> dict[str, dict[str, float | int]]:
        """Return count/min/max/mean per feature over valid pixels."""
        stats: dict[str, dict[str, float | int]] = {}
        for index, name in enumerate(self.feature_names):
            values = self.data[index][self.valid_mask]
            values = values[np.isfinite(values)]
            if values.size == 0:
                stats[name] = {"count": 0, "min": np.nan, "max": np.nan, "mean": np.nan}
                continue
            stats[name] = {
                "count": int(values.size),
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
            }
        return stats

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "band_names": list(self.band_names),
            "count": self.count,
            "width": self.width,
            "height": self.height,
            "crs": self.crs.to_string(),
            "transform": list(self.transform)[:6],
            "valid_pixels": self.valid_pixel_count,
            "total_pixels": self.total_pixel_count,
        }


def _canonicalize_bands(
    bands: Mapping[str, AlignedRaster],
) -> dict[str, AlignedRaster]:
    canonical: dict[str, AlignedRaster] = {}
    for name, raster in bands.items():
        key = resolve_band_key(name)
        if key in canonical:
            raise DataValidationError(f"Band {key} was supplied more than once")
        canonical[key] = raster
    return canonical


def _check_shared_grid(bands: Mapping[str, AlignedRaster]) -> None:
    first_name, first = next(iter(bands.items()))
    for name, raster in bands.items():
        if raster.grid != first.grid:
            raise DataValidationError(
                f"Band {name!r} is not on the same grid as {first_name!r}"
            )


def build_feature_stack(
    bands: Mapping[str, AlignedRaster],
    *,
    include_indices: bool = True,
    include_ndbi: bool | None = None,
) -> FeatureStack:
    """Build a feature stack from aligned Sentinel-2 bands.

    Input names may be canonical keys (``B04``) or aliases (``red``, ``b8``).
    Core bands B02/B03/B04/B08 are required. B11/B12 are included as features
    when present. NDVI and NDWI are always computed when ``include_indices`` is
    true; NDBI is included automatically when B11 is available.

    Raises:
        DataValidationError: For unknown or duplicate bands, missing core
            bands, mismatched grids, or an unavailable requested NDBI.
    """
    canonical = _canonicalize_bands(bands)
    if not core_bands_present(set(canonical)):
        missing = [key for key in CORE_FEATURE_BANDS if key not in canonical]
        raise DataValidationError(f"Missing required Sentinel-2 core bands: {missing}")

    _check_shared_grid(canonical)

    ordered_bands = tuple(key for key in BAND_FEATURE_ORDER if key in canonical)
    layers = [
        np.ma.filled(
            np.ma.asarray(canonical[key].data, dtype=np.float32),
            np.nan,
        )
        for key in ordered_bands
    ]
    feature_names: list[str] = list(ordered_bands)

    if include_indices:
        indices = compute_spectral_indices(
            {key: canonical[key].data for key in ordered_bands},
            include_ndbi=include_ndbi,
        )
        for name, values in indices.items():
            layers.append(np.ma.filled(np.ma.asarray(values, dtype=np.float32), np.nan))
            feature_names.append(name)

    data = np.stack(layers).astype(np.float32, copy=False)
    valid_mask = np.all(np.isfinite(data), axis=0)
    first = canonical[ordered_bands[0]]

    logger.info(
        "Built %d-feature stack from %d bands (%d valid pixels)",
        data.shape[0],
        len(ordered_bands),
        int(np.count_nonzero(valid_mask)),
    )
    return FeatureStack(
        data=data,
        feature_names=tuple(feature_names),
        transform=first.transform,
        crs=first.crs,
        valid_mask=valid_mask,
        band_names=ordered_bands,
    )


def save_feature_stack(
    feature_stack: FeatureStack,
    output_path: str | Path,
    *,
    feature_names: Sequence[str] | None = None,
) -> Path:
    """Write a feature stack to a multi-band Float32 GeoTIFF."""
    descriptions = (
        tuple(feature_names) if feature_names is not None else feature_stack.feature_names
    )
    raster_stack = RasterStack(
        data=feature_stack.data,
        band_names=descriptions,
        transform=feature_stack.transform,
        crs=feature_stack.crs,
        nodata=np.nan,
        valid_mask=feature_stack.valid_mask,
    )
    return save_stack(raster_stack, output_path, band_descriptions=descriptions)
