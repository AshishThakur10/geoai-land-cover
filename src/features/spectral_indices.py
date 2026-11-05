"""Safe normalized-difference spectral indices.

Every calculation is performed in ``float32`` with NumPy error state suppressed.
Pixels with a zero denominator, non-finite input, masked input, or non-finite
result are masked, so outputs never contain infinities or silent NaNs.

Index conventions used here:

- NDVI = (NIR - Red) / (NIR + Red)
- NDWI = (Green - NIR) / (Green + NIR)   (McFeeters, 1996)
- NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)   (requires B11)
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from config.sentinel2_bands import get_band, resolve_band_key
from src.utils.errors import DataValidationError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

INDEX_NAMES: tuple[str, ...] = ("NDVI", "NDWI", "NDBI")

INDEX_REQUIRED_BANDS: Mapping[str, tuple[str, ...]] = {
    "NDVI": ("B08", "B04"),
    "NDWI": ("B03", "B08"),
    "NDBI": ("B11", "B08"),
}


def _as_masked_float(data: np.ndarray | np.ma.MaskedArray) -> np.ma.MaskedArray:
    values = np.ma.asarray(data, dtype=np.float32)
    raw = np.ma.getdata(values)
    mask = np.ma.getmaskarray(values).astype(bool, copy=True)
    mask |= ~np.isfinite(raw)
    return np.ma.array(raw, mask=mask)


def _check_same_shape(
    first: np.ndarray | np.ma.MaskedArray,
    second: np.ndarray | np.ma.MaskedArray,
) -> None:
    if np.shape(first) != np.shape(second):
        raise DataValidationError(
            f"Index inputs must share a shape, got {np.shape(first)} and {np.shape(second)}"
        )


def normalized_difference(
    positive: np.ndarray | np.ma.MaskedArray,
    negative: np.ndarray | np.ma.MaskedArray,
    *,
    clip: bool = True,
) -> np.ma.MaskedArray:
    """Compute ``(positive - negative) / (positive + negative)`` safely.

    Zero denominators and non-finite values are masked. With ``clip=True``
    (the default), valid results are clipped to ``[-1, 1]`` to absorb tiny
    floating-point excursions.
    """
    _check_same_shape(positive, negative)
    first = _as_masked_float(positive)
    second = _as_masked_float(negative)

    if first.ndim == 0:
        raise DataValidationError("Index inputs must be at least one-dimensional")

    numerator = first - second
    denominator = first + second
    invalid = np.ma.getmaskarray(first) | np.ma.getmaskarray(second)
    invalid |= ~np.isfinite(denominator) | (denominator == 0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result_values = np.ma.getdata(numerator) / np.ma.getdata(denominator)
    invalid |= ~np.isfinite(result_values)
    result = np.ma.array(result_values, mask=invalid, dtype=np.float32)

    if clip:
        result = np.ma.clip(result, -1.0, 1.0)
    result.set_fill_value(np.nan)
    return result


def compute_ndvi(
    nir: np.ndarray | np.ma.MaskedArray,
    red: np.ndarray | np.ma.MaskedArray,
) -> np.ma.MaskedArray:
    """Compute the Normalized Difference Vegetation Index."""
    return normalized_difference(nir, red)


def compute_ndwi(
    green: np.ndarray | np.ma.MaskedArray,
    nir: np.ndarray | np.ma.MaskedArray,
) -> np.ma.MaskedArray:
    """Compute the McFeeters Normalized Difference Water Index."""
    return normalized_difference(green, nir)


def compute_ndbi(
    swir1: np.ndarray | np.ma.MaskedArray,
    nir: np.ndarray | np.ma.MaskedArray,
) -> np.ma.MaskedArray:
    """Compute the Normalized Difference Built-up Index (requires SWIR1)."""
    return normalized_difference(swir1, nir)


def compute_index(
    name: str,
    bands: Mapping[str, np.ndarray | np.ma.MaskedArray],
) -> np.ma.MaskedArray:
    """Compute one named index from a mapping of band keys/aliases.

    Raises:
        DataValidationError: If the index or a required band is unavailable.
    """
    index_name = str(name).strip().upper()
    if index_name not in INDEX_REQUIRED_BANDS:
        allowed = ", ".join(INDEX_NAMES)
        raise DataValidationError(f"Unknown spectral index {name!r}; expected one of {allowed}")

    resolved: dict[str, np.ndarray | np.ma.MaskedArray] = {}
    for band_key in INDEX_REQUIRED_BANDS[index_name]:
        if band_key in bands:
            resolved[band_key] = bands[band_key]
            continue
        matches = [
            value
            for key, value in bands.items()
            if resolve_band_key(key) == band_key
        ]
        if not matches:
            band = get_band(band_key)
            raise DataValidationError(
                f"{index_name} requires {band_key} ({band.name}), which is not available"
            )
        resolved[band_key] = matches[0]

    if index_name == "NDVI":
        return compute_ndvi(resolved["B08"], resolved["B04"])
    if index_name == "NDWI":
        return compute_ndwi(resolved["B03"], resolved["B08"])
    return compute_ndbi(resolved["B11"], resolved["B08"])


def compute_spectral_indices(
    bands: Mapping[str, np.ndarray | np.ma.MaskedArray],
    *,
    include_ndbi: bool | None = None,
) -> dict[str, np.ma.MaskedArray]:
    """Compute NDVI and NDWI, plus NDBI when SWIR1 (B11) is available.

    Args:
        bands: Mapping of canonical keys or aliases to 2D band arrays.
        include_ndbi: Force NDBI on/off. By default it is enabled only when a
            resolvable B11 band is present.
    """
    available = {resolve_band_key(key) for key in bands}
    indices = {
        "NDVI": compute_index("NDVI", bands),
        "NDWI": compute_index("NDWI", bands),
    }

    has_swir = "B11" in available
    if include_ndbi is True and not has_swir:
        raise DataValidationError("NDBI requires B11 (SWIR1), which is not available")
    if include_ndbi is not False and has_swir:
        indices["NDBI"] = compute_index("NDBI", bands)

    logger.debug("Computed spectral indices: %s", ", ".join(indices))
    return indices
