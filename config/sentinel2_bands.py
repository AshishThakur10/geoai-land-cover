"""Sentinel-2 Level-2A band definitions and filename inference."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from src.utils.errors import DataValidationError


@dataclass(frozen=True)
class Sentinel2Band:
    """Metadata for one Sentinel-2 band."""

    key: str
    name: str
    resolution_m: int
    wavelength_nm: float
    aliases: tuple[str, ...] = ()


SENTINEL2_BANDS: Final[Mapping[str, Sentinel2Band]] = MappingProxyType(
    {
        "B02": Sentinel2Band("B02", "Blue", 10, 492.4, ("B2", "BLUE")),
        "B03": Sentinel2Band("B03", "Green", 10, 559.8, ("B3", "GREEN")),
        "B04": Sentinel2Band("B04", "Red", 10, 664.6, ("B4", "RED")),
        "B08": Sentinel2Band("B08", "NIR", 10, 832.8, ("B8", "NIR", "NEARINFRARED")),
        "B11": Sentinel2Band("B11", "SWIR1", 20, 1613.7, ("B11", "SWIR1", "SWIR")),
        "B12": Sentinel2Band("B12", "SWIR2", 20, 2202.4, ("B12", "SWIR2")),
    }
)

CORE_FEATURE_BANDS: Final[tuple[str, ...]] = ("B02", "B03", "B04", "B08")
OPTIONAL_FEATURE_BANDS: Final[tuple[str, ...]] = ("B11", "B12")

_BAND_PATTERN = re.compile(r"(?<![A-Z0-9])B0?(\d{1,2})(?![0-9])", re.IGNORECASE)


def _build_alias_map() -> Mapping[str, str]:
    aliases: dict[str, str] = {}
    for key, band in SENTINEL2_BANDS.items():
        for alias in (key, *band.aliases):
            aliases[alias.casefold()] = key
    return MappingProxyType(aliases)


_ALIAS_TO_KEY: Final[Mapping[str, str]] = _build_alias_map()


def resolve_band_key(value: str) -> str:
    """Resolve a band key, alias, or case variant to a canonical key.

    Examples: ``b8``, ``B08``, and ``nir`` all resolve to ``B08``.

    Raises:
        DataValidationError: If the band is not recognised.
    """
    key = _ALIAS_TO_KEY.get(str(value).strip().casefold())
    if key is None:
        allowed = ", ".join(SENTINEL2_BANDS)
        raise DataValidationError(f"Unknown Sentinel-2 band {value!r}; expected one of {allowed}")
    return key


def get_band(value: str) -> Sentinel2Band:
    """Return band metadata for a key or alias."""
    return SENTINEL2_BANDS[resolve_band_key(value)]


def infer_band_key(filename: str) -> str | None:
    """Infer a canonical band key from a Sentinel-2 filename, if possible.

    Matches common patterns such as ``T33UVP_20210701_B04_10m.tif`` and returns
    ``None`` when no supported band code is present.
    """
    stem = Path(str(filename).replace("\\", "/")).stem.upper()
    for match in _BAND_PATTERN.finditer(stem):
        candidate = f"B{int(match.group(1)):02d}"
        if candidate in SENTINEL2_BANDS:
            return candidate
    return None


def core_bands_present(keys: set[str] | frozenset[str]) -> bool:
    """Return ``True`` when all core 10 m feature bands are present."""
    return set(CORE_FEATURE_BANDS).issubset(keys)
