"""ESA WorldCover 2021 class definitions and helpers.

This module is the single source of truth for land-cover codes used by label
preparation, training, classification, statistics, visualization, and export.
The codes and colours follow the ESA WorldCover 2021 v200 product.

Reference: https://esa-worldcover.org/en
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

NODATA_CLASS_CODE: Final[int] = 0

ESA_WORLDCOVER_CLASSES: Final[Mapping[int, str]] = MappingProxyType(
    {
        10: "Tree Cover",
        20: "Shrubland",
        30: "Grassland",
        40: "Cropland",
        50: "Built-up",
        60: "Bare / Sparse Vegetation",
        70: "Snow and Ice",
        80: "Permanent Water Bodies",
        90: "Herbaceous Wetland",
        95: "Mangroves",
        100: "Moss and Lichen",
    }
)

ESA_WORLDCOVER_COLORS: Final[Mapping[int, str]] = MappingProxyType(
    {
        10: "#006400",
        20: "#FFBB22",
        30: "#FFFF4C",
        40: "#F096FF",
        50: "#FA0000",
        60: "#B4B4B4",
        70: "#F0F0F0",
        80: "#0064C8",
        90: "#0096A0",
        95: "#00CF75",
        100: "#FAE6A0",
    }
)

VALID_CLASS_CODES: Final[frozenset[int]] = frozenset(ESA_WORLDCOVER_CLASSES)

_CLASS_CODE_BY_NAME: Final[Mapping[str, int]] = MappingProxyType(
    {name.casefold(): code for code, name in ESA_WORLDCOVER_CLASSES.items()}
)


def is_valid_class_code(code: int) -> bool:
    """Return ``True`` if ``code`` is a recognised ESA WorldCover class code."""
    return code in VALID_CLASS_CODES


def get_class_name(code: int) -> str:
    """Return the display name for a class code.

    Raises:
        KeyError: If the code is not a recognised ESA WorldCover class.
    """
    try:
        return ESA_WORLDCOVER_CLASSES[code]
    except KeyError as exc:
        raise KeyError(f"Unknown ESA WorldCover class code: {code!r}") from exc


def get_class_code(name: str) -> int:
    """Return the class code for a class name (case-insensitive).

    Raises:
        KeyError: If the name is not a recognised ESA WorldCover class.
    """
    key = name.strip().casefold()
    try:
        return _CLASS_CODE_BY_NAME[key]
    except KeyError as exc:
        raise KeyError(f"Unknown ESA WorldCover class name: {name!r}") from exc


def get_class_color(code: int) -> str:
    """Return the official hex colour for a class code."""
    if not is_valid_class_code(code):
        raise KeyError(f"Unknown ESA WorldCover class code: {code!r}")
    return ESA_WORLDCOVER_COLORS[code]


def get_class_colors() -> dict[int, str]:
    """Return a mutable copy of the class-code to colour mapping."""
    return dict(ESA_WORLDCOVER_COLORS)


def get_class_legend() -> list[dict[str, object]]:
    """Return legend entries sorted by class code."""
    return [
        {
            "code": code,
            "name": ESA_WORLDCOVER_CLASSES[code],
            "color": ESA_WORLDCOVER_COLORS[code],
        }
        for code in sorted(ESA_WORLDCOVER_CLASSES)
    ]
