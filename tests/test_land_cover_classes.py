from __future__ import annotations

import pytest

from config.land_cover_classes import (
    ESA_WORLDCOVER_CLASSES,
    ESA_WORLDCOVER_COLORS,
    NODATA_CLASS_CODE,
    VALID_CLASS_CODES,
    get_class_code,
    get_class_color,
    get_class_colors,
    get_class_legend,
    get_class_name,
    is_valid_class_code,
)


def test_expected_worldcover_classes_are_present() -> None:
    assert len(ESA_WORLDCOVER_CLASSES) == 11
    assert dict(ESA_WORLDCOVER_CLASSES) == {
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


def test_valid_codes_match_class_mapping() -> None:
    assert frozenset(ESA_WORLDCOVER_CLASSES) == VALID_CLASS_CODES
    assert is_valid_class_code(10)
    assert not is_valid_class_code(0)
    assert not is_valid_class_code(999)
    assert NODATA_CLASS_CODE == 0


def test_get_class_name() -> None:
    assert get_class_name(50) == "Built-up"
    with pytest.raises(KeyError, match="999"):
        get_class_name(999)


def test_get_class_code_is_case_insensitive_and_strips_whitespace() -> None:
    assert get_class_code(" tree cover ") == 10
    assert get_class_code("BUILT-UP") == 50
    with pytest.raises(KeyError, match="Forest"):
        get_class_code("Forest")


def test_colors_exist_for_every_class() -> None:
    assert set(ESA_WORLDCOVER_COLORS) == set(ESA_WORLDCOVER_CLASSES)
    assert get_class_color(10) == "#006400"
    with pytest.raises(KeyError):
        get_class_color(999)

    colors = get_class_colors()
    colors[10] = "#FFFFFF"
    assert get_class_color(10) == "#006400"


def test_legend_entries_are_sorted_and_complete() -> None:
    legend = get_class_legend()

    assert [entry["code"] for entry in legend] == sorted(ESA_WORLDCOVER_CLASSES)
    assert all(set(entry) == {"code", "name", "color"} for entry in legend)
    assert legend[0] == {"code": 10, "name": "Tree Cover", "color": "#006400"}


def test_class_mapping_is_read_only() -> None:
    with pytest.raises(TypeError):
        ESA_WORLDCOVER_CLASSES[10] = "Changed"  # type: ignore[index]
