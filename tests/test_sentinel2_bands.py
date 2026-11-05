from __future__ import annotations

import pytest

from config.sentinel2_bands import (
    CORE_FEATURE_BANDS,
    OPTIONAL_FEATURE_BANDS,
    SENTINEL2_BANDS,
    core_bands_present,
    get_band,
    infer_band_key,
    resolve_band_key,
)
from src.utils.errors import DataValidationError


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("B02", "B02"),
        ("b2", "B02"),
        ("blue", "B02"),
        ("B03", "B03"),
        ("GREEN", "B03"),
        ("b4", "B04"),
        ("red", "B04"),
        ("B08", "B08"),
        ("b8", "B08"),
        ("nir", "B08"),
        ("B11", "B11"),
        ("swir1", "B11"),
        ("b12", "B12"),
        ("SWIR2", "B12"),
    ],
)
def test_resolve_band_key_accepts_keys_aliases_and_case(value: str, expected: str) -> None:
    assert resolve_band_key(value) == expected


def test_resolve_band_key_rejects_unknown_band() -> None:
    with pytest.raises(DataValidationError, match="Unknown Sentinel-2 band"):
        resolve_band_key("B01")


def test_band_registry_contains_required_metadata() -> None:
    assert set(SENTINEL2_BANDS) == {"B02", "B03", "B04", "B08", "B11", "B12"}
    assert CORE_FEATURE_BANDS == ("B02", "B03", "B04", "B08")
    assert OPTIONAL_FEATURE_BANDS == ("B11", "B12")

    red = get_band("red")
    assert red.key == "B04"
    assert red.name == "Red"
    assert red.resolution_m == 10
    assert red.wavelength_nm == pytest.approx(664.6)
    assert get_band("swir1").resolution_m == 20


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("T33UVP_20210701T101031_B02_10m.tif", "B02"),
        ("T33UVP_20210701_B04_10m.tif", "B04"),
        ("S2B_MSIL2A_20210701_B8_10m.tif", "B08"),
        ("sentinel_b11_20m.tif", "B11"),
        ("B12.tiff", "B12"),
        ("B01.tif", "B01"),
        ("B13.tif", "B13"),
        ("red.tif", None),
        ("B02", "B02"),
        ("prefix_B04_suffix.tif", "B04"),
    ],
)
def test_infer_band_key(filename: str, expected: str | None) -> None:
    if expected in SENTINEL2_BANDS:
        assert infer_band_key(filename) == expected
    else:
        assert infer_band_key(filename) is None


def test_infer_band_key_handles_windows_paths() -> None:
    assert infer_band_key(r"C:\data\scene_B08_10m.tif") == "B08"


def test_core_bands_present() -> None:
    assert core_bands_present({"B02", "B03", "B04", "B08"})
    assert core_bands_present({"B02", "B03", "B04", "B08", "B11"})
    assert not core_bands_present({"B02", "B03", "B04"})
