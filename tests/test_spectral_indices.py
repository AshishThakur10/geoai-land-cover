from __future__ import annotations

import numpy as np
import pytest

from src.features.spectral_indices import (
    compute_index,
    compute_ndbi,
    compute_ndvi,
    compute_ndwi,
    compute_spectral_indices,
    normalized_difference,
)
from src.utils.errors import DataValidationError


def _arrays():
    nir = np.array([[0.8, 0.5], [0.2, 0.4]], dtype=np.float32)
    red = np.array([[0.2, 0.3], [0.1, 0.2]], dtype=np.float32)
    green = np.array([[0.1, 0.2], [0.3, 0.1]], dtype=np.float32)
    swir = np.array([[0.3, 0.4], [0.5, 0.1]], dtype=np.float32)
    return nir, red, green, swir


def test_ndvi_ndwi_and_ndbi_formulas() -> None:
    nir, red, green, swir = _arrays()

    ndvi = compute_ndvi(nir, red)
    ndwi = compute_ndwi(green, nir)
    ndbi = compute_ndbi(swir, nir)

    assert ndvi[0, 0] == pytest.approx(0.6)
    assert ndvi[0, 1] == pytest.approx(0.25)
    assert ndwi[0, 0] == pytest.approx((0.1 - 0.8) / (0.1 + 0.8))
    assert ndbi[0, 0] == pytest.approx((0.3 - 0.8) / (0.3 + 0.8))
    assert ndvi.dtype == np.dtype("float32")


def test_normalized_difference_masks_zero_denominator() -> None:
    positive = np.array([[1.0, 0.0, np.nan]], dtype=np.float32)
    negative = np.array([[-1.0, 0.0, 0.5]], dtype=np.float32)

    result = normalized_difference(positive, negative)

    assert np.ma.getmaskarray(result).all()
    assert not np.isinf(result.filled(np.nan)).any()


def test_normalized_difference_propagates_input_masks() -> None:
    positive = np.ma.array([[1.0, 2.0]], mask=[[False, True]], dtype=np.float32)
    negative = np.ma.array([[1.0, 1.0]], mask=[[False, False]], dtype=np.float32)

    result = normalized_difference(positive, negative)

    assert result[0, 0] == pytest.approx(0.0)
    assert bool(np.ma.getmaskarray(result)[0, 1])


def test_normalized_difference_requires_matching_shapes() -> None:
    with pytest.raises(DataValidationError, match="share a shape"):
        normalized_difference(np.ones((2, 2)), np.ones((3, 3)))
    with pytest.raises(DataValidationError, match="at least one-dimensional"):
        normalized_difference(np.array(1.0), np.array(2.0))


def test_indices_stay_within_minus_one_and_one() -> None:
    rng = np.random.default_rng(42)
    first = rng.uniform(0.01, 1.0, size=(50, 50)).astype(np.float32)
    second = rng.uniform(0.01, 1.0, size=(50, 50)).astype(np.float32)

    result = normalized_difference(first, second)
    valid = result.compressed()

    assert valid.size > 0
    assert valid.min() >= -1.0
    assert valid.max() <= 1.0


def test_compute_index_accepts_aliases() -> None:
    nir, red, _, _ = _arrays()
    bands = {"nir": nir, "red": red}

    result = compute_index("ndvi", bands)

    assert result[0, 0] == pytest.approx(0.6)


def test_compute_index_reports_missing_bands_and_unknown_index() -> None:
    nir, _, _, _ = _arrays()

    with pytest.raises(DataValidationError, match="requires B04"):
        compute_index("NDVI", {"B08": nir})
    with pytest.raises(DataValidationError, match="Unknown spectral index"):
        compute_index("SAVI", {"B08": nir})


def test_compute_spectral_indices_omits_ndbi_without_swir() -> None:
    nir, red, green, _ = _arrays()

    indices = compute_spectral_indices({"B08": nir, "B04": red, "B03": green})

    assert set(indices) == {"NDVI", "NDWI"}
    assert "NDBI" not in indices


def test_compute_spectral_indices_includes_ndbi_with_swir() -> None:
    nir, red, green, swir = _arrays()

    indices = compute_spectral_indices(
        {"B08": nir, "B04": red, "B03": green, "B11": swir}
    )

    assert set(indices) == {"NDVI", "NDWI", "NDBI"}


def test_compute_spectral_indices_respects_ndbi_flag() -> None:
    nir, red, green, swir = _arrays()
    bands = {"B08": nir, "B04": red, "B03": green, "B11": swir}

    assert "NDBI" not in compute_spectral_indices(bands, include_ndbi=False)
    with pytest.raises(DataValidationError, match="requires B11"):
        compute_spectral_indices(
            {"B08": nir, "B04": red, "B03": green},
            include_ndbi=True,
        )
