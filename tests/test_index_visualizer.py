from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.features.spectral_indices import compute_ndvi
from src.visualization.index_visualizer import (
    plot_index,
    plot_index_distribution,
    save_index_preview,
)


def _ndvi() -> np.ma.MaskedArray:
    nir = np.array([[0.8, 0.5, 0.0], [0.2, 0.4, 0.6]], dtype=np.float32)
    red = np.array([[0.2, 0.3, 0.0], [0.1, 0.2, 0.3]], dtype=np.float32)
    return compute_ndvi(nir, red)


def test_plot_index_uses_fixed_scale_and_title() -> None:
    figure = plot_index(_ndvi(), name="NDVI")

    axes = figure.axes[0]
    assert axes.get_title() == "NDVI map"
    image = axes.get_images()[0]
    assert image.get_clim() == (-1.0, 1.0)
    assert axes.get_xlabel() == "Column"
    assert axes.get_ylabel() == "Row"


def test_plot_index_rejects_non_2d_data() -> None:
    with pytest.raises(ValueError, match="2D index array"):
        plot_index(np.ones((2, 2, 2)))


def test_plot_index_distribution_uses_valid_values_only() -> None:
    figure = plot_index_distribution(_ndvi(), name="NDVI", bins=10)

    axes = figure.axes[0]
    assert axes.get_title() == "NDVI distribution"
    assert axes.get_ylabel() == "Pixel count"
    assert len(axes.patches) > 0
    assert axes.patches[0].get_x() == pytest.approx(-1.0)
    assert axes.patches[-1].get_x() + axes.patches[-1].get_width() == pytest.approx(1.0)


def test_plot_index_distribution_requires_valid_values() -> None:
    with pytest.raises(ValueError, match="without valid"):
        plot_index_distribution(np.ma.masked_all((3, 3)))
    with pytest.raises(ValueError, match="2D index array"):
        plot_index_distribution(np.ones(3))


def test_save_index_preview_writes_png(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "ndvi.png"

    result = save_index_preview(_ndvi(), output, name="NDVI")

    assert result == output.resolve()
    assert output.exists()
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
