from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from src.preprocessing.raster_clipping import clip_raster_to_aoi
from src.utils.errors import RasterError
from src.visualization.raster_visualizer import plot_band, plot_rgb, save_band_preview
from tests.fixtures import write_synthetic_aoi_geojson, write_synthetic_raster


def test_plot_band_returns_figure() -> None:
    data = np.arange(100, dtype=float).reshape(10, 10)

    figure = plot_band(data, title="Synthetic band")

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2
    assert figure.axes[0].get_title() == "Synthetic band"


def test_plot_band_without_colorbar() -> None:
    data = np.arange(100, dtype=float).reshape(10, 10)

    figure = plot_band(data, colorbar=False)

    assert len(figure.axes) == 1


def test_plot_band_handles_masked_values() -> None:
    data = np.ma.array(
        np.arange(100, dtype=float).reshape(10, 10),
        mask=np.zeros((10, 10), dtype=bool),
    )
    data.mask[0, 0] = True

    figure = plot_band(data)
    assert isinstance(figure, Figure)


def test_plot_band_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="Expected a 2D band array"):
        plot_band(np.arange(10))


def test_plot_rgb_returns_figure() -> None:
    red = np.arange(100, dtype=float).reshape(10, 10)
    green = red + 10
    blue = red + 20

    figure = plot_rgb(red, green, blue, title="Synthetic RGB")

    assert isinstance(figure, Figure)
    assert figure.axes[0].get_title() == "Synthetic RGB"


def test_plot_rgb_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        plot_rgb(np.zeros((10, 10)), np.zeros((9, 10)), np.zeros((10, 10)))


def test_save_band_preview(tmp_path: Path) -> None:
    raster = write_synthetic_raster(tmp_path / "raster.tif", width=20, height=20)
    output = tmp_path / "previews" / "band2.png"

    saved = save_band_preview(raster, output, band_index=2)

    assert saved == output.resolve()
    assert output.exists()
    assert output.stat().st_size > 0


def test_save_band_preview_rejects_invalid_band(tmp_path: Path) -> None:
    raster = write_synthetic_raster(tmp_path / "raster.tif", count=2)

    with pytest.raises(RasterError, match="out of range"):
        save_band_preview(raster, tmp_path / "preview.png", band_index=3)


def test_clip_result_can_be_visualized(tmp_path: Path) -> None:
    raster = write_synthetic_raster(tmp_path / "raster.tif", width=40, height=40)
    aoi = write_synthetic_aoi_geojson(tmp_path / "aoi.geojson")
    result = clip_raster_to_aoi(raster, aoi)

    figure = plot_band(result.data[0], title="Clipped band")

    assert isinstance(figure, Figure)
