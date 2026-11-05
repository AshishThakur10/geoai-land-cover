"""Matplotlib-based raster previews (paths are never fabricated or modified)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.data.raster_loader import read_band
from src.utils.file_utils import ensure_directory


def _as_masked_float(data: np.ndarray | np.ma.MaskedArray) -> np.ma.MaskedArray:
    values = np.ma.getdata(data).astype(float)
    mask = np.ma.getmaskarray(data).astype(bool, copy=True)
    mask |= ~np.isfinite(values)
    return np.ma.array(values, mask=mask)


def _percentile_stretch(
    array: np.ma.MaskedArray,
    lower: float = 2.0,
    upper: float = 98.0,
) -> np.ma.MaskedArray:
    valid = array.compressed()
    if valid.size == 0:
        return array.copy()
    vmin, vmax = np.percentile(valid, [lower, upper])
    if vmax <= vmin:
        return array.copy()
    return (array - vmin) / (vmax - vmin)


def plot_band(
    data: np.ndarray | np.ma.MaskedArray,
    *,
    title: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    figsize: tuple[float, float] = (6.0, 5.0),
) -> Figure:
    """Render a single raster band as a Matplotlib figure."""
    values = _as_masked_float(data)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D band array, got shape {values.shape}")

    figure = Figure(figsize=figsize, constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    image = axes.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    axes.set_title(title or "Raster band")
    axes.set_xlabel("Column")
    axes.set_ylabel("Row")
    if colorbar:
        figure.colorbar(image, ax=axes, shrink=0.85)
    return figure


def plot_rgb(
    red: np.ndarray | np.ma.MaskedArray,
    green: np.ndarray | np.ma.MaskedArray,
    blue: np.ndarray | np.ma.MaskedArray,
    *,
    title: str | None = None,
    stretch: bool = True,
    figsize: tuple[float, float] = (6.0, 5.0),
) -> Figure:
    """Render a three-band RGB composite.

    Args:
        red: Red band array.
        green: Green band array.
        blue: Blue band array.
        title: Optional figure title.
        stretch: Apply a 2nd–98th percentile stretch per channel.
        figsize: Figure size in inches.

    Raises:
        ValueError: If band shapes differ or arrays are not 2D.
    """
    bands = [_as_masked_float(band) for band in (red, green, blue)]
    shapes = {band.shape for band in bands}
    if len(shapes) != 1:
        raise ValueError(f"RGB bands must share a shape, got {sorted(shapes)}")
    if bands[0].ndim != 2:
        raise ValueError(f"Expected 2D bands, got shape {bands[0].shape}")

    combined_mask = np.logical_or.reduce([np.ma.getmaskarray(band) for band in bands])
    channels = []
    for band in bands:
        channel = _percentile_stretch(band) if stretch else band
        channels.append(np.clip(channel.filled(0.0), 0.0, 1.0))

    rgb = np.dstack(channels)
    figure = Figure(figsize=figsize, constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.imshow(rgb)
    axes.set_title(title or "RGB composite")
    axes.set_xlabel("Column")
    axes.set_ylabel("Row")
    if combined_mask.any():
        axes.contour(combined_mask, levels=[0.5], colors="black", linewidths=0.3, alpha=0.5)
    return figure


def save_band_preview(
    raster_path: str | Path,
    output_path: str | Path,
    *,
    band_index: int = 1,
    title: str | None = None,
    cmap: str = "viridis",
    dpi: int = 120,
) -> Path:
    """Save a single-band preview PNG (or another Matplotlib-supported format)."""
    destination = Path(output_path).expanduser().resolve()
    ensure_directory(destination.parent)
    figure = plot_band(
        read_band(raster_path, band_index=band_index),
        title=title or f"{Path(raster_path).name} - band {band_index}",
        cmap=cmap,
    )
    figure.savefig(destination, dpi=dpi)
    return destination
