"""Visualization helpers for spectral-index maps and distributions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.utils.file_utils import ensure_directory

INDEX_COLORMAPS: dict[str, str] = {
    "NDVI": "RdYlGn",
    "NDWI": "Blues",
    "NDBI": "RdBu",
}


def _as_masked_float(data: np.ndarray | np.ma.MaskedArray) -> np.ma.MaskedArray:
    values = np.ma.asarray(data, dtype=float)
    raw = np.ma.getdata(values)
    mask = np.ma.getmaskarray(values).astype(bool, copy=True)
    mask |= ~np.isfinite(raw)
    return np.ma.array(raw, mask=mask)


def plot_index(
    data: np.ndarray | np.ma.MaskedArray,
    *,
    name: str = "NDVI",
    title: str | None = None,
    cmap: str | None = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    figsize: tuple[float, float] = (6.0, 5.0),
) -> Figure:
    """Render a spectral-index map with a fixed ``[-1, 1]`` scale by default."""
    values = _as_masked_float(data)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D index array, got shape {values.shape}")

    figure = Figure(figsize=figsize, constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    image = axes.imshow(
        values,
        cmap=cmap or INDEX_COLORMAPS.get(name.upper(), "viridis"),
        vmin=vmin,
        vmax=vmax,
    )
    axes.set_title(title or f"{name.upper()} map")
    axes.set_xlabel("Column")
    axes.set_ylabel("Row")
    figure.colorbar(image, ax=axes, shrink=0.85, label=name.upper())
    return figure


def plot_index_distribution(
    data: np.ndarray | np.ma.MaskedArray,
    *,
    name: str = "NDVI",
    bins: int = 50,
    title: str | None = None,
    figsize: tuple[float, float] = (7.0, 4.0),
) -> Figure:
    """Plot a histogram of valid index values.

    Raises:
        ValueError: If the array is not 2D or contains no valid values.
    """
    values = _as_masked_float(data)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D index array, got shape {values.shape}")

    valid = values.compressed()
    if valid.size == 0:
        raise ValueError("Cannot plot a distribution without valid index values")

    figure = Figure(figsize=figsize, constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.hist(valid, bins=bins, range=(-1.0, 1.0), color="steelblue", edgecolor="white")
    axes.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes.set_title(title or f"{name.upper()} distribution")
    axes.set_xlabel(name.upper())
    axes.set_ylabel("Pixel count")
    return figure


def save_index_preview(
    data: np.ndarray | np.ma.MaskedArray,
    output_path: str | Path,
    *,
    name: str = "NDVI",
    title: str | None = None,
    dpi: int = 120,
) -> Path:
    """Save an index map preview as a PNG or another Matplotlib-supported format."""
    destination = Path(output_path).expanduser().resolve()
    ensure_directory(destination.parent)
    figure = plot_index(data, name=name, title=title)
    figure.savefig(destination, dpi=dpi)
    return destination
