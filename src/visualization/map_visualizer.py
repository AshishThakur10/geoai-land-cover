"""Folium maps for georeferenced rasters and vector boundaries."""

from __future__ import annotations

import math
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
from matplotlib import colormaps
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds

from config.land_cover_classes import get_class_color, get_class_name, is_valid_class_code
from src.data.raster_loader import ensure_georeferenced, open_raster, read_metadata
from src.utils.errors import DataValidationError, VectorError

WGS84 = "EPSG:4326"
CLASSIFICATION_COLORS = {
    "unchanged": (217, 217, 217, 255),
    "changed": (220, 30, 30, 255),
}


def _map_center(bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    west, south, east, north = bounds
    return (south + north) / 2.0, (west + east) / 2.0


def _web_bounds(
    source_crs: object,
    bounds: object,
) -> tuple[float, float, float, float]:
    try:
        west, south, east, north = transform_bounds(
            source_crs,
            WGS84,
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
            densify_pts=21,
        )
    except Exception as exc:
        raise DataValidationError(f"Unable to transform raster bounds to WGS84: {exc}") from exc
    values = (west, south, east, north)
    if not all(math.isfinite(value) for value in values):
        raise DataValidationError("Raster bounds are not finite in WGS84")
    return values


def _base_map(
    bounds: tuple[float, float, float, float],
    *,
    tile_layer: str = "OpenStreetMap",
) -> folium.Map:
    center = _map_center(bounds)
    raster_map = folium.Map(
        location=center,
        zoom_start=10,
        tiles=tile_layer,
        control_scale=True,
    )
    raster_map.fit_bounds(
        [[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        padding=(20, 20),
    )
    return raster_map


def _read_display_data(
    path: str | Path,
    *,
    band_index: int,
    max_size: int,
    resampling: Resampling,
) -> tuple[np.ma.MaskedArray, tuple[float, float, float, float]]:
    metadata = read_metadata(path)
    ensure_georeferenced(metadata)
    if band_index < 1 or band_index > metadata.count:
        raise DataValidationError(
            f"Band index {band_index} is out of range for {metadata.path.name} "
            f"({metadata.count} band(s))"
        )
    if max_size < 1:
        raise DataValidationError(f"max_size must be >= 1, got {max_size}")

    scale = min(1.0, max_size / max(metadata.width, metadata.height))
    output_height = max(1, int(round(metadata.height * scale)))
    output_width = max(1, int(round(metadata.width * scale)))
    with open_raster(path) as dataset:
        data = dataset.read(
            band_index,
            out_shape=(output_height, output_width),
            masked=True,
            resampling=resampling,
        )
    assert metadata.crs is not None
    return data, _web_bounds(metadata.crs, metadata.bounds)


def _stretch(values: np.ma.MaskedArray) -> np.ndarray:
    data = np.ma.getdata(values).astype(float)
    mask = np.ma.getmaskarray(values).astype(bool, copy=True)
    mask |= ~np.isfinite(data)
    valid = data[~mask]
    if valid.size == 0:
        normalized = np.zeros(data.shape, dtype=float)
    else:
        lower, upper = np.percentile(valid, [2.0, 98.0])
        if upper <= lower:
            normalized = np.zeros(data.shape, dtype=float)
        else:
            normalized = np.clip((data - lower) / (upper - lower), 0.0, 1.0)
    normalized[mask] = 0.0
    return normalized


def _add_raster_layer(
    raster_map: folium.Map,
    image: np.ndarray,
    *,
    bounds: tuple[float, float, float, float],
    name: str,
    opacity: float,
) -> None:
    folium.raster_layers.ImageOverlay(
        image=image,
        bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        name=name,
        opacity=opacity,
        pixelated=True,
        origin="upper",
    ).add_to(raster_map)


def _add_legend(
    raster_map: folium.Map,
    entries: list[tuple[str, str]],
    *,
    title: str,
) -> None:
    if not entries:
        return
    rows = "".join(
        '<div style="display:flex;align-items:center;margin:2px 0;">'
        f'<span style="width:14px;height:14px;background:{color};'
        'display:inline-block;margin-right:6px;border:1px solid #555;"></span>'
        f"<span>{label}</span></div>"
        for label, color in entries
    )
    html = (
        '<div style="position:fixed;bottom:30px;left:30px;z-index:9999;'
        "background:white;padding:8px 10px;border:1px solid #999;"
        'border-radius:4px;font-size:12px;max-height:260px;overflow:auto;">'
        f'<div style="font-weight:600;margin-bottom:4px;">{title}</div>{rows}</div>'
    )
    raster_map.get_root().html.add_child(folium.Element(html))


def classification_map(
    classification_path: str | Path,
    *,
    band_index: int = 1,
    max_size: int = 1400,
    opacity: float = 0.75,
    tile_layer: str = "OpenStreetMap",
) -> folium.Map:
    """Render a classified raster with ESA WorldCover colours and a legend."""
    metadata = read_metadata(classification_path)
    ensure_georeferenced(metadata)
    if band_index < 1 or band_index > metadata.count:
        raise DataValidationError(
            f"Band index {band_index} is out of range for {metadata.path.name}"
        )

    data, bounds = _read_display_data(
        classification_path,
        band_index=band_index,
        max_size=max_size,
        resampling=Resampling.nearest,
    )
    values = np.ma.getdata(data)
    nodata = metadata.nodata
    valid = ~np.ma.getmaskarray(data)
    if nodata is not None:
        valid &= values != nodata
    unique = sorted({int(value) for value in np.unique(values[valid])})
    invalid = [code for code in unique if not is_valid_class_code(code)]
    if invalid:
        raise DataValidationError(
            f"Classification contains unknown class codes: {invalid}"
        )

    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    legend: list[tuple[str, str]] = []
    for code in unique:
        color = get_class_color(code)
        raw = color.lstrip("#")
        rgba[valid & (values == code)] = (
            int(raw[0:2], 16),
            int(raw[2:4], 16),
            int(raw[4:6], 16),
            255,
        )
        legend.append((get_class_name(code), color))

    raster_map = _base_map(bounds, tile_layer=tile_layer)
    _add_raster_layer(
        raster_map,
        rgba,
        bounds=bounds,
        name="Land cover",
        opacity=opacity,
    )
    _add_legend(raster_map, legend, title="Land-cover classes")
    folium.LayerControl(collapsed=True).add_to(raster_map)
    return raster_map


def continuous_raster_map(
    raster_path: str | Path,
    *,
    band_index: int = 1,
    max_size: int = 1400,
    cmap: str = "viridis",
    opacity: float = 0.75,
    name: str | None = None,
) -> folium.Map:
    """Render a continuous raster band with a percentile-stretched colourmap."""
    data, bounds = _read_display_data(
        raster_path,
        band_index=band_index,
        max_size=max_size,
        resampling=Resampling.bilinear,
    )
    normalized = _stretch(data)
    colormap = colormaps.get_cmap(cmap)
    rgba = (colormap(normalized) * 255).astype(np.uint8)
    mask = np.ma.getmaskarray(data) | ~np.isfinite(np.ma.getdata(data))
    rgba[mask, 3] = 0

    raster_map = _base_map(bounds)
    _add_raster_layer(
        raster_map,
        rgba,
        bounds=bounds,
        name=name or f"Band {band_index}",
        opacity=opacity,
    )
    folium.LayerControl(collapsed=True).add_to(raster_map)
    return raster_map


def change_map(
    change_path: str | Path,
    *,
    band_index: int = 1,
    max_size: int = 1400,
    opacity: float = 0.8,
) -> folium.Map:
    """Render a change raster using grey for unchanged and red for changed."""
    data, bounds = _read_display_data(
        change_path,
        band_index=band_index,
        max_size=max_size,
        resampling=Resampling.nearest,
    )
    values = np.ma.getdata(data)
    mask = np.ma.getmaskarray(data)
    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    unchanged = (values == 0) & ~mask
    changed = (values == 1) & ~mask
    rgba[unchanged] = CLASSIFICATION_COLORS["unchanged"]
    rgba[changed] = CLASSIFICATION_COLORS["changed"]

    raster_map = _base_map(bounds)
    _add_raster_layer(
        raster_map,
        rgba,
        bounds=bounds,
        name="Land-cover change",
        opacity=opacity,
    )
    _add_legend(
        raster_map,
        [("Unchanged", "#D9D9D9"), ("Changed", "#DC1E1E")],
        title="Change detection",
    )
    folium.LayerControl(collapsed=True).add_to(raster_map)
    return raster_map


def boundary_map(
    frame: gpd.GeoDataFrame,
    *,
    color: str = "#1F77B4",
    fill_opacity: float = 0.15,
) -> folium.Map:
    """Render a vector boundary as a Folium GeoJSON layer."""
    if not isinstance(frame, gpd.GeoDataFrame):
        raise VectorError("Expected a GeoDataFrame")
    if frame.crs is None:
        raise VectorError("Cannot map a boundary without a CRS")
    geographic = frame.to_crs(WGS84)
    minx, miny, maxx, maxy = geographic.total_bounds
    bounds = (float(minx), float(miny), float(maxx), float(maxy))
    if not all(math.isfinite(value) for value in bounds):
        raise VectorError("Boundary bounds are not finite")

    raster_map = _base_map(bounds)
    folium.GeoJson(
        geographic.to_json(),
        name="Study area",
        style_function=lambda _: {
            "fillColor": color,
            "color": color,
            "weight": 2,
            "fillOpacity": fill_opacity,
        },
    ).add_to(raster_map)
    folium.LayerControl(collapsed=True).add_to(raster_map)
    return raster_map


__all__ = [
    "WGS84",
    "boundary_map",
    "change_map",
    "classification_map",
    "continuous_raster_map",
]
