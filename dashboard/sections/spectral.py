"""Spectral-index analysis with explicit, validated band mapping."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import streamlit as st

from config.sentinel2_bands import get_band, infer_band_key, resolve_band_key
from config.settings import Settings
from dashboard.common import (
    cached_indices,
    cached_raster_summary,
    error_message,
    file_token,
    page_header,
    select_file,
    set_spectral_context,
)
from dashboard.data_access import raster_files
from src.features.spectral_indices import INDEX_NAMES, INDEX_REQUIRED_BANDS
from src.utils.errors import GeoAIError
from src.visualization.index_visualizer import plot_index, plot_index_distribution

CONFIG_KEY = "spectral_config"


def _description_map(descriptions: Sequence[object]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for index, description in enumerate(descriptions, start=1):
        if description is None or not str(description).strip():
            continue
        value = str(description).strip()
        try:
            key = resolve_band_key(value)
        except GeoAIError:
            key = infer_band_key(value)
        if key is not None and key not in mapping:
            mapping[key] = index
    return mapping


def _band_label(index: int | None, descriptions: Sequence[object]) -> str:
    if index is None:
        return "Not available"
    if index - 1 < len(descriptions) and descriptions[index - 1]:
        return f"Band {index}: {descriptions[index - 1]}"
    return f"Band {index}"


def _index_statistics(values: np.ma.MaskedArray) -> dict[str, object]:
    valid = values.compressed()
    if valid.size == 0:
        raise ValueError("No valid index values")
    p10, p90 = np.percentile(valid, [10.0, 90.0])
    return {
        "valid_pixels": int(valid.size),
        "mean": float(valid.mean()),
        "median": float(np.median(valid)),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "p10": float(p10),
        "p90": float(p90),
    }


def _render_indices(
    path: str,
    token: int,
    band_mapping: tuple[tuple[str, int], ...],
) -> None:
    indices = cached_indices(path, token, band_mapping)
    summaries: dict[str, dict[str, object]] = {}
    tabs = st.tabs([name for name in INDEX_NAMES if name in indices])
    for tab, name in zip(tabs, [name for name in INDEX_NAMES if name in indices], strict=True):
        with tab:
            values = indices[name]
            try:
                statistics = _index_statistics(values)
            except ValueError as exc:
                st.warning(f"{name} has no valid values: {exc}")
                continue
            summaries[name] = statistics
            metrics = st.columns(4)
            metrics[0].metric("Valid pixels", f"{int(statistics['valid_pixels']):,}")
            metrics[1].metric("Mean", f"{float(statistics['mean']):.4f}")
            metrics[2].metric("Median", f"{float(statistics['median']):.4f}")
            metrics[3].metric(
                "10th–90th percentile",
                f"{float(statistics['p10']):.3f} to {float(statistics['p90']):.3f}",
            )
            map_column, distribution_column = st.columns(2)
            with map_column:
                st.pyplot(plot_index(values, name=name), use_container_width=True)
            with distribution_column:
                st.pyplot(
                    plot_index_distribution(values, name=name),
                    use_container_width=True,
                )
    set_spectral_context(
        {
            "source_path": str(path),
            "band_mapping": dict(band_mapping),
            "indices": summaries,
        }
    )


def render(settings: Settings) -> None:
    page_header(
        "Spectral Analysis",
        "Compute NDVI, NDWI, and optional NDBI after confirming the band mapping.",
    )
    records = raster_files(settings)
    selected = select_file(
        records,
        settings,
        label="Imagery raster for spectral analysis",
        key="spectral-existing",
    )
    if selected is None:
        st.info("Upload or prepare imagery before running spectral analysis.")
        return

    path = str(selected.path)
    token = file_token(path)
    try:
        summary = cached_raster_summary(path, token)
    except GeoAIError as exc:
        error_message("Unable to inspect the selected imagery", exc)
        return

    descriptions = summary.get("descriptions")
    if not isinstance(descriptions, list):
        descriptions = []
    count = int(summary["count"])
    defaults = _description_map(descriptions)
    st.caption(
        "Band mapping is auto-detected from descriptions when possible. "
        "Confirm every mapping before computing indices."
    )
    if not defaults:
        st.warning(
            "No band descriptions were recognised. Select the required bands manually. "
            "For Sentinel-2, the expected inputs are B03, B04, B08, and optional B11."
        )

    selections: dict[str, int | None] = {}
    required_keys = ("B03", "B04", "B08")
    for key in required_keys:
        band = get_band(key)
        default = defaults.get(key, 1)
        selections[key] = st.selectbox(
            f"{band.name} ({key}) band",
            options=list(range(1, count + 1)),
            index=min(max(default - 1, 0), count - 1),
            format_func=lambda value: _band_label(int(value), descriptions),
            key=f"spectral-{token}-{key}",
        )
    swir_band = get_band("B11")
    swir_default = defaults.get("B11")
    swir_options: list[int | None] = [None, *range(1, count + 1)]
    swir_index = 0 if swir_default is None else min(swir_default, count)
    selections["B11"] = st.selectbox(
        f"{swir_band.name} ({swir_band.key}) band (optional)",
        options=swir_options,
        index=swir_index,
        format_func=lambda value: _band_label(value, descriptions),
        key=f"spectral-{token}-B11",
    )
    include_ndbi = st.checkbox(
        "Compute NDBI when SWIR1 is selected",
        value=selections["B11"] is not None,
        disabled=selections["B11"] is None,
        key=f"spectral-{token}-ndbi",
    )

    use_swir = include_ndbi and selections["B11"] is not None
    used_keys = [*required_keys, "B11"] if use_swir else list(required_keys)
    current_bands = tuple(
        (key, int(selections[key])) for key in used_keys if selections[key] is not None
    )
    chosen = [band_index for _, band_index in current_bands]
    duplicate = len(chosen) != len(set(chosen))
    if duplicate:
        st.error("Each spectral index input must use a different band.")
    elif any(selections[key] is None for key in required_keys):
        st.error("Green, red, and NIR bands are required.")
    elif st.button("Compute spectral indices", type="primary", key="compute-spectral"):
        st.session_state[CONFIG_KEY] = {"path": path, "bands": current_bands}

    config = st.session_state.get(CONFIG_KEY)
    same_source = isinstance(config, dict) and config.get("path") == path
    if same_source and config.get("bands") == current_bands:
        bands = config.get("bands")
        if isinstance(bands, tuple) and bands:
            try:
                _render_indices(path, token, bands)
            except GeoAIError as exc:
                error_message("Unable to compute spectral indices", exc)
            st.caption(
                "Indices describe spectral responses, not confirmed land-cover classes. "
                "Clouds, shadows, mixed pixels, and atmospheric effects can bias results."
            )
    elif not duplicate:
        if same_source:
            st.info("The band mapping changed. Compute indices again to refresh results.")
        else:
            st.info("Confirm the band mapping, then compute indices to display results.")

    st.caption("Required bands by index: " + ", ".join(
        f"{name}={', '.join(INDEX_REQUIRED_BANDS[name])}" for name in INDEX_NAMES
    ))


__all__ = ["render"]
