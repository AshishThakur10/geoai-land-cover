"""Satellite-imagery metadata, preview, and map section."""

from __future__ import annotations

import streamlit as st
from streamlit_folium import st_folium

from config.settings import Settings
from dashboard.common import (
    cached_raster_summary,
    error_message,
    file_token,
    page_header,
    relative_label,
    select_file,
    set_imagery_context,
)
from dashboard.data_access import (
    max_upload_bytes,
    prepare_raster_upload,
    raster_files,
    read_band_preview,
)
from src.utils.errors import GeoAIError
from src.visualization.map_visualizer import continuous_raster_map
from src.visualization.raster_visualizer import plot_band

INSIGHT_CAPTION = (
    "Previews are downsampled for display only. Calculations elsewhere use the "
    "source raster, and the raster is never modified."
)


def _show_raster(path: str, settings: Settings) -> None:
    summary = cached_raster_summary(path, file_token(path))
    descriptions = summary.get("descriptions")
    if not isinstance(descriptions, list):
        descriptions = []
    count = int(summary["count"])
    width = int(summary["width"])
    height = int(summary["height"])
    resolution = summary.get("resolutions")
    resolution_label = (
        f"{float(resolution[0]):.4g} × {float(resolution[1]):.4g}"
        if isinstance(resolution, list) and len(resolution) == 2
        else "Unknown"
    )

    columns = st.columns(4)
    columns[0].metric("Dimensions", f"{width:,} × {height:,}")
    columns[1].metric("Bands", count)
    columns[2].metric("Pixel size", resolution_label)
    columns[3].metric("CRS", str(summary.get("crs") or "Missing"))

    acquisition = summary.get("acquisition_date")
    if acquisition:
        st.write(f"**Acquisition date:** {acquisition}")
    else:
        st.caption("Acquisition date is not recorded in the raster metadata.")

    st.write(f"**Source:** `{relative_label(path, settings)}`")
    st.dataframe(
        [
            {"band": index + 1, "description": description or "—"}
            for index, description in enumerate(descriptions)
        ],
        use_container_width=True,
        hide_index=True,
    )

    options = list(range(1, count + 1))
    band_index = st.selectbox(
        "Preview band",
        options=options,
        format_func=lambda value: (
            f"Band {value}: {descriptions[value - 1]}"
            if value - 1 < len(descriptions) and descriptions[value - 1]
            else f"Band {value}"
        ),
        key=f"imagery-band-{file_token(path)}",
    )
    data = read_band_preview(path, int(band_index))
    figure = plot_band(data, title=f"Band {band_index} preview")
    st.pyplot(figure, use_container_width=True)
    st.caption(INSIGHT_CAPTION)

    try:
        raster_map = continuous_raster_map(
            path,
            band_index=int(band_index),
            name=f"Band {band_index}",
        )
        st_folium(
            raster_map,
            width=900,
            height=460,
            key=f"imagery-map-{file_token(path)}-{band_index}",
        )
    except GeoAIError as exc:
        st.warning(f"Preview map is unavailable: {exc}")

    context = dict(summary)
    context["display_band"] = int(band_index)
    set_imagery_context(context)


def render(settings: Settings) -> None:
    page_header(
        "Satellite Imagery",
        "Inspect prepared imagery, band metadata, acquisition tags, and map previews.",
    )
    records = raster_files(settings)
    existing_tab, upload_tab = st.tabs(["Existing imagery", "Upload imagery"])

    with existing_tab:
        selected = select_file(
            records,
            settings,
            label="Imagery raster",
            key="imagery-existing",
        )
        if selected is not None:
            try:
                _show_raster(str(selected.path), settings)
            except GeoAIError as exc:
                error_message("Unable to load the selected imagery", exc)

    with upload_tab:
        uploaded = st.file_uploader(
            "Upload a GeoTIFF",
            type=["tif", "tiff"],
            key="imagery-upload",
        )
        if uploaded is not None:
            maximum = max_upload_bytes(settings)
            if uploaded.size > maximum:
                st.error(
                    f"Upload is {uploaded.size / (1024 * 1024):.1f} MB; "
                    f"the limit is {settings.max_upload_mb} MB."
                )
            elif st.button("Save imagery upload", type="primary", key="save-imagery"):
                try:
                    path = prepare_raster_upload(uploaded.name, uploaded.getvalue(), settings)
                    st.success(f"Saved upload to `{relative_label(path, settings)}`.")
                except GeoAIError as exc:
                    error_message("Unable to save the uploaded imagery", exc)


__all__ = ["render"]
