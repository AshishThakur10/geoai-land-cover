"""Classification display, area statistics, and model metadata."""

from __future__ import annotations

import streamlit as st
from streamlit_folium import st_folium

from config.settings import Settings
from dashboard.common import (
    cached_area_statistics,
    cached_model_metadata,
    cached_raster_summary,
    error_message,
    file_token,
    page_header,
    relative_label,
    select_file,
    set_classification_context,
    set_model_context,
)
from dashboard.data_access import (
    classification_files,
    max_upload_bytes,
    model_metadata_files,
    prepare_raster_upload,
)
from src.utils.errors import GeoAIError, RasterError
from src.visualization.charts import class_distribution_chart
from src.visualization.map_visualizer import classification_map


def _show_classification(path: str, settings: Settings) -> None:
    token = file_token(path)
    summary = cached_raster_summary(path, token)
    statistics = cached_area_statistics(path, token)

    columns = st.columns(4)
    columns[0].metric("Total mapped area", f"{statistics.total_area_km2:,.3f} km²")
    columns[1].metric("Valid pixels", f"{statistics.valid_pixels:,}")
    columns[2].metric("Classes present", len(statistics.classes))
    columns[3].metric("Area method", statistics.area_method.title())
    st.write(f"**Source:** `{relative_label(path, settings)}`")
    st.caption(
        f"Raster grid: {statistics.width:,} × {statistics.height:,} · "
        f"CRS {statistics.crs.to_string()} · NoData {statistics.nodata}"
    )

    try:
        st_folium(
            classification_map(path),
            width=900,
            height=480,
            key=f"classification-map-{token}",
        )
    except RasterError as exc:
        st.warning(f"Classification map is unavailable: {exc}")

    chart_column, table_column = st.columns(2)
    rows = [item.to_dict() for item in statistics.classes]
    with chart_column:
        st.plotly_chart(
            class_distribution_chart(rows),
            use_container_width=True,
        )
    with table_column:
        st.dataframe(
            [
                {
                    "class": item["name"],
                    "code": item["code"],
                    "pixels": item["pixel_count"],
                    "area_km2": round(float(item["area_km2"]), 6),
                    "percent": round(float(item["percentage"]), 4),
                }
                for item in rows
            ],
            use_container_width=True,
            hide_index=True,
        )

    set_classification_context(
        {
            "source_path": path,
            "raster": summary,
            "area_statistics": statistics.to_dict(),
        }
    )
    st.caption(
        "Areas describe classified pixels. They are not a field-verified land-cover inventory."
    )


def _show_model_metadata(settings: Settings) -> None:
    records = model_metadata_files(settings)
    with st.expander("Model metadata", expanded=False):
        selected = select_file(
            records,
            settings,
            label="Model metadata JSON",
            key="classification-model-metadata",
        )
        if selected is None:
            st.caption("No model metadata JSON files were found.")
            set_model_context(None)
            return
        try:
            metadata = cached_model_metadata(str(selected.path), file_token(selected.path))
        except GeoAIError as exc:
            error_message("Unable to read model metadata", exc)
            return

        set_model_context(
            {
                "source_path": str(selected.path),
                "metadata": metadata,
            }
        )
        columns = st.columns(3)
        columns[0].metric("Format", str(metadata.get("format_version", "—")))
        columns[1].metric(
            "Features",
            len(metadata.get("feature_names", []))
            if isinstance(metadata.get("feature_names"), list)
            else "—",
        )
        columns[2].metric("Created", str(metadata.get("created_utc", "—"))[:19])
        st.write(f"**Source:** `{relative_label(selected.path, settings)}`")
        st.json(metadata, expanded=False)


def render(settings: Settings) -> None:
    page_header(
        "Classification",
        "Review classified rasters, class distributions, areas, and model metadata.",
    )
    records = classification_files(settings)
    existing_tab, upload_tab = st.tabs(["Existing classifications", "Upload classification"])

    with existing_tab:
        selected = select_file(
            records,
            settings,
            label="Classification raster",
            key="classification-existing",
        )
        if selected is not None:
            try:
                _show_classification(str(selected.path), settings)
            except GeoAIError as exc:
                error_message("Unable to load the selected classification", exc)

    with upload_tab:
        uploaded = st.file_uploader(
            "Upload a classified GeoTIFF",
            type=["tif", "tiff"],
            key="classification-upload",
        )
        if uploaded is not None:
            maximum = max_upload_bytes(settings)
            if uploaded.size > maximum:
                st.error(
                    f"Upload is {uploaded.size / (1024 * 1024):.1f} MB; "
                    f"the limit is {settings.max_upload_mb} MB."
                )
            elif st.button(
                "Save classification upload",
                type="primary",
                key="save-classification",
            ):
                try:
                    path = prepare_raster_upload(uploaded.name, uploaded.getvalue(), settings)
                    st.success(f"Saved upload to `{relative_label(path, settings)}`.")
                except GeoAIError as exc:
                    error_message("Unable to save the uploaded classification", exc)

    _show_model_metadata(settings)


__all__ = ["render"]
