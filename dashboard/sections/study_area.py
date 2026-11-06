"""Study-area selection, upload, validation, and mapping."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit_folium import st_folium

from config.settings import Settings
from dashboard.common import (
    active_aoi_path,
    cached_aoi,
    cached_vector_metadata,
    error_message,
    file_token,
    page_header,
    relative_label,
    set_active_aoi,
)
from dashboard.data_access import max_upload_bytes, prepare_vector_upload, vector_files
from src.data.vector_loader import geodesic_area_km2
from src.utils.errors import GeoAIError
from src.visualization.map_visualizer import boundary_map


def _describe(path: str | Path, settings: Settings) -> dict[str, object]:
    resolved = str(path)
    metadata = cached_vector_metadata(resolved, file_token(resolved))
    frame = cached_aoi(resolved, file_token(resolved))
    area_km2 = geodesic_area_km2(frame)

    columns = st.columns(4)
    columns[0].metric("Features", metadata.feature_count)
    columns[1].metric("Area", f"{area_km2:,.3f} km²")
    columns[2].metric(
        "CRS",
        metadata.crs.to_string() if metadata.crs is not None else "Missing",
    )
    columns[3].metric("Geometry types", len(metadata.geometry_types))

    st.write(f"**Source:** `{relative_label(resolved, settings)}`")
    st.dataframe(
        [
            {
                "geometry_type": geometry_type,
                "feature_count": int(frame.geom_type.eq(geometry_type).sum()),
            }
            for geometry_type in metadata.geometry_types
        ],
        use_container_width=True,
        hide_index=True,
    )
    st_folium(
        boundary_map(frame),
        width=900,
        height=460,
        key=f"aoi-map-{file_token(resolved)}",
    )
    st.caption(
        "Study-area area is an approximate WGS84 geodesic measurement, not a "
        "cadastral or legally binding area."
    )

    context = metadata.to_dict()
    context["area_km2"] = area_km2
    context["source"] = "dashboard"
    return context


def _activate(path: str | Path, settings: Settings, context: dict[str, object]) -> None:
    set_active_aoi(path, metadata=context)
    st.success(f"Active study area set to `{relative_label(path, settings)}`.")


def render(settings: Settings) -> None:
    page_header(
        "Study Area",
        "Select or upload the AOI used to interpret imagery and land-cover outputs.",
    )
    active = active_aoi_path(settings)
    if active is not None:
        st.info(f"Active study area: `{relative_label(active, settings)}`")

    records = vector_files(settings)
    existing_tab, upload_tab = st.tabs(["Existing study areas", "Upload study area"])

    with existing_tab:
        active_index = next(
            (index for index, record in enumerate(records) if record.path == active),
            0,
        )
        if records:
            labels = [
                f"{relative_label(record.path, settings)} · {record.size_label}"
                for record in records
            ]
            selected_index = st.selectbox(
                "Study-area file",
                options=list(range(len(records))),
                index=active_index,
                format_func=lambda value: labels[value],
                key="study-area-existing",
            )
            selected = records[int(selected_index)]
            try:
                context = _describe(selected.path, settings)
                if st.button("Use as active study area", type="primary", key="activate-aoi"):
                    _activate(selected.path, settings, context)
            except GeoAIError as exc:
                error_message("Unable to load the selected study area", exc)
        else:
            st.info(
                "No study-area vector files were found. Upload a GeoJSON, GeoPackage, "
                "or zipped Shapefile."
            )

    with upload_tab:
        uploaded = st.file_uploader(
            "Upload a study-area vector",
            type=["geojson", "json", "gpkg", "shp", "zip"],
            help=(
                "GeoJSON and GeoPackage are single files. For Shapefiles, upload a ZIP "
                "archive containing the .shp, .shx, .dbf, and .prj files."
            ),
            key="study-area-upload",
        )
        if uploaded is not None:
            maximum = max_upload_bytes(settings)
            if uploaded.size > maximum:
                st.error(
                    f"Upload is {uploaded.size / (1024 * 1024):.1f} MB; "
                    f"the limit is {settings.max_upload_mb} MB."
                )
            elif st.button("Validate and use upload", type="primary", key="save-aoi"):
                try:
                    path = prepare_vector_upload(uploaded.name, uploaded.getvalue(), settings)
                    context = _describe(path, settings)
                    _activate(path, settings, context)
                except GeoAIError as exc:
                    error_message("Unable to use the uploaded study area", exc)

    if active is not None and st.button("Clear active study area", key="clear-aoi"):
        set_active_aoi(None)
        st.rerun()


__all__ = ["render"]
