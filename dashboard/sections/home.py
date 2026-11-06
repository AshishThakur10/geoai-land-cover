"""Dashboard home and project inventory."""

from __future__ import annotations

import streamlit as st

from config.settings import Settings
from dashboard import data_access as data
from dashboard.common import (
    active_aoi_path,
    cached_area_statistics,
    cached_vector_metadata,
    error_message,
    file_token,
    page_header,
    records_table,
    relative_label,
)
from src.data.vector_loader import geodesic_area_km2, load_aoi


def render(settings: Settings) -> None:
    page_header(
        "GeoAI Land Cover",
        "Explore prepared imagery, classifications, model evaluation, and change analysis.",
    )

    imagery = data.raster_files(settings)
    classifications = data.classification_files(settings)
    changes = data.change_files(settings)
    aois = data.vector_files(settings)
    evaluations = data.evaluation_reports(settings)
    models = data.model_metadata_files(settings)

    columns = st.columns(5)
    columns[0].metric("Imagery rasters", len(imagery))
    columns[1].metric("Classifications", len(classifications))
    columns[2].metric("Change outputs", len(changes))
    columns[3].metric("Study areas", len(aois))
    columns[4].metric("Evaluation reports", len(evaluations))

    st.subheader("Current analysis")
    aoi_path = active_aoi_path(settings)
    if aoi_path is None:
        st.info(
            "No study area is active. Open **Study Area** to select or upload a boundary."
        )
    else:
        try:
            metadata = cached_vector_metadata(str(aoi_path), file_token(aoi_path))
            frame = load_aoi(aoi_path)
            area_km2 = geodesic_area_km2(frame)
            st.write(f"**Study area:** `{relative_label(aoi_path, settings)}`")
            st.write(
                f"{metadata.feature_count} feature(s) · {area_km2:,.3f} km² · "
                f"CRS {metadata.crs.to_string() if metadata.crs is not None else 'missing'}"
            )
        except Exception as exc:
            error_message("Unable to summarize the active study area", exc)

    if classifications:
        latest = classifications[0]
        try:
            statistics = cached_area_statistics(str(latest.path), file_token(latest.path))
            st.metric("Latest classified area", f"{statistics.total_area_km2:,.3f} km²")
            st.caption(f"Source: `{relative_label(latest.path, settings)}`")
        except Exception as exc:
            error_message("Unable to compute the latest classified area", exc)

    latest_records = sorted(
        classifications + changes,
        key=lambda item: item.modified_utc,
        reverse=True,
    )
    if latest_records:
        latest_output = latest_records[0]
        st.write(
            f"**Latest output:** `{relative_label(latest_output.path, settings)}` "
            f"({latest_output.modified_utc:%Y-%m-%d %H:%M} UTC)"
        )
    else:
        st.caption("No classification or change outputs have been written yet.")

    st.subheader("Available data")
    inventory = [
        ("Imagery rasters", imagery),
        ("Classification rasters", classifications),
        ("Change rasters", changes),
        ("Study-area vectors", aois),
        ("Model metadata", models),
        ("Evaluation reports", evaluations),
    ]
    st.dataframe(
        [
            {
                "category": label,
                "files": len(records),
                "latest": (
                    relative_label(records[0].path, settings) if records else "—"
                ),
                "latest_modified_utc": (
                    records[0].modified_utc.strftime("%Y-%m-%d %H:%M:%S")
                    if records
                    else "—"
                ),
            }
            for label, records in inventory
        ],
        use_container_width=True,
        hide_index=True,
    )

    recent = sorted(
        {record.path: record for category in inventory for record in category[1]}.values(),
        key=lambda item: item.modified_utc,
        reverse=True,
    )
    with st.expander("Recent files", expanded=False):
        records_table(recent[:20], settings)

    st.caption(
        "All maps and statistics are derived from the selected real files. "
        "Model predictions are not field-verified land-cover observations."
    )


__all__ = ["render"]
