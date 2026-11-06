"""Change-detection processing, maps, and transition analysis."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import streamlit as st
from streamlit_folium import st_folium

from config.settings import Settings
from dashboard.common import (
    CHANGE_RESULT_KEY,
    error_message,
    page_header,
    relative_label,
    select_file,
)
from dashboard.data_access import change_files, classification_files
from src.analysis.change_detection import detect_changes
from src.analysis.transition_matrix import CHANGE_CAVEAT
from src.utils.errors import GeoAIError, RasterError
from src.utils.file_utils import safe_filename
from src.visualization.charts import transition_heatmap
from src.visualization.map_visualizer import change_map, classification_map


def _run_change_detection(
    earlier: Path,
    later: Path,
    output_path: Path,
    *,
    alignment: str,
    window_size: int,
) -> None:
    try:
        result = detect_changes(
            earlier,
            later,
            output_path,
            alignment=alignment,
            window_size=window_size,
        )
    except GeoAIError as exc:
        error_message("Change detection failed", exc)
        return
    st.session_state[CHANGE_RESULT_KEY] = result
    st.success(
        f"Wrote change raster to `{output_path.name}`. "
        "Open Exports to download the GeoTIFF and transition CSV."
    )


def _show_result(result: object) -> None:
    matrix = result.matrix
    st.subheader("Change summary")
    columns = st.columns(4)
    columns[0].metric("Changed area", f"{matrix.changed_area_km2:,.3f} km²")
    columns[1].metric("Changed share", f"{matrix.changed_percentage:.3f}%")
    columns[2].metric("Changed pixels", f"{matrix.changed_pixels:,}")
    columns[3].metric("Valid pixels", f"{matrix.valid_pixels:,}")
    st.write(
        f"**Earlier:** `{result.earlier_path.name}` → "
        f"**Later:** `{result.later_path.name}` "
        f"({result.alignment.replace('_', ' ')})"
    )

    st.warning(CHANGE_CAVEAT)

    earlier_tab, later_tab, change_tab = st.tabs(
        ["Earlier classification", "Later classification", "Change map"]
    )
    try:
        with earlier_tab:
            st_folium(
                classification_map(result.earlier_path),
                width=900,
                height=460,
                key="change-earlier-map",
            )
        with later_tab:
            st_folium(
                classification_map(result.later_path),
                width=900,
                height=460,
                key="change-later-map",
            )
        with change_tab:
            st_folium(
                change_map(result.output_path),
                width=900,
                height=460,
                key="change-output-map",
            )
    except RasterError as exc:
        st.warning(f"One or more change maps are unavailable: {exc}")

    st.subheader("Transition matrix")
    st.plotly_chart(transition_heatmap(matrix), use_container_width=True)
    changed = [item.to_dict() for item in matrix.changed_transitions]
    if changed:
        st.dataframe(
            [
                {
                    "from": item["from_name"],
                    "to": item["to_name"],
                    "pixels": item["pixel_count"],
                    "area_km2": round(float(item["area_km2"]), 6),
                    "percent_of_changed": round(
                        float(item["percentage_of_changed"]), 4
                    ),
                }
                for item in changed
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No changed pixels were found between these classifications.")

    with st.expander("Full transition table", expanded=False):
        st.dataframe(
            [
                {
                    "from": item.from_name,
                    "to": item.to_name,
                    "pixels": item.pixel_count,
                    "area_km2": round(item.area_km2, 6),
                    "percent_of_valid": round(item.percentage_of_valid, 4),
                    "changed": item.changed,
                }
                for item in matrix.transitions
            ],
            use_container_width=True,
            hide_index=True,
        )


def render(settings: Settings) -> None:
    page_header(
        "Change Detection",
        "Compare two classifications and quantify class transitions and changed area.",
    )
    records = classification_files(settings)
    if len(records) < 2:
        st.info(
            "At least two classification rasters are required. Upload or create a "
            "second classification first."
        )
    else:
        labels = [
            f"{relative_label(record.path, settings)} · {record.modified_utc:%Y-%m-%d}"
            for record in records
        ]
        with st.form("change-detection-form"):
            first, second = st.columns(2)
            with first:
                earlier_index = st.selectbox(
                    "Earlier classification",
                    options=list(range(len(records))),
                    index=min(1, len(records) - 1),
                    format_func=lambda value: labels[value],
                )
            with second:
                later_index = st.selectbox(
                    "Later classification",
                    options=list(range(len(records))),
                    index=0,
                    format_func=lambda value: labels[value],
                )
            controls = st.columns(2)
            alignment = controls[0].selectbox(
                "Grid alignment",
                options=["strict", "reproject"],
                help=(
                    "Strict requires identical grids. Reproject warps the later "
                    "classification onto the earlier grid with nearest-neighbour "
                    "resampling."
                ),
            )
            window_size = int(
                controls[1].number_input(
                    "Window size (pixels)",
                    min_value=128,
                    max_value=8192,
                    value=1024,
                    step=128,
                )
            )
            submitted = st.form_submit_button(
                "Run change detection",
                type="primary",
            )

        if submitted:
            earlier = records[int(earlier_index)].path
            later = records[int(later_index)].path
            if earlier == later:
                st.error("Select two different classification rasters.")
            else:
                stem = (
                    f"{safe_filename(earlier.stem)}_{safe_filename(later.stem)}_"
                    f"{uuid4().hex[:8]}"
                )
                output = settings.change_detection_dir / f"change_{stem}.tif"
                _run_change_detection(
                    earlier,
                    later,
                    output,
                    alignment=alignment,
                    window_size=window_size,
                )

    result = st.session_state.get(CHANGE_RESULT_KEY)
    if result is not None:
        _show_result(result)
    else:
        outputs = change_files(settings)
        with st.expander("Existing change rasters", expanded=not outputs):
            selected = select_file(
                outputs,
                settings,
                label="Change raster",
                key="change-existing-output",
            )
            if selected is not None:
                try:
                    st_folium(
                        change_map(selected.path),
                        width=900,
                        height=460,
                        key="existing-change-map",
                    )
                except RasterError as exc:
                    st.warning(f"Change map is unavailable: {exc}")
                st.caption(
                    "Transition statistics require running change detection in this "
                    "session or selecting the source classifications above."
                )


__all__ = ["render"]
