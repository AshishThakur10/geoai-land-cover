"""Download real products and build a consistent analysis report."""

from __future__ import annotations

import streamlit as st

from config.settings import Settings
from dashboard.common import (
    CHANGE_RESULT_KEY,
    CLASSIFICATION_CONTEXT_KEY,
    EVALUATION_CONTEXT_KEY,
    IMAGERY_CONTEXT_KEY,
    SPECTRAL_CONTEXT_KEY,
    STUDY_AREA_CONTEXT_KEY,
    active_aoi_path,
    cached_area_statistics,
    cached_evaluation_report,
    cached_vector_metadata,
    error_message,
    file_bytes,
    file_token,
    page_header,
    relative_label,
    select_file,
    set_classification_context,
    set_evaluation_context,
    written_file_bytes,
)
from dashboard.data_access import (
    classification_files,
    evaluation_reports,
)
from src.analysis.area_statistics import save_area_statistics_csv
from src.analysis.transition_matrix import save_transition_matrix_csv
from src.data.vector_loader import geodesic_area_km2, load_aoi
from src.reports.report_generator import (
    analysis_report_json,
    analysis_report_markdown,
    build_analysis_report,
    classification_report_csv,
    confusion_matrix_csv,
)
from src.utils.errors import GeoAIError


def _download(
    label: str,
    data: bytes | str,
    *,
    file_name: str,
    mime: str,
    key: str,
) -> None:
    st.download_button(
        label,
        data=data,
        file_name=file_name,
        mime=mime,
        key=key,
        use_container_width=True,
    )


def _study_area_export(settings: Settings) -> dict[str, object] | None:
    path = active_aoi_path(settings)
    st.subheader("Study-area boundary")
    if path is None:
        st.caption("No active study area. Select one in Study Area.")
        return None
    try:
        metadata = cached_vector_metadata(str(path), file_token(path))
        frame = load_aoi(path)
        area = geodesic_area_km2(frame)
        context = metadata.to_dict()
        context["area_km2"] = area
        st.write(f"`{relative_label(path, settings)}` · {area:,.3f} km²")
        data = frame.to_crs("EPSG:4326").to_json()
        _download(
            "Download study-area GeoJSON",
            data,
            file_name=f"{path.stem}.geojson",
            mime="application/geo+json",
            key="download-aoi",
        )
        return context
    except GeoAIError as exc:
        error_message("Unable to prepare the study-area export", exc)
        return None


def _classification_export(settings: Settings) -> dict[str, object] | None:
    st.subheader("Classification products")
    records = classification_files(settings)
    if not records:
        st.caption("No classification rasters are available.")
        return None
    current = st.session_state.get(CLASSIFICATION_CONTEXT_KEY)
    current_path = current.get("source_path") if isinstance(current, dict) else None
    default_index = next(
        (index for index, record in enumerate(records) if str(record.path) == current_path),
        0,
    )
    selected = select_file(
        records,
        settings,
        label="Classification to export",
        key="export-classification",
        default_index=default_index,
    )
    if selected is None:
        return None
    try:
        statistics = cached_area_statistics(str(selected.path), file_token(selected.path))
    except GeoAIError as exc:
        error_message("Unable to compute area statistics for export", exc)
        return None

    st.dataframe(
        [
            {
                "class": item.name,
                "code": item.code,
                "pixels": item.pixel_count,
                "area_km2": round(item.area_km2, 6),
                "percent": round(item.percentage, 4),
            }
            for item in statistics.classes
        ],
        use_container_width=True,
        hide_index=True,
    )
    columns = st.columns(2)
    with columns[0]:
        _download(
            "Download classified GeoTIFF",
            file_bytes(selected.path),
            file_name=selected.path.name,
            mime="image/tiff",
            key="download-classification-tif",
        )
    with columns[1]:
        csv_bytes = written_file_bytes(
            save_area_statistics_csv,
            statistics,
            suffix=".csv",
        )
        _download(
            "Download area statistics CSV",
            csv_bytes,
            file_name=f"{selected.path.stem}_area_statistics.csv",
            mime="text/csv",
            key="download-classification-csv",
        )

    context = {
        "source_path": str(selected.path),
        "raster": {
            "crs": statistics.crs.to_string(),
            "width": statistics.width,
            "height": statistics.height,
            "nodata": statistics.nodata,
        },
        "area_statistics": statistics.to_dict(),
    }
    set_classification_context(context)
    return context


def _change_export() -> dict[str, object] | None:
    st.subheader("Change-detection products")
    result = st.session_state.get(CHANGE_RESULT_KEY)
    if result is None:
        st.caption(
            "No change analysis is active in this session. Run change detection first."
        )
        return None
    st.write(
        f"Changed area: **{result.matrix.changed_area_km2:,.3f} km²** · "
        f"**{result.matrix.changed_percentage:.3f}%** of valid pixels"
    )
    columns = st.columns(2)
    with columns[0]:
        _download(
            "Download change GeoTIFF",
            file_bytes(result.output_path),
            file_name=result.output_path.name,
            mime="image/tiff",
            key="download-change-tif",
        )
    with columns[1]:
        csv_bytes = written_file_bytes(
            save_transition_matrix_csv,
            result.matrix,
            suffix=".csv",
        )
        _download(
            "Download transition matrix CSV",
            csv_bytes,
            file_name=f"{result.output_path.stem}_transitions.csv",
            mime="text/csv",
            key="download-transition-csv",
        )
    return result.to_dict()


def _evaluation_export(settings: Settings) -> dict[str, object] | None:
    st.subheader("Model-evaluation tables")
    records = evaluation_reports(settings)
    if not records:
        st.caption("No evaluation reports are available.")
        return None
    current = st.session_state.get(EVALUATION_CONTEXT_KEY)
    current_path = current.get("source_path") if isinstance(current, dict) else None
    default_index = next(
        (index for index, record in enumerate(records) if str(record.path) == current_path),
        0,
    )
    selected = select_file(
        records,
        settings,
        label="Evaluation report to export",
        key="export-evaluation",
        default_index=default_index,
    )
    if selected is None:
        return None
    try:
        report = cached_evaluation_report(str(selected.path), file_token(selected.path))
        confusion = confusion_matrix_csv(report)
        per_class = classification_report_csv(report)
    except GeoAIError as exc:
        error_message("Unable to prepare evaluation exports", exc)
        return None

    columns = st.columns(2)
    with columns[0]:
        _download(
            "Download confusion matrix CSV",
            confusion,
            file_name=f"{selected.path.stem}_confusion_matrix.csv",
            mime="text/csv",
            key="download-confusion-csv",
        )
    with columns[1]:
        _download(
            "Download per-class metrics CSV",
            per_class,
            file_name=f"{selected.path.stem}_classification_report.csv",
            mime="text/csv",
            key="download-per-class-csv",
        )
    set_evaluation_context({"source_path": str(selected.path), "report": report})
    return report


def _analysis_report(
    *,
    study_area: dict[str, object] | None,
    classification: dict[str, object] | None,
    change: dict[str, object] | None,
    evaluation: dict[str, object] | None,
) -> None:
    st.subheader("Analysis report")
    study_context = st.session_state.get(STUDY_AREA_CONTEXT_KEY)
    imagery = st.session_state.get(IMAGERY_CONTEXT_KEY)
    spectral = st.session_state.get(SPECTRAL_CONTEXT_KEY)
    sections = {
        "study_area": study_context or study_area,
        "imagery": imagery,
        "spectral_analysis": spectral,
        "classification": classification,
        "model_evaluation": evaluation,
        "change_detection": change,
    }
    chosen = {key: value for key, value in sections.items() if value is not None}
    if not chosen:
        st.caption(
            "Open a data section or run an analysis before building a report. "
            "The report only includes real selected or computed data."
        )
        return
    try:
        report = build_analysis_report(
            study_area=chosen.get("study_area"),
            imagery=chosen.get("imagery"),
            spectral_analysis=chosen.get("spectral_analysis"),
            classification=chosen.get("classification"),
            model_evaluation=chosen.get("model_evaluation"),
            change_detection=chosen.get("change_detection"),
        )
        payload = analysis_report_json(report)
        markdown = analysis_report_markdown(report)
    except GeoAIError as exc:
        error_message("Unable to build the analysis report", exc)
        return

    st.caption(
        "The report includes these sections: " + ", ".join(sorted(chosen)) + "."
    )
    columns = st.columns(2)
    with columns[0]:
        _download(
            "Download analysis report JSON",
            payload,
            file_name="land_cover_analysis_report.json",
            mime="application/json",
            key="download-report-json",
        )
    with columns[1]:
        _download(
            "Download analysis report Markdown",
            markdown,
            file_name="land_cover_analysis_report.md",
            mime="text/markdown",
            key="download-report-markdown",
        )
    with st.expander("Preview report", expanded=False):
        st.markdown(markdown)


def render(settings: Settings) -> None:
    page_header(
        "Exports",
        "Download real output files and a reproducible report for the current analysis.",
    )
    study_area = _study_area_export(settings)
    st.divider()
    classification = _classification_export(settings)
    st.divider()
    change = _change_export()
    st.divider()
    evaluation = _evaluation_export(settings)
    st.divider()
    _analysis_report(
        study_area=study_area,
        classification=classification,
        change=change,
        evaluation=evaluation,
    )
    st.caption(
        "Downloads preserve source rasters and use the same statistics shown in the "
        "dashboard. Reports are not substitutes for field validation."
    )


__all__ = ["render"]
