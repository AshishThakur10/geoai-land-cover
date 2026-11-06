"""Shared Streamlit state, cached loaders, and presentation helpers."""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import streamlit as st

from config.settings import Settings
from dashboard.data_access import (
    FileRecord,
    raster_summary,
    read_evaluation_report,
    read_model_metadata,
)
from src.analysis.area_statistics import AreaStatistics, compute_area_statistics
from src.data.data_metadata import VectorMetadata
from src.data.raster_loader import read_band
from src.data.vector_loader import load_aoi, read_vector_metadata
from src.features.spectral_indices import compute_index
from src.utils.errors import GeoAIError

ACTIVE_AOI_KEY = "active_aoi_path"
STUDY_AREA_CONTEXT_KEY = "study_area_context"
IMAGERY_CONTEXT_KEY = "imagery_context"
CLASSIFICATION_CONTEXT_KEY = "classification_context"
MODEL_CONTEXT_KEY = "model_context"
EVALUATION_CONTEXT_KEY = "model_evaluation_context"
CHANGE_RESULT_KEY = "change_result"
SPECTRAL_CONTEXT_KEY = "spectral_context"


def file_token(path: str | Path) -> int:
    try:
        return Path(path).stat().st_mtime_ns
    except OSError:
        return 0


@st.cache_data(show_spinner=False, max_entries=64)
def cached_raster_summary(path: str, token: int) -> dict[str, object]:
    del token
    return raster_summary(path)


@st.cache_data(show_spinner="Computing area statistics...", max_entries=16)
def cached_area_statistics(path: str, token: int) -> AreaStatistics:
    del token
    return compute_area_statistics(path)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_vector_metadata(path: str, token: int) -> VectorMetadata:
    del token
    return read_vector_metadata(path)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_aoi(path: str, token: int) -> gpd.GeoDataFrame:
    del token
    return load_aoi(path)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_evaluation_report(path: str, token: int) -> dict[str, object]:
    del token
    return read_evaluation_report(path)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_model_metadata(path: str, token: int) -> dict[str, object]:
    del token
    return read_model_metadata(path)


@st.cache_data(show_spinner="Computing spectral indices...", max_entries=4)
def cached_indices(
    path: str,
    token: int,
    band_mapping: tuple[tuple[str, int], ...],
) -> dict[str, np.ma.MaskedArray]:
    del token
    bands = {
        key: read_band(path, band_index, masked=True)
        for key, band_index in band_mapping
    }
    return {
        name: compute_index(name, bands)
        for name in ("NDVI", "NDWI", "NDBI")
        if name != "NDBI" or "B11" in bands
    }


def page_header(title: str, description: str) -> None:
    st.title(title)
    st.caption(description)


def error_message(context: str, error: Exception) -> None:
    st.error(f"{context}: {error}")


def unexpected_error(context: str, error: Exception) -> None:
    st.error(f"{context}: unexpected error. See the application log for details.")
    st.exception(error)


def relative_label(path: str | Path, settings: Settings) -> str:
    candidate = Path(path)
    try:
        return str(candidate.relative_to(settings.project_root))
    except ValueError:
        return str(candidate)


def record_label(record: FileRecord, settings: Settings) -> str:
    return (
        f"{relative_label(record.path, settings)} · {record.size_label} · "
        f"{record.modified_utc:%Y-%m-%d %H:%M} UTC"
    )


def select_file(
    records: Sequence[FileRecord],
    settings: Settings,
    *,
    label: str,
    key: str,
    default_index: int = 0,
) -> FileRecord | None:
    if not records:
        st.info(f"No files are available for {label.lower()}.")
        return None
    labels = [record_label(record, settings) for record in records]
    index = st.selectbox(
        label,
        options=list(range(len(records))),
        index=min(max(default_index, 0), len(records) - 1),
        format_func=lambda value: labels[value],
        key=key,
    )
    return records[int(index)]


def records_table(records: Sequence[FileRecord], settings: Settings) -> None:
    if not records:
        st.caption("No matching files found.")
        return
    st.dataframe(
        [
            {
                "file": relative_label(record.path, settings),
                "size": record.size_label,
                "modified_utc": record.modified_utc.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for record in records
        ],
        use_container_width=True,
        hide_index=True,
    )


def set_active_aoi(
    path: str | Path | None,
    *,
    metadata: Mapping[str, object] | None = None,
) -> None:
    if path is None:
        st.session_state.pop(ACTIVE_AOI_KEY, None)
        st.session_state.pop(STUDY_AREA_CONTEXT_KEY, None)
        return
    st.session_state[ACTIVE_AOI_KEY] = str(Path(path).resolve())
    if metadata is not None:
        st.session_state[STUDY_AREA_CONTEXT_KEY] = dict(metadata)


def active_aoi_path(settings: Settings) -> Path | None:
    stored = st.session_state.get(ACTIVE_AOI_KEY)
    if stored:
        path = Path(stored)
        if path.is_file():
            return path
    if settings.default_aoi is not None and settings.default_aoi.is_file():
        return settings.default_aoi
    return None


def active_vector_files(settings: Settings) -> list[FileRecord]:
    from dashboard.data_access import vector_files

    return vector_files(settings)


def set_imagery_context(summary: Mapping[str, object] | None) -> None:
    if summary is None:
        st.session_state.pop(IMAGERY_CONTEXT_KEY, None)
    else:
        st.session_state[IMAGERY_CONTEXT_KEY] = dict(summary)


def set_classification_context(value: Mapping[str, object] | None) -> None:
    if value is None:
        st.session_state.pop(CLASSIFICATION_CONTEXT_KEY, None)
    else:
        st.session_state[CLASSIFICATION_CONTEXT_KEY] = dict(value)


def set_model_context(value: Mapping[str, object] | None) -> None:
    if value is None:
        st.session_state.pop(MODEL_CONTEXT_KEY, None)
    else:
        st.session_state[MODEL_CONTEXT_KEY] = dict(value)


def set_evaluation_context(value: Mapping[str, object] | None) -> None:
    if value is None:
        st.session_state.pop(EVALUATION_CONTEXT_KEY, None)
    else:
        st.session_state[EVALUATION_CONTEXT_KEY] = dict(value)


def set_spectral_context(value: Mapping[str, object] | None) -> None:
    if value is None:
        st.session_state.pop(SPECTRAL_CONTEXT_KEY, None)
    else:
        st.session_state[SPECTRAL_CONTEXT_KEY] = dict(value)


def file_bytes(path: str | Path) -> bytes:
    return Path(path).read_bytes()


def written_file_bytes(
    writer: Callable[[Any, Path], Path],
    value: Any,
    *,
    suffix: str,
) -> bytes:
    """Render an existing file writer into bytes without retaining a download copy."""
    with tempfile.TemporaryDirectory(prefix="geoai-download-") as directory:
        output = writer(value, Path(directory) / f"export{suffix}")
        return output.read_bytes()


def safe_render(context: str, render: Callable[[], None]) -> None:
    try:
        render()
    except GeoAIError as exc:
        error_message(context, exc)
    except Exception as exc:  # pragma: no cover - defensive UI boundary
        unexpected_error(context, exc)


__all__ = [
    "ACTIVE_AOI_KEY",
    "CHANGE_RESULT_KEY",
    "CLASSIFICATION_CONTEXT_KEY",
    "EVALUATION_CONTEXT_KEY",
    "IMAGERY_CONTEXT_KEY",
    "MODEL_CONTEXT_KEY",
    "SPECTRAL_CONTEXT_KEY",
    "STUDY_AREA_CONTEXT_KEY",
    "active_aoi_path",
    "active_vector_files",
    "cached_aoi",
    "cached_area_statistics",
    "cached_evaluation_report",
    "cached_indices",
    "cached_model_metadata",
    "cached_raster_summary",
    "cached_vector_metadata",
    "error_message",
    "file_bytes",
    "file_token",
    "page_header",
    "record_label",
    "records_table",
    "relative_label",
    "safe_render",
    "select_file",
    "set_active_aoi",
    "set_classification_context",
    "set_evaluation_context",
    "set_imagery_context",
    "set_model_context",
    "set_spectral_context",
    "unexpected_error",
    "written_file_bytes",
]
