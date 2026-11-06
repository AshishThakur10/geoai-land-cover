"""Streamlit dashboard entry point for the GeoAI land-cover workflow."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

import src
from config.settings import Settings, get_settings
from dashboard.common import active_aoi_path, relative_label, safe_render
from dashboard.sections import (
    change_detection,
    classification,
    exports,
    home,
    imagery,
    model_evaluation,
    spectral,
    study_area,
)
from src.utils.logging_config import get_logger, setup_logging

settings = get_settings()
settings.ensure_directories()
setup_logging(level=settings.log_level, log_file=settings.log_file)
logger = get_logger(__name__)

st.set_page_config(
    page_title="GeoAI Land Cover",
    page_icon=None,
    layout="wide",
)

RENDERERS: dict[str, Callable[[Settings], None]] = {
    "Home": home.render,
    "Study Area": study_area.render,
    "Satellite Imagery": imagery.render,
    "Spectral Analysis": spectral.render,
    "Classification": classification.render,
    "Model Evaluation": model_evaluation.render,
    "Change Detection": change_detection.render,
    "Exports": exports.render,
}

with st.sidebar:
    st.title("GeoAI Land Cover")
    st.caption("Sentinel-2 · ESA WorldCover reference labels · Random Forest")
    section = st.radio(
        "Navigation",
        options=list(RENDERERS),
        key="dashboard-section",
    )
    st.divider()
    active = active_aoi_path(settings)
    st.caption(
        "Active study area: "
        + (
            f"`{relative_label(active, settings)}`"
            if active is not None
            else "none selected"
        )
    )
    with st.expander("Project paths", expanded=False):
        st.write(f"Package version: `{src.__version__}`")
        st.write(f"Data: `{settings.data_dir}`")
        st.write(f"Outputs: `{settings.outputs_dir}`")
        st.write(f"Models: `{settings.models_dir}`")
        st.write(f"Upload limit: `{settings.max_upload_mb} MB`")

safe_render(f"{section} section", lambda: RENDERERS[section](settings))

st.divider()
st.caption(
    "Classifications and change maps are model-derived products. WorldCover labels are "
    "reference data, not unquestioned ground truth, and all results require validation "
    "before operational use."
)
logger.info("Rendered dashboard section: %s", section)
