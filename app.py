"""Streamlit entry point.

Phase 1 intentionally exposes only project setup information. Data loading,
preprocessing, model training, classification, change detection, and the full
dashboard are added in later phases.
"""

from __future__ import annotations

import streamlit as st

from config.land_cover_classes import ESA_WORLDCOVER_CLASSES, get_class_legend
from config.settings import get_settings
from src.utils.logging_config import get_logger, setup_logging

settings = get_settings()
settings.ensure_directories()
setup_logging(level=settings.log_level, log_file=settings.log_file)
logger = get_logger(__name__)

st.set_page_config(
    page_title="GeoAI Land Cover",
    layout="wide",
)

st.title("GeoAI Land Cover Classification and Change Detection")
st.caption("Sentinel-2 Level-2A imagery · ESA WorldCover 2021 reference labels · Random Forest")

st.info(
    "Phase 1 (project setup) is complete. Data loading, spectral analysis, "
    "model training, classification, change detection, and the full dashboard "
    "are implemented in subsequent phases."
)

left, right = st.columns(2)
with left:
    st.subheader("Project status")
    st.write(f"Python package version: `{__import__('src').__version__}`")
    st.write(f"Log level: `{settings.log_level}`")
    st.write(f"Data directory: `{settings.data_dir}`")
    st.write(f"Output directory: `{settings.outputs_dir}`")
    st.write(f"Random seed: `{settings.random_seed}`")
    st.write(f"Random Forest trees: `{settings.n_estimators}`")

with right:
    st.subheader("Reference classes")
    st.write(f"{len(ESA_WORLDCOVER_CLASSES)} ESA WorldCover 2021 classes configured.")
    st.dataframe(get_class_legend(), use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Reference labels are derived from ESA WorldCover 2021 and are not treated as "
    "unquestioned ground truth. Classification accuracy must be validated before "
    "results are used for decision-making."
)

logger.info("Rendered Phase 1 landing page")
