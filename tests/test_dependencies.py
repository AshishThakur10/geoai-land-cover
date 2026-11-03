from __future__ import annotations

import sys

import pytest

import config.land_cover_classes
import config.settings
import src
from src.utils import errors, file_utils, logging_config, validation


def test_python_version_is_supported() -> None:
    assert sys.version_info[:2] == (3, 11)


def test_core_geospatial_and_ml_dependencies_import() -> None:
    import geopandas
    import numpy
    import pandas
    import rasterio
    import shapely
    import sklearn

    assert numpy.__version__
    assert pandas.__version__
    assert rasterio.__version__
    assert geopandas.__version__
    assert shapely.__version__
    assert sklearn.__version__


def test_dashboard_and_plotting_dependencies_import() -> None:
    import folium
    import matplotlib
    import plotly
    import seaborn
    import streamlit

    assert matplotlib.__version__
    assert seaborn.__version__
    assert plotly.__version__
    assert folium.__version__
    assert streamlit.__version__


def test_project_packages_import() -> None:
    assert src.__version__ == "0.1.0"
    assert config.settings.PROJECT_ROOT.is_dir()
    assert callable(errors.GeoAIError)
    assert callable(file_utils.ensure_directory)
    assert callable(logging_config.setup_logging)
    assert callable(validation.validate_uploaded_file)


def test_rasterio_has_bundled_gdal() -> None:
    import rasterio

    assert rasterio.__gdal_version__


@pytest.mark.parametrize("module", ["geopandas", "rasterio", "sklearn", "streamlit"])
def test_required_library_is_installed(module: str) -> None:
    assert __import__(module) is not None
