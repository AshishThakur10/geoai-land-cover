from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from config.settings import Settings, load_settings
from src.training.evaluation import evaluate_predictions, save_evaluation_report
from tests.fixtures import (
    synthetic_worldcover_codes,
    write_synthetic_aoi_geojson,
    write_synthetic_raster,
    write_synthetic_worldcover,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _populate(settings: Settings) -> None:
    codes = synthetic_worldcover_codes()
    earlier = write_synthetic_worldcover(settings.classifications_dir / "earlier.tif", codes=codes)
    changed = codes.copy()
    changed[0, 0] = 80
    write_synthetic_worldcover(settings.classifications_dir / "later.tif", codes=changed)
    old = time.time() - 100
    os.utime(earlier, (old, old))

    write_synthetic_raster(settings.sentinel2_dir / "scene" / "scene_B04.tif", count=4)
    write_synthetic_aoi_geojson(settings.aoi_dir / "study.geojson")
    result = evaluate_predictions(
        np.array([10, 10, 40, 40]),
        np.array([10, 40, 40, 40]),
        feature_names=("B02", "NDVI"),
        feature_importances=(0.6, 0.4),
    )
    save_evaluation_report(result, settings.reports_dir / "evaluation.json")
    metadata = settings.models_dir / "model.metadata.json"
    metadata.write_text(
        json.dumps({"format_version": 1, "feature_names": ["B02", "NIR"]}),
        encoding="utf-8",
    )


def _app_test(tmp_path: Path) -> AppTest:
    data_dir = tmp_path / "data"
    root = tmp_path
    settings = load_settings(
        {
            "GEOAI_DATA_DIR": str(data_dir),
            "GEOAI_LOG_FILE": str(tmp_path / "logs" / "geoai.log"),
        },
        project_root=root,
    )
    settings.ensure_directories()
    _populate(settings)
    return AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30)


def test_dashboard_home_and_navigation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GEOAI_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GEOAI_LOG_FILE", str(tmp_path / "logs" / "geoai.log"))
    app = _app_test(tmp_path).run()

    assert not app.exception
    assert app.title[0].value == "GeoAI Land Cover"

    app.sidebar.radio[0].set_value("Classification").run()
    assert not app.exception
    assert app.title[0].value == "Classification"
    assert any("Total mapped area" in metric.label for metric in app.metric)

    app.sidebar.radio[0].set_value("Exports").run()
    assert not app.exception
    assert app.title[0].value == "Exports"
    download_labels = [item.label for item in app.get("download_button")]
    assert "Download classified GeoTIFF" in download_labels


def test_dashboard_study_area_activation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GEOAI_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GEOAI_LOG_FILE", str(tmp_path / "logs" / "geoai.log"))
    app = _app_test(tmp_path).run()

    app.sidebar.radio[0].set_value("Study Area").run()
    assert not app.exception
    assert app.title[0].value == "Study Area"
    assert any("Bounding box" in caption.value for caption in app.caption)

    activate = next(
        button for button in app.button if button.label == "Use as active study area"
    )
    activate.click().run()
    assert not app.exception
    assert any("Active study area set to" in message.value for message in app.success)

    app.sidebar.radio[0].set_value("Exports").run()
    assert not app.exception
    labels = [item.label for item in app.get("download_button")]
    assert "Download study-area GeoJSON" in labels


def test_dashboard_handles_empty_project(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GEOAI_DATA_DIR", str(tmp_path / "empty-data"))
    monkeypatch.setenv("GEOAI_LOG_FILE", str(tmp_path / "logs" / "empty.log"))
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "GeoAI Land Cover"
    assert any("No study area is active" in info.value for info in app.info)


def test_dashboard_imagery_spectral_and_evaluation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GEOAI_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GEOAI_LOG_FILE", str(tmp_path / "logs" / "geoai.log"))
    app = _app_test(tmp_path).run()

    app.sidebar.radio[0].set_value("Satellite Imagery").run()
    assert not app.exception
    assert app.title[0].value == "Satellite Imagery"
    assert any(metric.label == "Bands" for metric in app.metric)

    app.sidebar.radio[0].set_value("Spectral Analysis").run()
    assert not app.exception
    compute = next(
        button for button in app.button if button.label == "Compute spectral indices"
    )
    compute.click().run()
    assert not app.exception
    assert any(metric.label == "Valid pixels" for metric in app.metric)

    red_band = next(
        selectbox for selectbox in app.selectbox if selectbox.label.startswith("Red")
    )
    red_band.set_value(1).run()
    assert not app.exception
    assert any("band mapping changed" in info.value.lower() for info in app.info)
    assert not any(metric.label == "Valid pixels" for metric in app.metric)

    app.sidebar.radio[0].set_value("Model Evaluation").run()
    assert not app.exception
    assert app.title[0].value == "Model Evaluation"
    assert any(metric.label == "Accuracy" for metric in app.metric)


def test_dashboard_change_detection_and_exports(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GEOAI_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GEOAI_LOG_FILE", str(tmp_path / "logs" / "geoai.log"))
    app = _app_test(tmp_path).run()

    app.sidebar.radio[0].set_value("Change Detection").run()
    assert not app.exception
    run = next(
        button for button in app.button if button.label == "Run change detection"
    )
    run.click().run()
    assert not app.exception
    assert any(metric.label == "Changed area" for metric in app.metric)
    assert any("transition" in caption.value.lower() for caption in app.caption)

    app.sidebar.radio[0].set_value("Exports").run()
    assert not app.exception
    labels = [item.label for item in app.get("download_button")]
    assert "Download change GeoTIFF" in labels
    assert "Download transition matrix CSV" in labels
    assert "Download analysis report JSON" in labels
