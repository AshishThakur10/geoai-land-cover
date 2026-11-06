from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import rasterio

from config.settings import Settings, load_settings
from dashboard import data_access as data
from src.training.evaluation import evaluate_predictions, save_evaluation_report
from src.utils.errors import DataValidationError
from tests.fixtures import (
    write_synthetic_aoi_geojson,
    write_synthetic_raster,
    write_synthetic_worldcover,
)


def _settings(tmp_path: Path) -> Settings:
    settings = load_settings({}, project_root=tmp_path)
    settings.ensure_directories()
    return settings


def _evaluation_report(path: Path) -> Path:
    result = evaluate_predictions(
        np.array([10, 10, 40, 40]),
        np.array([10, 40, 40, 40]),
        feature_names=("B02", "NDVI"),
        feature_importances=(0.6, 0.4),
    )
    return save_evaluation_report(result, path)


def test_store_upload_validates_and_sanitises(tmp_path: Path) -> None:
    directory = tmp_path / "uploads"
    stored = data.store_upload(
        "../../unsafe name.tif",
        b"synthetic",
        directory=directory,
        suffixes={".tif"},
        maximum_bytes=100,
    )

    assert stored.parent == directory
    assert stored.name.startswith("unsafe_name_")
    assert stored.suffix == ".tif"
    assert stored.read_bytes() == b"synthetic"

    with pytest.raises(DataValidationError, match="Unsupported upload type"):
        data.store_upload(
            "notes.txt",
            b"x",
            directory=directory,
            suffixes={".tif"},
            maximum_bytes=100,
        )
    with pytest.raises(DataValidationError, match="empty"):
        data.store_upload(
            "empty.tif",
            b"",
            directory=directory,
            suffixes={".tif"},
            maximum_bytes=100,
        )
    with pytest.raises(DataValidationError, match="limit"):
        data.store_upload(
            "large.tif",
            b"x" * 101,
            directory=directory,
            suffixes={".tif"},
            maximum_bytes=100,
        )


def test_prepare_vector_upload_extracts_zip(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    aoi = write_synthetic_aoi_geojson(tmp_path / "source" / "aoi.geojson")
    archive = tmp_path / "aoi.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(aoi, arcname="nested/aoi.geojson")

    extracted = data.prepare_vector_upload(
        "study area.zip",
        archive.read_bytes(),
        settings,
    )

    assert extracted.suffix == ".geojson"
    assert extracted.name == "aoi.geojson"
    assert extracted.is_file()


def test_vector_zip_rejects_traversal(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    buffer = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(buffer, "w") as handle:
        handle.writestr("../escape.geojson", "{}")

    with pytest.raises(DataValidationError, match="unsafe path"):
        data.prepare_vector_upload("unsafe.zip", buffer.read_bytes(), settings)


def test_vector_zip_requires_one_layer(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    buffer = tmp_path / "multiple.zip"
    with zipfile.ZipFile(buffer, "w") as handle:
        handle.writestr("first.geojson", "{}")
        handle.writestr("second.geojson", "{}")

    with pytest.raises(DataValidationError, match="exactly one vector layer"):
        data.prepare_vector_upload("multiple.zip", buffer.read_bytes(), settings)


def test_discovery_matches_real_output_shapes(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    classification = write_synthetic_worldcover(
        settings.classifications_dir / "classified.tif"
    )
    imagery = write_synthetic_worldcover(
        settings.sentinel2_dir / "scene" / "scene_B04.tif"
    )
    aoi = write_synthetic_aoi_geojson(settings.aoi_dir / "study.geojson")
    evaluation = _evaluation_report(settings.reports_dir / "evaluation.json")
    metadata_path = settings.models_dir / "model.metadata.json"
    metadata_path.write_text(
        json.dumps({"format_version": 1, "feature_names": ["B02", "B03"]}),
        encoding="utf-8",
    )

    assert [record.path for record in data.classification_files(settings)] == [
        classification.resolve()
    ]
    assert imagery.resolve() in {record.path for record in data.raster_files(settings)}
    assert aoi.resolve() in {record.path for record in data.vector_files(settings)}
    assert [record.path for record in data.evaluation_reports(settings)] == [
        evaluation.resolve()
    ]
    assert [record.path for record in data.model_metadata_files(settings)] == [
        metadata_path.resolve()
    ]
    assert data.read_evaluation_report(evaluation)["metrics"]["accuracy"] == pytest.approx(
        0.75
    )
    assert data.read_model_metadata(metadata_path)["feature_names"] == ["B02", "B03"]


def test_raster_summary_uses_actual_tags_only(tmp_path: Path) -> None:
    path = write_synthetic_worldcover(tmp_path / "class.tif")
    summary = data.raster_summary(path)
    assert summary["acquisition_date"] is None

    with rasterio.open(path, "r+") as dataset:
        dataset.update_tags(acquisition_date="2026-01-02T03:04:05Z")
    summary = data.raster_summary(path)
    assert summary["acquisition_date"] == "2026-01-02T03:04:05Z"
    assert summary["tags"]["acquisition_date"] == "2026-01-02T03:04:05Z"


def test_raster_preview_and_band_validation(tmp_path: Path) -> None:
    path = write_synthetic_raster(tmp_path / "stack.tif", width=40, height=20)
    preview = data.read_band_preview(path, max_size=10)

    assert preview.shape == (5, 10)
    assert preview.dtype == np.uint16
    with pytest.raises(DataValidationError, match="out of range"):
        data.read_band_preview(path, band_index=5)
    with pytest.raises(DataValidationError, match="max_size"):
        data.read_band_preview(path, max_size=0)


def test_read_json_object_is_bounded_and_typed(tmp_path: Path) -> None:
    valid = tmp_path / "valid.json"
    valid.write_text('{"value": 1}', encoding="utf-8")
    assert data.read_json_object(valid) == {"value": 1}

    with pytest.raises(DataValidationError, match="limit"):
        data.read_json_object(valid, maximum_bytes=2)

    array = tmp_path / "array.json"
    array.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(DataValidationError, match="must contain an object"):
        data.read_json_object(array)

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(DataValidationError, match="Unable to read JSON"):
        data.read_json_object(invalid)
