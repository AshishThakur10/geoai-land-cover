from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.training.evaluation import evaluate_random_forest
from src.training.model_manager import (
    METADATA_FORMAT_VERSION,
    ModelRunMetadata,
    load_model,
    save_model,
)
from src.training.random_forest import RandomForestConfig, train_random_forest
from src.utils.errors import DataValidationError, ModelError
from tests.fixtures import synthetic_ml_samples


def _trained_and_evaluation():
    samples = synthetic_ml_samples()
    validation_indices = np.arange(samples.n_samples).reshape(4, 50)[:, ::5].ravel()
    train_indices = np.setdiff1d(np.arange(samples.n_samples), validation_indices)
    trained = train_random_forest(
        samples.take(train_indices),
        RandomForestConfig(n_estimators=11, max_depth=5, random_state=9, n_jobs=1),
    )
    validation = samples.take(validation_indices)
    return trained, validation, evaluate_random_forest(trained, validation)


def test_save_and_load_model_reproduces_predictions(tmp_path: Path) -> None:
    trained, validation, evaluation = _trained_and_evaluation()
    run_metadata = ModelRunMetadata(
        input_paths=("data/raw/sentinel2/B02.tif", "data/raw/sentinel2/B04.tif"),
        image_dates=("2021-07-01", "2021-07-11"),
        aoi_path="data/raw/aoi/study.geojson",
        label_path="data/raw/worldcover/worldcover_2021.tif",
        study_area="Synthetic test area",
        notes="Synthetic test run",
    )

    artifacts = save_model(
        trained,
        tmp_path / "models" / "random_forest.joblib",
        run_metadata=run_metadata,
        evaluation=evaluation,
    )
    loaded = load_model(artifacts.model_path)

    assert artifacts.model_path.exists()
    assert artifacts.metadata_path.exists()
    assert artifacts.metadata["format_version"] == METADATA_FORMAT_VERSION
    assert loaded.feature_names == trained.feature_names
    assert loaded.classes == trained.classes
    assert np.array_equal(loaded.predict(validation.X), trained.predict(validation.X))
    assert np.array_equal(loaded.predict(validation.X), loaded.model.predict(validation.X))


def test_saved_metadata_records_reproducibility_and_metrics(tmp_path: Path) -> None:
    trained, _, evaluation = _trained_and_evaluation()
    run_metadata = ModelRunMetadata(
        input_paths=("scene.tif",),
        image_dates=("2021-07-01",),
        aoi_path="aoi.geojson",
        label_path="labels.tif",
    )

    artifacts = save_model(
        trained,
        tmp_path / "random_forest.joblib",
        run_metadata=run_metadata,
        evaluation=evaluation,
    )
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))

    assert metadata["feature_names"] == list(trained.feature_names)
    assert metadata["classes"] == list(trained.classes)
    assert metadata["parameters"]["n_estimators"] == 11
    assert metadata["seed"] == 9
    assert metadata["training"]["n_samples"] == trained.training_samples
    assert metadata["training"]["class_counts"] == {"10": 40, "30": 40, "40": 40, "80": 40}
    assert metadata["run"] == {
        "input_paths": ["scene.tif"],
        "image_dates": ["2021-07-01"],
        "aoi_path": "aoi.geojson",
        "label_path": "labels.tif",
        "study_area": None,
        "notes": None,
    }
    assert metadata["metrics"]["n_samples"] == 40
    assert 0.0 <= metadata["metrics"]["metrics"]["accuracy"] <= 1.0
    assert metadata["software_versions"]["python"]
    assert metadata["software_versions"]["scikit-learn"]
    assert len(metadata["model"]["sha256"]) == 64
    assert artifacts.to_dict()["model_path"] == str(artifacts.model_path)


def test_save_model_without_evaluation_records_null_metrics(tmp_path: Path) -> None:
    trained, _, _ = _trained_and_evaluation()

    artifacts = save_model(trained, tmp_path / "model.joblib")
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))

    assert metadata["metrics"] is None
    assert metadata["run"]["input_paths"] == []
    assert loaded_feature_names(artifacts.model_path) == trained.feature_names


def loaded_feature_names(path: Path) -> tuple[str, ...]:
    return load_model(path).feature_names


def test_save_model_rejects_invalid_paths_and_metadata(tmp_path: Path) -> None:
    trained, _, _ = _trained_and_evaluation()

    with pytest.raises(DataValidationError, match="Unsupported model type"):
        save_model(trained, tmp_path / "model.pkl")
    with pytest.raises(DataValidationError, match="Unsupported metadata type"):
        save_model(trained, tmp_path / "model.joblib", metadata_path=tmp_path / "meta.txt")
    with pytest.raises(DataValidationError, match="must be a ModelRunMetadata"):
        save_model(trained, tmp_path / "model.joblib", run_metadata={})  # type: ignore[arg-type]


def test_load_model_reports_missing_and_invalid_files(tmp_path: Path) -> None:
    trained, _, _ = _trained_and_evaluation()
    artifacts = save_model(trained, tmp_path / "model.joblib")

    with pytest.raises(DataValidationError, match="does not exist"):
        load_model(tmp_path / "missing.joblib")
    unsupported = tmp_path / "model.txt"
    unsupported.write_text("not a model", encoding="utf-8")
    with pytest.raises(DataValidationError, match="Unsupported model type"):
        load_model(unsupported)
    with pytest.raises(DataValidationError, match="metadata does not exist"):
        load_model(
            artifacts.model_path,
            metadata_path=tmp_path / "missing.metadata.json",
        )

    artifacts.metadata_path.write_text("not-json", encoding="utf-8")
    with pytest.raises(ModelError, match="Unable to read model metadata"):
        load_model(artifacts.model_path)


def test_load_model_validates_format_feature_and_class_contracts(tmp_path: Path) -> None:
    trained, _, _ = _trained_and_evaluation()
    artifacts = save_model(trained, tmp_path / "model.joblib")
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))

    metadata["format_version"] = 99
    artifacts.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ModelError, match="Unsupported metadata format"):
        load_model(artifacts.model_path)

    metadata["format_version"] = METADATA_FORMAT_VERSION
    metadata["feature_names"] = ["only-one"]
    artifacts.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ModelError, match="expects .* features but metadata lists"):
        load_model(artifacts.model_path)

    metadata["feature_names"] = list(trained.feature_names)
    metadata["classes"] = [10, 30]
    artifacts.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ModelError, match="classes do not match"):
        load_model(artifacts.model_path)


def test_load_model_rejects_non_rf_and_corrupt_joblib(tmp_path: Path) -> None:
    trained, _, _ = _trained_and_evaluation()
    artifacts = save_model(trained, tmp_path / "model.joblib")
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))

    wrong_model = tmp_path / "wrong.joblib"
    joblib.dump({"not": "a model"}, wrong_model)
    wrong_metadata = tmp_path / "wrong.metadata.json"
    wrong_metadata.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ModelError, match="does not contain a RandomForestClassifier"):
        load_model(wrong_model)

    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"not a joblib file")
    corrupt_metadata = tmp_path / "corrupt.metadata.json"
    corrupt_metadata.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ModelError, match="Unable to load model"):
        load_model(corrupt)
