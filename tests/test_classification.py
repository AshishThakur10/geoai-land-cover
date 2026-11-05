from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import rasterio

from src.classification.predictor import (
    classify_feature_raster,
    classify_feature_stack,
)
from src.features.feature_extraction import FeatureStack, save_feature_stack
from src.training.model_manager import load_model, save_model
from src.training.random_forest import RandomForestConfig, train_random_forest
from src.utils.errors import DataValidationError, ModelError, RasterError
from tests.fixtures import (
    RASTER_CRS,
    default_transform,
    synthetic_ml_samples,
    synthetic_worldcover_codes,
    write_synthetic_raster,
)


def _trained_model():
    samples = synthetic_ml_samples()
    return train_random_forest(
        samples,
        RandomForestConfig(n_estimators=25, max_depth=8, random_state=4, n_jobs=1),
    )


def _feature_stack() -> tuple[FeatureStack, np.ndarray]:
    samples = synthetic_ml_samples()
    codes = synthetic_worldcover_codes(width=15, height=12)
    data = np.empty((samples.n_features, 12, 15), dtype=np.float32)
    for code in np.unique(codes):
        class_mean = samples.X[samples.y == code].mean(axis=0)
        data[:, codes == code] = class_mean[:, np.newaxis]
    data[:, 0, 0] = np.nan
    data[:, 5, 7] = np.nan
    expected = codes.copy()
    expected[0, 0] = 0
    expected[5, 7] = 0
    valid_mask = np.all(np.isfinite(data), axis=0)
    features = FeatureStack(
        data=data,
        feature_names=samples.feature_names,
        transform=default_transform(),
        crs=rasterio.crs.CRS.from_string(RASTER_CRS),
        valid_mask=valid_mask,
        band_names=("B02", "B03", "B04", "B08"),
    )
    return features, expected


def _read_classification(path: Path) -> np.ndarray:
    with rasterio.open(path) as dataset:
        return dataset.read(1)


def test_classify_feature_stack_writes_georeferenced_classes(tmp_path: Path) -> None:
    trained = _trained_model()
    features, expected = _feature_stack()

    result = classify_feature_stack(
        trained,
        features,
        tmp_path / "classification.tif",
        window_size=4,
    )

    assert result.output_path.exists()
    assert np.array_equal(_read_classification(result.output_path), expected)
    assert result.valid_pixels == 178
    assert result.nodata_pixels == 2
    assert result.class_counts[10] == int(np.count_nonzero(expected == 10))
    assert result.class_counts[30] == int(np.count_nonzero(expected == 30))
    assert result.class_counts[40] == int(np.count_nonzero(expected == 40))
    assert result.class_counts[80] == int(np.count_nonzero(expected == 80))
    assert result.to_dict()["legend"]


def test_classification_preserves_grid_and_writes_legend(tmp_path: Path) -> None:
    trained = _trained_model()
    features, _ = _feature_stack()

    result = classify_feature_stack(trained, features, tmp_path / "classification.tif")

    with rasterio.open(result.output_path) as dataset:
        assert dataset.crs == features.crs
        assert dataset.transform == features.transform
        assert (dataset.width, dataset.height) == (features.width, features.height)
        assert dataset.dtypes == ("uint8",)
        assert dataset.nodata == 0
        assert dataset.descriptions == ("Land-cover class code",)
        assert "Tree Cover" in dataset.tags()["CLASS_LEGEND"]
        assert dataset.colormap(1)[10][:3] == (0, 100, 0)


def test_classification_is_deterministic(tmp_path: Path) -> None:
    trained = _trained_model()
    features, _ = _feature_stack()

    first = classify_feature_stack(
        trained,
        features,
        tmp_path / "first.tif",
        window_size=5,
    )
    second = classify_feature_stack(
        trained,
        features,
        tmp_path / "second.tif",
        window_size=7,
    )

    assert np.array_equal(
        _read_classification(first.output_path),
        _read_classification(second.output_path),
    )


def test_classify_feature_raster_matches_in_memory_classification(tmp_path: Path) -> None:
    trained = _trained_model()
    features, expected = _feature_stack()
    feature_path = save_feature_stack(features, tmp_path / "features.tif")
    artifacts = save_model(trained, tmp_path / "model.joblib")
    loaded = load_model(artifacts.model_path)

    result = classify_feature_raster(
        loaded,
        feature_path,
        tmp_path / "classification.tif",
        window_size=4,
    )

    assert np.array_equal(_read_classification(result.output_path), expected)
    assert result.feature_names == trained.feature_names
    assert result.valid_pixels == 178


def test_classify_feature_raster_allows_explicit_feature_names(tmp_path: Path) -> None:
    trained = _trained_model()
    features, expected = _feature_stack()
    feature_path = save_feature_stack(features, tmp_path / "features.tif")

    result = classify_feature_raster(
        trained,
        feature_path,
        tmp_path / "classification.tif",
        feature_names=trained.feature_names,
    )

    assert np.array_equal(_read_classification(result.output_path), expected)


def test_classify_feature_raster_without_descriptions_uses_model_order(tmp_path: Path) -> None:
    trained = _trained_model()
    features, expected = _feature_stack()
    feature_path = save_feature_stack(features, tmp_path / "features.tif")
    no_descriptions = tmp_path / "features_without_descriptions.tif"
    with rasterio.open(feature_path) as source:
        data = source.read()
        profile = source.profile
    with rasterio.open(no_descriptions, "w", **profile) as destination:
        destination.write(data)

    result = classify_feature_raster(
        trained,
        no_descriptions,
        tmp_path / "classification.tif",
    )

    assert np.array_equal(_read_classification(result.output_path), expected)


def test_classification_rejects_invalid_options_and_contracts(tmp_path: Path) -> None:
    trained = _trained_model()
    features, _ = _feature_stack()

    with pytest.raises(DataValidationError, match="window_size"):
        classify_feature_stack(trained, features, tmp_path / "a.tif", window_size=0)
    with pytest.raises(DataValidationError, match="window_size"):
        classify_feature_stack(trained, features, tmp_path / "a.tif", window_size=True)
    with pytest.raises(DataValidationError, match="nodata"):
        classify_feature_stack(trained, features, tmp_path / "a.tif", nodata=1.5)
    with pytest.raises(DataValidationError, match="nodata"):
        classify_feature_stack(trained, features, tmp_path / "a.tif", nodata=10)
    with pytest.raises(DataValidationError, match="nodata"):
        classify_feature_stack(trained, features, tmp_path / "a.tif", nodata=256)
    with pytest.raises(DataValidationError, match="Unsupported classification"):
        classify_feature_stack(trained, features, tmp_path / "a.png")

    mismatched = replace(
        features,
        feature_names=tuple(f"feature_{index}" for index in range(features.count)),
    )
    with pytest.raises(ModelError, match="Feature names/order"):
        classify_feature_stack(trained, mismatched, tmp_path / "a.tif")


def test_classify_feature_raster_rejects_invalid_sources(tmp_path: Path) -> None:
    trained = _trained_model()
    features, _ = _feature_stack()
    feature_path = save_feature_stack(features, tmp_path / "features.tif")

    with pytest.raises(ModelError, match="Feature names/order"):
        classify_feature_raster(
            trained,
            feature_path,
            tmp_path / "a.tif",
            feature_names=tuple(f"feature_{index}" for index in range(6)),
        )

    wrong_count = write_synthetic_raster(
        tmp_path / "wrong_count.tif",
        count=3,
        dtype="float32",
        nodata=np.nan,
    )
    with pytest.raises(ModelError, match="3 bands"):
        classify_feature_raster(trained, wrong_count, tmp_path / "b.tif")

    no_crs = write_synthetic_raster(
        tmp_path / "no_crs.tif",
        count=6,
        crs=None,
        dtype="float32",
        nodata=np.nan,
    )
    with pytest.raises(RasterError, match="CRS"):
        classify_feature_raster(trained, no_crs, tmp_path / "c.tif")


def test_classification_marks_all_invalid_features_as_nodata(tmp_path: Path) -> None:
    trained = _trained_model()
    features, _ = _feature_stack()
    invalid = replace(
        features,
        data=np.full_like(features.data, np.nan),
        valid_mask=np.zeros((features.height, features.width), dtype=bool),
    )

    result = classify_feature_stack(trained, invalid, tmp_path / "empty.tif")

    assert result.valid_pixels == 0
    assert result.nodata_pixels == result.total_pixels
    assert np.all(_read_classification(result.output_path) == 0)
