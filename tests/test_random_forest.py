from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from config.settings import load_settings
from src.training.random_forest import (
    RandomForestConfig,
    TrainedRandomForest,
    build_random_forest,
    train_random_forest,
)
from src.utils.errors import DataValidationError, ModelError
from tests.fixtures import synthetic_ml_samples


def _validation_indices(n_samples: int, n_per_class: int = 50) -> np.ndarray:
    return np.arange(n_samples).reshape(4, n_per_class)[:, ::5].ravel()


def test_config_from_settings(tmp_path: Path) -> None:
    settings = load_settings(
        {
            "GEOAI_N_ESTIMATORS": "7",
            "GEOAI_MAX_DEPTH": "3",
            "GEOAI_RANDOM_SEED": "11",
            "GEOAI_CLASS_WEIGHT": "balanced",
        },
        project_root=tmp_path,
    )

    config = RandomForestConfig.from_settings(settings)

    assert config.n_estimators == 7
    assert config.max_depth == 3
    assert config.random_state == 11
    assert config.class_weight == "balanced"
    assert config.to_dict()["random_state"] == 11


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_estimators": 0},
        {"n_estimators": 1.5},
        {"max_depth": 0},
        {"min_samples_leaf": 0},
        {"max_features": "auto"},
        {"max_features": 0.0},
        {"max_features": 1.5},
        {"max_features": True},
        {"class_weight": "custom"},
        {"random_state": -1},
        {"random_state": 1.5},
        {"n_jobs": 0},
        {"n_jobs": 1.5},
    ],
)
def test_config_rejects_invalid_parameters(kwargs: dict[str, object]) -> None:
    with pytest.raises(DataValidationError):
        RandomForestConfig(**kwargs)  # type: ignore[arg-type]


def test_config_normalises_max_features() -> None:
    assert RandomForestConfig(max_features=" SQRT ").max_features == "sqrt"
    assert RandomForestConfig(max_features=0.5).max_features == 0.5
    assert RandomForestConfig(max_features=None).max_features is None


def test_build_random_forest_uses_config() -> None:
    config = RandomForestConfig(n_estimators=5, max_depth=2, random_state=7, n_jobs=1)

    model = build_random_forest(config)

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 5
    assert model.max_depth == 2
    assert model.random_state == 7
    assert model.n_jobs == 1
    assert not hasattr(model, "classes_")


def test_train_random_forest_returns_provenance() -> None:
    samples = synthetic_ml_samples()
    config = RandomForestConfig(n_estimators=7, max_depth=4, random_state=3, n_jobs=1)

    trained = train_random_forest(samples, config)

    assert isinstance(trained, TrainedRandomForest)
    assert trained.training_samples == 200
    assert trained.classes == (10, 30, 40, 80)
    assert dict(trained.training_class_counts) == {10: 50, 30: 50, 40: 50, 80: 50}
    assert trained.feature_names == samples.feature_names
    assert trained.label_source == samples.label_source
    assert not trained.used_balanced_weights
    assert set(trained.to_dict()) >= {
        "parameters",
        "seed",
        "classes",
        "training_class_counts",
    }


def test_training_is_reproducible_with_same_seed() -> None:
    samples = synthetic_ml_samples()
    config = RandomForestConfig(n_estimators=9, max_depth=5, random_state=19, n_jobs=1)

    first = train_random_forest(samples, config)
    second = train_random_forest(samples, config)

    assert np.array_equal(first.predict(samples.X), second.predict(samples.X))


def test_training_can_use_balanced_weights() -> None:
    samples = synthetic_ml_samples()

    trained = train_random_forest(
        samples,
        RandomForestConfig(n_estimators=5, random_state=1, n_jobs=1),
        use_balanced_weights=True,
    )

    assert trained.used_balanced_weights
    assert trained.to_dict()["used_balanced_weights"] is True


def test_training_rejects_too_few_samples_or_classes() -> None:
    samples = synthetic_ml_samples()

    with pytest.raises(ModelError, match="At least two training samples"):
        train_random_forest(samples.take(np.array([0])))
    with pytest.raises(ModelError, match="At least two land-cover classes"):
        train_random_forest(samples.take(np.arange(0, 10)))


def test_trained_model_predict_validates_input() -> None:
    samples = synthetic_ml_samples()
    trained = train_random_forest(
        samples,
        RandomForestConfig(n_estimators=3, random_state=1, n_jobs=1),
    )

    with pytest.raises(ModelError, match="2D feature matrix"):
        trained.predict(np.ones((2, 2)))
    with pytest.raises(ModelError, match="non-finite"):
        bad = np.ones((2, 6), dtype=np.float32)
        bad[0, 0] = np.nan
        trained.predict(bad)


def test_trained_model_rejects_feature_name_mismatch() -> None:
    samples = synthetic_ml_samples()
    trained = train_random_forest(
        samples,
        RandomForestConfig(n_estimators=3, random_state=1, n_jobs=1),
    )

    with pytest.raises(DataValidationError, match="feature"):
        TrainedRandomForest(
            model=trained.model,
            config=trained.config,
            feature_names=("only",),
            training_samples=trained.training_samples,
            training_class_counts=trained.training_class_counts,
            label_source=trained.label_source,
            used_balanced_weights=False,
        )


def test_validation_indices_cover_every_class() -> None:
    samples = synthetic_ml_samples()

    validation = samples.take(_validation_indices(samples.n_samples))

    assert validation.n_samples == 40
    assert set(validation.class_counts) == {10, 30, 40, 80}
