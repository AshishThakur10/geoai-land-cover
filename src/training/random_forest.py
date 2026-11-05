"""Configurable Random Forest training for labeled spectral samples."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from config.settings import VALID_CLASS_WEIGHTS, Settings
from src.training.sample_generator import LabeledSamples
from src.utils.errors import DataValidationError, ModelError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

VALID_MAX_FEATURES = frozenset({"sqrt", "log2"})


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(f"{name} must be an integer, got {value!r}")
    if value < 1:
        raise DataValidationError(f"{name} must be >= 1, got {value}")
    return value


@dataclass(frozen=True)
class RandomForestConfig:
    """Validated Random Forest parameters."""

    n_estimators: int = 150
    max_depth: int | None = None
    min_samples_leaf: int = 1
    max_features: str | float | None = "sqrt"
    class_weight: str | None = "balanced_subsample"
    random_state: int = 42
    n_jobs: int = -1

    def __post_init__(self) -> None:
        _positive_int(self.n_estimators, "n_estimators")
        _positive_int(self.min_samples_leaf, "min_samples_leaf")
        if self.max_depth is not None:
            _positive_int(self.max_depth, "max_depth")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, int):
            raise DataValidationError(
                f"random_state must be an integer, got {self.random_state!r}"
            )
        if self.random_state < 0:
            raise DataValidationError(f"random_state must be >= 0, got {self.random_state}")
        if isinstance(self.n_jobs, bool) or not isinstance(self.n_jobs, int):
            raise DataValidationError(f"n_jobs must be an integer, got {self.n_jobs!r}")
        if self.n_jobs == 0:
            raise DataValidationError("n_jobs must not be 0")

        if isinstance(self.max_features, bool):
            raise DataValidationError("max_features must be 'sqrt', 'log2', a float, or None")
        if isinstance(self.max_features, float):
            if not 0.0 < self.max_features <= 1.0:
                raise DataValidationError(
                    f"max_features float must be in (0, 1], got {self.max_features}"
                )
        elif isinstance(self.max_features, str):
            normalized = self.max_features.strip().lower()
            if normalized not in VALID_MAX_FEATURES:
                allowed = ", ".join(sorted(VALID_MAX_FEATURES))
                raise DataValidationError(
                    f"max_features must be one of {allowed}, a float, or None"
                )
            object.__setattr__(self, "max_features", normalized)
        elif self.max_features is not None:
            raise DataValidationError(
                f"max_features must be 'sqrt', 'log2', a float, or None, "
                f"got {self.max_features!r}"
            )

        if self.class_weight is not None and self.class_weight not in VALID_CLASS_WEIGHTS:
            allowed = ", ".join(sorted(VALID_CLASS_WEIGHTS))
            raise DataValidationError(
                f"class_weight must be one of {allowed}, or None; got {self.class_weight!r}"
            )

    @classmethod
    def from_settings(cls, settings: Settings) -> RandomForestConfig:
        """Build a configuration from application settings."""
        return cls(
            n_estimators=settings.n_estimators,
            max_depth=settings.max_depth,
            class_weight=settings.class_weight,
            random_state=settings.random_seed,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }


def build_random_forest(config: RandomForestConfig | None = None) -> RandomForestClassifier:
    """Create an unfitted Random Forest classifier."""
    resolved = config or RandomForestConfig()
    return RandomForestClassifier(
        n_estimators=resolved.n_estimators,
        max_depth=resolved.max_depth,
        min_samples_leaf=resolved.min_samples_leaf,
        max_features=resolved.max_features,
        class_weight=resolved.class_weight,
        random_state=resolved.random_state,
        n_jobs=resolved.n_jobs,
    )


@dataclass(frozen=True)
class TrainedRandomForest:
    """A fitted Random Forest plus training provenance."""

    model: RandomForestClassifier
    config: RandomForestConfig
    feature_names: tuple[str, ...]
    training_samples: int
    training_class_counts: Mapping[int, int]
    label_source: Path | None
    used_balanced_weights: bool

    def __post_init__(self) -> None:
        if not hasattr(self.model, "classes_"):
            raise DataValidationError("The Random Forest model must be fitted")
        if self.model.n_features_in_ != len(self.feature_names):
            raise DataValidationError(
                f"Model expects {self.model.n_features_in_} features but "
                f"{len(self.feature_names)} names were supplied"
            )
        if sum(self.training_class_counts.values()) != self.training_samples:
            raise DataValidationError("Training class counts do not match sample count")

    @property
    def classes(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.model.classes_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class codes with shape validation."""
        values = np.asarray(X)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ModelError(
                f"Expected a 2D feature matrix with {len(self.feature_names)} columns, "
                f"got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ModelError("Prediction features contain non-finite values")
        return self.model.predict(values)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": type(self.model).__name__,
            "feature_names": list(self.feature_names),
            "classes": list(self.classes),
            "training_samples": self.training_samples,
            "training_class_counts": {
                str(code): count for code, count in sorted(self.training_class_counts.items())
            },
            "label_source": str(self.label_source) if self.label_source else None,
            "used_balanced_weights": self.used_balanced_weights,
            "parameters": self.config.to_dict(),
            "seed": self.config.random_state,
        }


def train_random_forest(
    samples: LabeledSamples,
    config: RandomForestConfig | None = None,
    *,
    use_balanced_weights: bool = False,
) -> TrainedRandomForest:
    """Train a Random Forest on labeled spectral samples.

    The caller is responsible for passing the training side of a spatially
    separated split. Validation must remain untouched until evaluation.

    Args:
        samples: Training samples with finite features and valid class codes.
        config: Optional Random Forest configuration.
        use_balanced_weights: Also pass inverse-frequency sample weights.
            Random Forest ``class_weight`` remains active independently.

    Raises:
        ModelError: If there are too few samples/classes or fitting fails.
    """
    resolved = config or RandomForestConfig()
    if samples.n_samples < 2:
        raise ModelError("At least two training samples are required")
    if samples.n_features != len(samples.feature_names):
        raise ModelError("Training feature names do not match the feature matrix")

    class_counts = samples.class_counts
    if len(class_counts) < 2:
        raise ModelError("At least two land-cover classes are required for training")

    sample_weight = samples.balanced_sample_weights() if use_balanced_weights else None
    model = build_random_forest(resolved)
    try:
        model.fit(samples.X, samples.y, sample_weight=sample_weight)
    except (TypeError, ValueError) as exc:
        raise ModelError(f"Random Forest training failed: {exc}") from exc

    trained = TrainedRandomForest(
        model=model,
        config=resolved,
        feature_names=samples.feature_names,
        training_samples=samples.n_samples,
        training_class_counts=MappingProxyType(dict(sorted(class_counts.items()))),
        label_source=samples.label_source,
        used_balanced_weights=use_balanced_weights,
    )

    logger.info(
        "Trained Random Forest with %d trees on %d samples across %d classes",
        resolved.n_estimators,
        trained.training_samples,
        len(trained.classes),
    )
    return trained
