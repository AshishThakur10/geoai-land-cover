"""Persist Random Forest models with reproducibility metadata."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

from src.training.evaluation import EvaluationResult
from src.training.random_forest import TrainedRandomForest
from src.utils.errors import DataValidationError, ModelError
from src.utils.file_utils import ensure_directory
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

MODEL_SUFFIXES: frozenset[str] = frozenset({".joblib"})
METADATA_SUFFIXES: frozenset[str] = frozenset({".json"})
METADATA_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ModelRunMetadata:
    """Caller-supplied provenance for inputs, dates, and study area."""

    input_paths: tuple[str, ...] = ()
    image_dates: tuple[str, ...] = ()
    aoi_path: str | None = None
    label_path: str | None = None
    study_area: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_paths",
            tuple(str(value) for value in self.input_paths),
        )
        object.__setattr__(
            self,
            "image_dates",
            tuple(str(value) for value in self.image_dates),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "input_paths": list(self.input_paths),
            "image_dates": list(self.image_dates),
            "aoi_path": self.aoi_path,
            "label_path": self.label_path,
            "study_area": self.study_area,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ModelArtifacts:
    """Paths and metadata written for a saved model."""

    model_path: Path
    metadata_path: Path
    metadata: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_path": str(self.model_path),
            "metadata_path": str(self.metadata_path),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LoadedModel:
    """A persisted model with its metadata and validated feature contract."""

    model: RandomForestClassifier
    metadata: Mapping[str, object]
    model_path: Path
    metadata_path: Path

    @property
    def feature_names(self) -> tuple[str, ...]:
        names = self.metadata.get("feature_names")
        if not isinstance(names, list):
            raise ModelError("Model metadata does not contain feature_names")
        return tuple(str(name) for name in names)

    @property
    def classes(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.model.classes_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class codes with the persisted feature contract enforced."""
        values = np.asarray(X)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ModelError(
                f"Expected a 2D feature matrix with {len(self.feature_names)} columns, "
                f"got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ModelError("Prediction features contain non-finite values")
        return self.model.predict(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def save_model(
    trained: TrainedRandomForest,
    model_path: str | Path,
    *,
    run_metadata: ModelRunMetadata | None = None,
    evaluation: EvaluationResult | None = None,
    metadata_path: str | Path | None = None,
) -> ModelArtifacts:
    """Save a fitted Random Forest and a JSON metadata sidecar.

    The sidecar records feature names/order, class codes, hyperparameters, seed,
    training counts and label source, optional run inputs/dates/AOI, optional
    held-out metrics, and software versions.

    Only trusted model files should be loaded later, because joblib
    deserialisation can execute code.
    """
    destination = Path(model_path).expanduser().resolve()
    if destination.suffix.lower() not in MODEL_SUFFIXES:
        raise DataValidationError(
            f"Unsupported model type {destination.suffix!r}; expected .joblib"
        )
    metadata_destination = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path is not None
        else destination.with_suffix(".metadata.json")
    )
    if metadata_destination.suffix.lower() not in METADATA_SUFFIXES:
        raise DataValidationError(
            f"Unsupported metadata type {metadata_destination.suffix!r}; expected .json"
        )
    if metadata_destination == destination:
        raise DataValidationError("Model and metadata paths must differ")
    if run_metadata is not None and not isinstance(run_metadata, ModelRunMetadata):
        raise DataValidationError("run_metadata must be a ModelRunMetadata instance")

    ensure_directory(destination.parent)
    ensure_directory(metadata_destination.parent)

    try:
        joblib.dump(trained.model, destination, compress=3)
    except (OSError, TypeError, ValueError) as exc:
        raise ModelError(f"Unable to save model {destination}: {exc}") from exc

    metadata: dict[str, object] = {
        "format_version": METADATA_FORMAT_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "model": {
            "type": type(trained.model).__name__,
            "file": destination.name,
            "sha256": _sha256(destination),
        },
        "feature_names": list(trained.feature_names),
        "classes": list(trained.classes),
        "parameters": trained.config.to_dict(),
        "seed": trained.config.random_state,
        "training": {
            "n_samples": trained.training_samples,
            "class_counts": {
                str(code): count
                for code, count in sorted(trained.training_class_counts.items())
            },
            "label_source": str(trained.label_source) if trained.label_source else None,
            "used_balanced_weights": trained.used_balanced_weights,
        },
        "run": run_metadata.to_dict() if run_metadata else ModelRunMetadata().to_dict(),
        "metrics": evaluation.to_dict() if evaluation is not None else None,
        "software_versions": _software_versions(),
    }

    try:
        metadata_destination.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise ModelError(f"Unable to write model metadata {metadata_destination}: {exc}") from exc

    logger.info("Saved model to %s and metadata to %s", destination, metadata_destination)
    return ModelArtifacts(
        model_path=destination,
        metadata_path=metadata_destination,
        metadata=MappingProxyType(metadata),
    )


def load_model(
    model_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
) -> LoadedModel:
    """Load a trusted joblib model and validate its metadata contract."""
    source = Path(model_path).expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise DataValidationError(f"Model file does not exist: {source}")
    if source.suffix.lower() not in MODEL_SUFFIXES:
        raise DataValidationError(
            f"Unsupported model type {source.suffix!r}; expected .joblib"
        )
    metadata_source = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path is not None
        else source.with_suffix(".metadata.json")
    )
    if not metadata_source.exists() or not metadata_source.is_file():
        raise DataValidationError(f"Model metadata does not exist: {metadata_source}")

    try:
        metadata = json.loads(metadata_source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(f"Unable to read model metadata {metadata_source}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ModelError("Model metadata must be a JSON object")
    if metadata.get("format_version") != METADATA_FORMAT_VERSION:
        raise ModelError(
            f"Unsupported metadata format version: {metadata.get('format_version')!r}"
        )

    try:
        model = joblib.load(source)
    except Exception as exc:
        raise ModelError(f"Unable to load model {source}: {exc}") from exc
    if not isinstance(model, RandomForestClassifier):
        raise ModelError(f"Model file does not contain a RandomForestClassifier: {source}")
    if not hasattr(model, "classes_"):
        raise ModelError("Loaded model is not fitted")

    feature_names = metadata.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise ModelError("Model metadata must contain a non-empty feature_names list")
    if model.n_features_in_ != len(feature_names):
        raise ModelError(
            f"Model expects {model.n_features_in_} features but metadata lists "
            f"{len(feature_names)}"
        )
    expected_classes = metadata.get("classes")
    if expected_classes != [int(value) for value in model.classes_]:
        raise ModelError("Metadata classes do not match the loaded model")

    logger.info("Loaded model from %s", source)
    return LoadedModel(
        model=model,
        metadata=MappingProxyType(metadata),
        model_path=source,
        metadata_path=metadata_source,
    )
