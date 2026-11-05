"""Held-out evaluation metrics for land-cover classifiers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from config.land_cover_classes import VALID_CLASS_CODES, get_class_name
from src.training.random_forest import TrainedRandomForest
from src.training.sample_generator import LabeledSamples
from src.utils.errors import DataValidationError, ModelError
from src.utils.file_utils import ensure_directory
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ClassMetrics:
    """Precision, recall, F1, and support for one class code."""

    code: int
    name: str
    precision: float
    recall: float
    f1: float
    support: int

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "name": self.name,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
        }


@dataclass(frozen=True)
class FeatureImportance:
    """Random Forest importance for one named feature."""

    name: str
    importance: float

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "importance": self.importance}


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics computed from real held-out predictions only."""

    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    labels: tuple[int, ...]
    class_names: tuple[str, ...]
    confusion_matrix: np.ndarray
    per_class: tuple[ClassMetrics, ...]
    feature_importances: tuple[FeatureImportance, ...]
    n_samples: int
    true_class_counts: Mapping[int, int]
    predicted_class_counts: Mapping[int, int]

    def __post_init__(self) -> None:
        if self.n_samples < 1:
            raise DataValidationError("Evaluation requires at least one sample")
        expected_shape = (len(self.labels), len(self.labels))
        if self.confusion_matrix.shape != expected_shape:
            raise DataValidationError(
                f"Confusion matrix shape {self.confusion_matrix.shape} does not match "
                f"{expected_shape}"
            )
        if len(self.class_names) != len(self.labels):
            raise DataValidationError("Class names must match the number of labels")
        if sum(item.support for item in self.per_class) != self.n_samples:
            raise DataValidationError("Per-class support does not sum to the sample count")

    def to_dict(self) -> dict[str, object]:
        return {
            "metrics": {
                "accuracy": self.accuracy,
                "balanced_accuracy": self.balanced_accuracy,
                "precision_macro": self.precision_macro,
                "recall_macro": self.recall_macro,
                "f1_macro": self.f1_macro,
                "precision_weighted": self.precision_weighted,
                "recall_weighted": self.recall_weighted,
                "f1_weighted": self.f1_weighted,
            },
            "n_samples": self.n_samples,
            "labels": list(self.labels),
            "class_names": list(self.class_names),
            "confusion_matrix": {
                "labels": list(self.labels),
                "matrix": self.confusion_matrix.astype(int).tolist(),
            },
            "per_class": [item.to_dict() for item in self.per_class],
            "feature_importances": [item.to_dict() for item in self.feature_importances],
            "true_class_counts": {
                str(code): count for code, count in sorted(self.true_class_counts.items())
            },
            "predicted_class_counts": {
                str(code): count for code, count in sorted(self.predicted_class_counts.items())
            },
        }


def _class_counts(values: np.ndarray) -> dict[int, int]:
    codes, counts = np.unique(values, return_counts=True)
    return {int(code): int(count) for code, count in zip(codes, counts, strict=True)}


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    feature_names: Sequence[str],
    feature_importances: Sequence[float] | None = None,
    labels: Sequence[int] | None = None,
) -> EvaluationResult:
    """Evaluate real predictions with explicit labels and zero division handled.

    Macro metrics use the union of observed true and predicted classes unless
    ``labels`` is supplied. Classes with no support still appear in the
    confusion matrix and per-class report with zero values.
    """
    true = np.asarray(y_true, dtype=np.int16).ravel()
    predicted = np.asarray(y_pred, dtype=np.int16).ravel()
    if true.size == 0:
        raise DataValidationError("Evaluation labels must not be empty")
    if true.shape != predicted.shape:
        raise DataValidationError(
            f"True and predicted labels must match in length, got {true.size} and "
            f"{predicted.size}"
        )

    if labels is None:
        resolved_labels = tuple(sorted(set(true.tolist()) | set(predicted.tolist())))
    else:
        resolved_labels = tuple(sorted({int(value) for value in labels}))
    if not resolved_labels:
        raise DataValidationError("At least one evaluation label is required")
    invalid = set(resolved_labels) - set(VALID_CLASS_CODES)
    if invalid:
        raise DataValidationError(f"Evaluation labels contain unknown class codes: {invalid}")

    precision, recall, f1, support = precision_recall_fscore_support(
        true,
        predicted,
        labels=resolved_labels,
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        true,
        predicted,
        labels=resolved_labels,
        average="macro",
        zero_division=0,
    )
    weighted = precision_recall_fscore_support(
        true,
        predicted,
        labels=resolved_labels,
        average="weighted",
        zero_division=0,
    )
    matrix = confusion_matrix(true, predicted, labels=resolved_labels)

    per_class = tuple(
        ClassMetrics(
            code=code,
            name=get_class_name(code),
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(f1[index]),
            support=int(support[index]),
        )
        for index, code in enumerate(resolved_labels)
    )
    observed_recalls = [item.recall for item in per_class if item.support > 0]
    balanced_accuracy = float(np.mean(observed_recalls)) if observed_recalls else 0.0

    names = tuple(str(name) for name in feature_names)
    if feature_importances is None:
        importance_items: tuple[FeatureImportance, ...] = ()
    else:
        values = np.asarray(feature_importances, dtype=np.float64).ravel()
        if values.size != len(names):
            raise DataValidationError(
                f"Expected {len(names)} feature importances, got {values.size}"
            )
        if not np.isfinite(values).all():
            raise DataValidationError("Feature importances contain non-finite values")
        importance_items = tuple(
            FeatureImportance(name=name, importance=float(value))
            for name, value in zip(names, values, strict=True)
        )

    return EvaluationResult(
        accuracy=float(accuracy_score(true, predicted)),
        balanced_accuracy=balanced_accuracy,
        precision_macro=float(macro[0]),
        recall_macro=float(macro[1]),
        f1_macro=float(macro[2]),
        precision_weighted=float(weighted[0]),
        recall_weighted=float(weighted[1]),
        f1_weighted=float(weighted[2]),
        labels=resolved_labels,
        class_names=tuple(get_class_name(code) for code in resolved_labels),
        confusion_matrix=matrix,
        per_class=per_class,
        feature_importances=importance_items,
        n_samples=int(true.size),
        true_class_counts=_class_counts(true),
        predicted_class_counts=_class_counts(predicted),
    )


def evaluate_random_forest(
    trained: TrainedRandomForest,
    validation: LabeledSamples,
) -> EvaluationResult:
    """Evaluate a trained model on a held-out LabeledSamples subset."""
    if validation.feature_names != trained.feature_names:
        raise ModelError(
            f"Validation features {validation.feature_names} do not match training "
            f"features {trained.feature_names}"
        )
    predictions = trained.predict(validation.X)
    result = evaluate_predictions(
        validation.y,
        predictions,
        feature_names=trained.feature_names,
        feature_importances=trained.model.feature_importances_,
    )
    logger.info(
        "Evaluated %d held-out samples: accuracy=%.4f, macro F1=%.4f",
        result.n_samples,
        result.accuracy,
        result.f1_macro,
    )
    return result


def save_evaluation_report(result: EvaluationResult, output_path: str | Path) -> Path:
    """Write evaluation metrics, confusion matrix, and importances as JSON."""
    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() != ".json":
        raise DataValidationError(
            f"Unsupported evaluation report type {destination.suffix!r}; expected .json"
        )
    ensure_directory(destination.parent)
    destination.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote evaluation report to %s", destination)
    return destination
