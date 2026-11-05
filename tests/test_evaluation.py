from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.training.evaluation import (
    evaluate_predictions,
    evaluate_random_forest,
    save_evaluation_report,
)
from src.training.random_forest import RandomForestConfig, train_random_forest
from src.utils.errors import DataValidationError, ModelError
from tests.fixtures import synthetic_ml_samples


def test_evaluate_predictions_computes_known_metrics() -> None:
    y_true = np.array([10, 10, 40, 40], dtype=np.int16)
    y_pred = np.array([10, 40, 40, 40], dtype=np.int16)

    result = evaluate_predictions(
        y_true,
        y_pred,
        feature_names=("B02", "NDVI"),
        feature_importances=(0.6, 0.4),
    )

    assert result.labels == (10, 40)
    assert result.class_names == ("Tree Cover", "Cropland")
    assert result.n_samples == 4
    assert result.accuracy == pytest.approx(0.75)
    assert result.balanced_accuracy == pytest.approx(0.75)
    assert result.precision_macro == pytest.approx((1.0 + 2 / 3) / 2)
    assert result.recall_macro == pytest.approx(0.75)
    assert result.f1_macro == pytest.approx((2 / 3 + 0.8) / 2)
    assert result.precision_weighted == pytest.approx((1.0 + 2 / 3) / 2)
    assert result.f1_weighted == pytest.approx((2 / 3 + 0.8) / 2)
    assert result.confusion_matrix.tolist() == [[1, 1], [0, 2]]
    assert [item.support for item in result.per_class] == [2, 2]
    assert result.per_class[0].precision == pytest.approx(1.0)
    assert result.per_class[0].recall == pytest.approx(0.5)
    assert result.per_class[1].f1 == pytest.approx(0.8)
    assert [item.importance for item in result.feature_importances] == pytest.approx([0.6, 0.4])
    assert result.true_class_counts == {10: 2, 40: 2}
    assert result.predicted_class_counts == {10: 1, 40: 3}


def test_evaluate_predictions_handles_explicit_absent_labels() -> None:
    result = evaluate_predictions(
        np.array([10, 10]),
        np.array([10, 40]),
        feature_names=(),
        labels=(10, 40, 80),
    )

    assert result.labels == (10, 40, 80)
    assert [item.support for item in result.per_class] == [2, 0, 0]
    assert result.per_class[2].precision == 0.0
    assert result.true_class_counts == {10: 2}
    assert result.predicted_class_counts == {10: 1, 40: 1}


def test_evaluate_predictions_rejects_invalid_inputs() -> None:
    with pytest.raises(DataValidationError, match="must not be empty"):
        evaluate_predictions(np.array([]), np.array([]), feature_names=())
    with pytest.raises(DataValidationError, match="match in length"):
        evaluate_predictions(np.array([10]), np.array([10, 40]), feature_names=())
    with pytest.raises(DataValidationError, match="unknown class codes"):
        evaluate_predictions(np.array([99]), np.array([10]), feature_names=())
    with pytest.raises(DataValidationError, match="Expected 2 feature importances"):
        evaluate_predictions(
            np.array([10]),
            np.array([10]),
            feature_names=("a", "b"),
            feature_importances=(1.0,),
            labels=(10, 40),
        )
    with pytest.raises(DataValidationError, match="non-finite"):
        evaluate_predictions(
            np.array([10]),
            np.array([10]),
            feature_names=("a",),
            feature_importances=(np.nan,),
            labels=(10, 40),
        )


def _train_and_validate():
    samples = synthetic_ml_samples()
    validation_indices = np.arange(samples.n_samples).reshape(4, 50)[:, ::5].ravel()
    train_indices = np.setdiff1d(np.arange(samples.n_samples), validation_indices)
    trained = train_random_forest(
        samples.take(train_indices),
        RandomForestConfig(n_estimators=15, max_depth=5, random_state=5, n_jobs=1),
    )
    return trained, samples.take(validation_indices)


def test_evaluate_random_forest_on_held_out_samples() -> None:
    trained, validation = _train_and_validate()

    result = evaluate_random_forest(trained, validation)

    assert result.n_samples == 40
    assert result.accuracy > 0.9
    assert sum(item.support for item in result.per_class) == 40
    assert result.feature_importances
    assert len(result.feature_importances) == len(trained.feature_names)
    assert sum(item.importance for item in result.feature_importances) == pytest.approx(1.0)


def test_evaluate_random_forest_rejects_feature_mismatch() -> None:
    trained, validation = _train_and_validate()
    mismatched = replace(
        validation,
        feature_names=tuple(f"feature_{index}" for index in range(validation.n_features)),
    )

    with pytest.raises(ModelError, match="do not match"):
        evaluate_random_forest(trained, mismatched)


def test_evaluation_result_serialises_and_saves(tmp_path: Path) -> None:
    result = evaluate_predictions(
        np.array([10, 10, 40, 40]),
        np.array([10, 40, 40, 40]),
        feature_names=("B02", "NDVI"),
        feature_importances=(0.6, 0.4),
    )
    output = tmp_path / "nested" / "evaluation.json"

    saved = save_evaluation_report(result, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert saved == output.resolve()
    assert payload["metrics"]["accuracy"] == pytest.approx(0.75)
    assert payload["confusion_matrix"]["matrix"] == [[1, 1], [0, 2]]
    assert payload["per_class"][0]["name"] == "Tree Cover"
    assert payload["feature_importances"][1]["name"] == "NDVI"
    assert result.to_dict()["true_class_counts"] == {"10": 2, "40": 2}


def test_save_evaluation_report_rejects_unsupported_suffix(tmp_path: Path) -> None:
    result = evaluate_predictions(
        np.array([10]),
        np.array([10]),
        feature_names=(),
        labels=(10, 40),
    )

    with pytest.raises(DataValidationError, match="Unsupported evaluation report"):
        save_evaluation_report(result, tmp_path / "evaluation.csv")
