from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.transition_matrix import TransitionMatrix
from src.reports.report_generator import (
    PREDICTION_CAVEAT,
    analysis_report_json,
    analysis_report_markdown,
    build_analysis_report,
    classification_report_csv,
    confusion_matrix_csv,
)
from src.utils.errors import DataValidationError
from src.visualization.charts import (
    class_distribution_chart,
    confusion_matrix_chart,
    feature_importance_chart,
    transition_heatmap,
)


def _evaluation_report() -> dict[str, object]:
    return {
        "metrics": {"accuracy": 0.75},
        "n_samples": 6,
        "labels": [10, 40],
        "class_names": ["Tree Cover", "Cropland"],
        "confusion_matrix": {
            "labels": [10, 40],
            "matrix": [[2, 1], [0, 3]],
        },
        "per_class": [
            {
                "code": 10,
                "name": "Tree Cover",
                "precision": 1.0,
                "recall": 2 / 3,
                "f1": 0.8,
                "support": 3,
            },
            {
                "code": 40,
                "name": "Cropland",
                "precision": 0.75,
                "recall": 1.0,
                "f1": 0.857143,
                "support": 3,
            },
        ],
        "feature_importances": [
            {"name": "NDVI", "importance": 0.7},
            {"name": "B02", "importance": 0.3},
        ],
    }


def _transition_matrix() -> TransitionMatrix:
    counts = {10: {10: 2, 40: 1}, 40: {10: 0, 40: 3}}
    areas = {10: {10: 0.0002, 40: 0.0001}, 40: {10: 0.0, 40: 0.0003}}
    return TransitionMatrix(
        earlier_path=Path("earlier.tif"),
        later_path=Path("later.tif"),
        classes=(10, 40),
        counts=counts,
        areas_km2=areas,
        area_method="projected",
        pixel_area_km2=0.0001,
        valid_pixels=6,
        changed_pixels=1,
    )


def test_class_distribution_chart_uses_real_rows() -> None:
    figure = class_distribution_chart(
        [
            {
                "code": 10,
                "name": "Tree Cover",
                "color": "#006400",
                "pixel_count": 4,
                "area_km2": 2.5,
                "percentage": 40.0,
            },
            {
                "code": 40,
                "name": "Cropland",
                "color": "#FFFF4C",
                "pixel_count": 6,
                "area_km2": 3.75,
                "percentage": 60.0,
            },
        ]
    )

    assert list(figure.data[0].x) == ["Tree Cover", "Cropland"]
    assert list(figure.data[0].y) == [2.5, 3.75]
    with pytest.raises(DataValidationError, match="at least one class"):
        class_distribution_chart([])


def test_transition_heatmap_preserves_matrix() -> None:
    matrix = _transition_matrix()
    figure = transition_heatmap(matrix)

    assert [list(row) for row in figure.data[0].z] == [[2, 1], [0, 3]]
    assert list(figure.data[0].x) == ["Tree Cover", "Cropland"]


def test_confusion_and_importance_charts() -> None:
    report = _evaluation_report()
    confusion = confusion_matrix_chart(report)
    importance = feature_importance_chart(report)

    assert [list(row) for row in confusion.data[0].z] == [[2, 1], [0, 3]]
    assert list(importance.data[0].x) == [0.3, 0.7]
    assert list(importance.data[0].y) == ["B02", "NDVI"]

    with pytest.raises(DataValidationError, match="no feature importances"):
        feature_importance_chart({"feature_importances": []})
    with pytest.raises(DataValidationError, match="dimensions"):
        confusion_matrix_chart(
            {
                "confusion_matrix": {"labels": [10, 40], "matrix": [[1]]},
                "class_names": ["Tree Cover", "Cropland"],
            }
        )


def test_confusion_matrix_csv_matches_report() -> None:
    payload = confusion_matrix_csv(_evaluation_report())
    lines = payload.splitlines()

    assert lines[0] == (
        "actual_code,actual_land_cover_type,10: Tree Cover,40: Cropland"
    )
    assert lines[1] == "10,Tree Cover,2,1"
    assert lines[2] == "40,Cropland,0,3"


def test_classification_report_csv_matches_report() -> None:
    payload = classification_report_csv(_evaluation_report())
    lines = payload.splitlines()

    assert lines[0] == "code,land_cover_type,precision,recall,f1,support"
    assert lines[1].startswith("10,Tree Cover,1.000000,0.666667,0.800000,3")
    assert "Cropland" in lines[2]

    with pytest.raises(DataValidationError, match="per_class"):
        classification_report_csv({})


def test_report_csv_rejects_malformed_metric_values() -> None:
    report = _evaluation_report()
    report["per_class"] = [{"code": 10}]
    with pytest.raises(DataValidationError, match="require code, name"):
        classification_report_csv(report)

    malformed = _evaluation_report()
    malformed["confusion_matrix"] = {
        "labels": [10, 40],
        "matrix": [["not-a-count", 1], [0, 3]],
    }
    with pytest.raises(DataValidationError, match="integer confusion-matrix"):
        confusion_matrix_csv(malformed)


def test_analysis_report_json_and_markdown_round_trip() -> None:
    report = build_analysis_report(
        study_area={"feature_count": 1, "area_km2": 1.25},
        classification={"total_area_km2": 1.25},
        generated_utc="2026-01-02T03:04:05+00:00",
    )
    payload = analysis_report_json(report)
    parsed = json.loads(payload)
    markdown = analysis_report_markdown(report)

    assert parsed["generated_utc"] == "2026-01-02T03:04:05+00:00"
    assert parsed["sections"]["study_area"]["area_km2"] == 1.25
    assert parsed["scientific_caveats"][0] == PREDICTION_CAVEAT
    assert "# GeoAI Land Cover Analysis Report" in markdown
    assert "## Study area" in markdown
    assert "## Classification and area statistics" in markdown
    assert "## Scientific caveats" in markdown

    with pytest.raises(DataValidationError, match="at least one data section"):
        build_analysis_report()
    with pytest.raises(DataValidationError, match="sections mapping"):
        analysis_report_markdown({"title": "x"})
