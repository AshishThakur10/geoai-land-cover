"""Analysis report and evaluation-table serialization."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from src.analysis.transition_matrix import CHANGE_CAVEAT
from src.utils.errors import DataValidationError

PREDICTION_CAVEAT = (
    "Classified maps and model metrics describe model predictions and held-out "
    "samples; they are not a field-verified inventory of land cover."
)
REPORT_SECTIONS: tuple[tuple[str, str], ...] = (
    ("study_area", "Study area"),
    ("imagery", "Satellite imagery"),
    ("spectral_analysis", "Spectral analysis"),
    ("classification", "Classification and area statistics"),
    ("model_evaluation", "Model evaluation"),
    ("change_detection", "Change detection"),
)


def _rows(report: Mapping[str, object]) -> tuple[list[int], list[str], list[list[int]]]:
    raw = report.get("confusion_matrix")
    if not isinstance(raw, Mapping):
        raise DataValidationError("Evaluation report requires confusion_matrix")
    labels = raw.get("labels")
    matrix = raw.get("matrix")
    class_names = report.get("class_names")
    if not isinstance(labels, list) or not isinstance(matrix, list):
        raise DataValidationError("Evaluation report requires confusion-matrix labels")
    if not isinstance(class_names, list) or len(class_names) != len(labels):
        class_names = [str(label) for label in labels]
    resolved_labels = [int(label) for label in labels]
    names = [str(name) for name in class_names]
    values: list[list[int]] = []
    for row in matrix:
        if not isinstance(row, list) or len(row) != len(resolved_labels):
            raise DataValidationError("Confusion matrix row has the wrong length")
        values.append([int(value) for value in row])
    if len(values) != len(resolved_labels):
        raise DataValidationError("Confusion matrix has the wrong number of rows")
    return resolved_labels, names, values


def _csv_text(rows: Sequence[Sequence[object]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def confusion_matrix_csv(report: Mapping[str, object]) -> str:
    """Serialize an evaluation report's confusion matrix as CSV."""
    labels, names, matrix = _rows(report)
    header: list[object] = ["actual_code", "actual_land_cover_type"]
    header.extend(f"{code}: {name}" for code, name in zip(labels, names, strict=True))
    rows: list[list[object]] = [header]
    rows.extend(
        [code, name, *matrix[index]]
        for index, (code, name) in enumerate(zip(labels, names, strict=True))
    )
    return _csv_text(rows)


def classification_report_csv(report: Mapping[str, object]) -> str:
    """Serialize per-class precision, recall, F1, and support as CSV."""
    per_class = report.get("per_class")
    if not isinstance(per_class, list) or not per_class:
        raise DataValidationError("Evaluation report requires per_class metrics")
    rows: list[list[object]] = [
        ["code", "land_cover_type", "precision", "recall", "f1", "support"]
    ]
    for item in per_class:
        if not isinstance(item, Mapping):
            raise DataValidationError("Per-class metrics must be mappings")
        rows.append(
            [
                int(item["code"]),
                str(item["name"]),
                f"{float(item['precision']):.6f}",
                f"{float(item['recall']):.6f}",
                f"{float(item['f1']):.6f}",
                int(item["support"]),
            ]
        )
    return _csv_text(rows)


def build_analysis_report(
    *,
    study_area: Mapping[str, object] | None = None,
    imagery: Mapping[str, object] | None = None,
    spectral_analysis: Mapping[str, object] | None = None,
    classification: Mapping[str, object] | None = None,
    model_evaluation: Mapping[str, object] | None = None,
    change_detection: Mapping[str, object] | None = None,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Assemble a JSON-safe report from real selected-analysis data."""
    sections = {
        "study_area": study_area,
        "imagery": imagery,
        "spectral_analysis": spectral_analysis,
        "classification": classification,
        "model_evaluation": model_evaluation,
        "change_detection": change_detection,
    }
    populated = {
        key: value for key, value in sections.items() if value is not None
    }
    if not populated:
        raise DataValidationError("Analysis report requires at least one data section")
    timestamp = generated_utc or datetime.now(UTC).isoformat()
    return {
        "title": "GeoAI Land Cover Analysis Report",
        "generated_utc": timestamp,
        "sections": populated,
        "scientific_caveats": [PREDICTION_CAVEAT, CHANGE_CAVEAT],
    }


def analysis_report_json(report: Mapping[str, object]) -> str:
    """Serialize an analysis report as indented JSON."""
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def analysis_report_markdown(report: Mapping[str, object]) -> str:
    """Serialize an analysis report as Markdown with embedded source data."""
    title = str(report.get("title", "GeoAI Land Cover Analysis Report"))
    generated = str(report.get("generated_utc", "unknown"))
    sections = report.get("sections")
    if not isinstance(sections, Mapping):
        raise DataValidationError("Analysis report requires a sections mapping")
    lines = [f"# {title}", "", f"Generated: {generated}", ""]
    for key, heading in REPORT_SECTIONS:
        value = sections.get(key)
        if value is None:
            continue
        lines.extend(
            [
                f"## {heading}",
                "",
                "```json",
                json.dumps(value, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    caveats = report.get("scientific_caveats")
    if isinstance(caveats, list) and caveats:
        lines.extend(["## Scientific caveats", ""])
        lines.extend(f"- {caveat}" for caveat in caveats)
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "PREDICTION_CAVEAT",
    "analysis_report_json",
    "analysis_report_markdown",
    "build_analysis_report",
    "classification_report_csv",
    "confusion_matrix_csv",
]
