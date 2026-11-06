"""Plotly charts shared by the dashboard and report exports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import plotly.graph_objects as go

from src.analysis.transition_matrix import TransitionMatrix
from src.utils.errors import DataValidationError


def _measurement(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataValidationError(f"{name} must be a mapping")
    return value


def class_distribution_chart(classes: Sequence[Mapping[str, object]]) -> go.Figure:
    """Build a class-area bar chart from real area-statistics rows."""
    if not classes:
        raise DataValidationError("Class distribution requires at least one class")
    names = [str(item["name"]) for item in classes]
    areas = [float(item["area_km2"]) for item in classes]
    pixels = [int(item["pixel_count"]) for item in classes]
    percentages = [float(item["percentage"]) for item in classes]
    colors = [str(item.get("color", "#4C78A8")) for item in classes]

    figure = go.Figure(
        go.Bar(
            x=names,
            y=areas,
            marker_color=colors,
            customdata=list(zip(pixels, percentages, strict=True)),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Area: %{y:,.2f} km²<br>"
                "Pixels: %{customdata[0]:,}<br>"
                "Share: %{customdata[1]:.2f}%"
                "<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Land-cover area by class",
        xaxis_title="Land-cover class",
        yaxis_title="Area (km²)",
        margin={"l": 20, "r": 20, "t": 50, "b": 90},
        showlegend=False,
        xaxis_tickangle=-35,
    )
    return figure


def transition_heatmap(matrix: TransitionMatrix) -> go.Figure:
    """Build a pixel-count heatmap for a class transition matrix."""
    names = [str(item["name"]) for item in matrix.class_legend]
    areas = matrix.matrix_areas_km2
    labels = [[f"{value:.3f} km²" for value in row] for row in areas]
    figure = go.Figure(
        go.Heatmap(
            z=matrix.matrix_pixels,
            x=names,
            y=names,
            customdata=labels,
            colorscale="Blues",
            colorbar={"title": "Pixels"},
            hovertemplate=(
                "Earlier: %{y}<br>Later: %{x}<br>"
                "Pixels: %{z:,}<br>Area: %{customdata}"
                "<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Class transition matrix",
        xaxis_title="Later class",
        yaxis_title="Earlier class",
        margin={"l": 20, "r": 20, "t": 50, "b": 90},
        xaxis_tickangle=-35,
    )
    return figure


def confusion_matrix_chart(report: Mapping[str, object]) -> go.Figure:
    """Build a confusion-matrix heatmap from an evaluation report mapping."""
    raw = _measurement(report.get("confusion_matrix"), "confusion_matrix")
    matrix = raw.get("matrix")
    labels = raw.get("labels")
    if not isinstance(matrix, list) or not isinstance(labels, list):
        raise DataValidationError("Evaluation report requires matrix and labels")
    class_names = report.get("class_names")
    if not isinstance(class_names, list) or len(class_names) != len(labels):
        class_names = [str(label) for label in labels]
    names = [str(name) for name in class_names]
    values = [[int(value) for value in row] for row in matrix]
    if len(values) != len(names) or any(len(row) != len(names) for row in values):
        raise DataValidationError("Confusion matrix dimensions do not match class labels")

    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=names,
            y=names,
            text=values,
            texttemplate="%{text}",
            colorscale="Blues",
            colorbar={"title": "Samples"},
            hovertemplate=(
                "Actual: %{y}<br>Predicted: %{x}<br>Samples: %{z:,}"
                "<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Confusion matrix",
        xaxis_title="Predicted class",
        yaxis_title="Actual class",
        margin={"l": 20, "r": 20, "t": 50, "b": 90},
        xaxis_tickangle=-35,
    )
    return figure


def feature_importance_chart(
    report: Mapping[str, object],
    *,
    top_n: int = 15,
) -> go.Figure:
    """Build a descending Random Forest feature-importance bar chart."""
    raw = report.get("feature_importances")
    if not isinstance(raw, list) or not raw:
        raise DataValidationError("Evaluation report contains no feature importances")
    if top_n < 1:
        raise DataValidationError("top_n must be >= 1")
    items = sorted(
        (
            (str(item["name"]), float(item["importance"]))
            for item in raw
            if isinstance(item, Mapping)
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]
    names = [item[0] for item in items][::-1]
    values = [item[1] for item in items][::-1]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color="#4C78A8",
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Feature importance",
        xaxis_title="Importance",
        yaxis_title=None,
        margin={"l": 20, "r": 20, "t": 50, "b": 40},
        showlegend=False,
    )
    return figure


__all__ = [
    "class_distribution_chart",
    "confusion_matrix_chart",
    "feature_importance_chart",
    "transition_heatmap",
]
