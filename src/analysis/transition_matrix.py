"""Class-to-class transition matrix models and tabular export."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from config.land_cover_classes import get_class_color, get_class_name
from src.utils.errors import DataValidationError, RasterError
from src.utils.file_utils import ensure_directory
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_CSV_SUFFIXES: frozenset[str] = frozenset({".csv"})
CHANGE_CAVEAT = (
    "Differences between classified maps are not proof of real-world land-cover "
    "change. They may reflect seasonality, clouds, acquisition differences, "
    "classification errors, or different models, preprocessing, or class definitions."
)


@dataclass(frozen=True)
class Transition:
    """Pixel count and area moving from one class to another."""

    from_code: int
    from_name: str
    to_code: int
    to_name: str
    pixel_count: int
    area_km2: float
    percentage_of_valid: float
    percentage_of_changed: float

    @property
    def changed(self) -> bool:
        return self.from_code != self.to_code

    def to_dict(self) -> dict[str, object]:
        return {
            "from_code": self.from_code,
            "from_name": self.from_name,
            "to_code": self.to_code,
            "to_name": self.to_name,
            "pixel_count": self.pixel_count,
            "area_km2": self.area_km2,
            "percentage_of_valid": self.percentage_of_valid,
            "percentage_of_changed": self.percentage_of_changed,
            "changed": self.changed,
        }


@dataclass(frozen=True)
class TransitionMatrix:
    """Pixel counts and areas for every earlier/later class combination."""

    earlier_path: Path
    later_path: Path
    classes: tuple[int, ...]
    counts: Mapping[int, Mapping[int, int]]
    areas_km2: Mapping[int, Mapping[int, float]]
    area_method: str
    pixel_area_km2: float | None
    valid_pixels: int
    changed_pixels: int

    @property
    def unchanged_pixels(self) -> int:
        return self.valid_pixels - self.changed_pixels

    @property
    def changed_percentage(self) -> float:
        if not self.valid_pixels:
            return 0.0
        return (self.changed_pixels / self.valid_pixels) * 100.0

    @property
    def total_area_km2(self) -> float:
        return float(
            sum(
                self.areas_km2[from_code][to_code]
                for from_code in self.classes
                for to_code in self.classes
            )
        )

    @property
    def changed_area_km2(self) -> float:
        return float(
            sum(
                self.areas_km2[from_code][to_code]
                for from_code in self.classes
                for to_code in self.classes
                if from_code != to_code
            )
        )

    @property
    def transitions(self) -> tuple[Transition, ...]:
        return tuple(
            Transition(
                from_code=from_code,
                from_name=get_class_name(from_code),
                to_code=to_code,
                to_name=get_class_name(to_code),
                pixel_count=int(self.counts[from_code][to_code]),
                area_km2=float(self.areas_km2[from_code][to_code]),
                percentage_of_valid=(
                    int(self.counts[from_code][to_code]) / self.valid_pixels * 100.0
                    if self.valid_pixels
                    else 0.0
                ),
                percentage_of_changed=(
                    int(self.counts[from_code][to_code]) / self.changed_pixels * 100.0
                    if from_code != to_code and self.changed_pixels
                    else 0.0
                ),
            )
            for from_code in self.classes
            for to_code in self.classes
        )

    @property
    def changed_transitions(self) -> tuple[Transition, ...]:
        return tuple(
            sorted(
                (
                    item
                    for item in self.transitions
                    if item.changed and item.pixel_count > 0
                ),
                key=lambda item: (-item.area_km2, item.from_code, item.to_code),
            )
        )

    @property
    def matrix_pixels(self) -> list[list[int]]:
        return [
            [int(self.counts[from_code][to_code]) for to_code in self.classes]
            for from_code in self.classes
        ]

    @property
    def matrix_areas_km2(self) -> list[list[float]]:
        return [
            [float(self.areas_km2[from_code][to_code]) for to_code in self.classes]
            for from_code in self.classes
        ]

    @property
    def class_legend(self) -> list[dict[str, object]]:
        return [
            {
                "code": code,
                "name": get_class_name(code),
                "color": get_class_color(code),
            }
            for code in self.classes
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "earlier_path": str(self.earlier_path),
            "later_path": str(self.later_path),
            "area_method": self.area_method,
            "pixel_area_km2": self.pixel_area_km2,
            "classes": self.class_legend,
            "matrix_pixels": self.matrix_pixels,
            "matrix_area_km2": self.matrix_areas_km2,
            "transitions": [item.to_dict() for item in self.transitions],
            "changed_transitions": [
                item.to_dict() for item in self.changed_transitions
            ],
            "valid_pixels": self.valid_pixels,
            "changed_pixels": self.changed_pixels,
            "unchanged_pixels": self.unchanged_pixels,
            "changed_percentage": self.changed_percentage,
            "total_area_km2": self.total_area_km2,
            "changed_area_km2": self.changed_area_km2,
            "scientific_caveat": CHANGE_CAVEAT,
        }


def save_transition_matrix_csv(
    matrix: TransitionMatrix,
    output_path: str | Path,
) -> Path:
    """Write every transition-matrix cell to a CSV table."""
    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() not in SUPPORTED_CSV_SUFFIXES:
        raise DataValidationError(
            f"Unsupported transition matrix type {destination.suffix!r}; expected .csv"
        )
    ensure_directory(destination.parent)

    try:
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "from_code",
                    "from_land_cover_type",
                    "to_code",
                    "to_land_cover_type",
                    "pixel_count",
                    "area_km2",
                    "percentage_of_valid",
                    "percentage_of_changed",
                    "changed",
                ]
            )
            for item in matrix.transitions:
                writer.writerow(
                    [
                        item.from_code,
                        item.from_name,
                        item.to_code,
                        item.to_name,
                        item.pixel_count,
                        f"{item.area_km2:.6f}",
                        f"{item.percentage_of_valid:.6f}",
                        f"{item.percentage_of_changed:.6f}",
                        "yes" if item.changed else "no",
                    ]
                )
    except OSError as exc:
        raise RasterError(
            f"Unable to write transition matrix {destination}: {exc}"
        ) from exc

    logger.info("Wrote transition matrix to %s", destination)
    return destination
