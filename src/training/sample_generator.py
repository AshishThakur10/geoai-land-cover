"""Align ESA WorldCover labels with spectral features and build samples."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from affine import Affine
from rasterio.crs import CRS

from config.land_cover_classes import (
    ESA_WORLDCOVER_CLASSES,
    NODATA_CLASS_CODE,
    VALID_CLASS_CODES,
    get_class_name,
)
from src.features.feature_extraction import FeatureStack
from src.preprocessing.raster_alignment import (
    AlignedRaster,
    TargetGrid,
    align_raster_to_grid,
)
from src.utils.errors import DataValidationError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _feature_grid(features: FeatureStack) -> TargetGrid:
    return TargetGrid(
        crs=features.crs,
        transform=features.transform,
        width=features.width,
        height=features.height,
    )


def align_labels_to_features(
    features: FeatureStack,
    label_path: str | Path,
    *,
    label_band_index: int = 1,
) -> AlignedRaster:
    """Align a WorldCover label raster to a feature stack's exact grid.

    Nearest-neighbour resampling is mandatory for categorical labels; this
    helper does not expose another method. Missing areas and label NoData become
    ``NaN``.
    """
    return align_raster_to_grid(
        label_path,
        _feature_grid(features),
        band_index=label_band_index,
        resampling="nearest",
        name="worldcover",
    )


def _as_label_codes(aligned: AlignedRaster) -> tuple[np.ndarray, np.ndarray]:
    raw = np.ma.filled(np.ma.asarray(aligned.data, dtype=np.float64), np.nan)
    finite = np.isfinite(raw)
    codes = np.zeros(raw.shape, dtype=np.int16)
    codes[finite] = np.rint(raw[finite]).astype(np.int16)
    return codes, finite


@dataclass(frozen=True)
class LabeledSamples:
    """Spectral samples paired with validated WorldCover class codes.

    ``X`` has shape ``(n_samples, n_features)`` and contains only finite values.
    ``y`` contains only recognised ESA WorldCover codes. Spatial coordinates are
    retained so splits and predictions can be traced back to the feature grid.
    """

    X: np.ndarray
    y: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    feature_names: tuple[str, ...]
    transform: Affine
    crs: CRS
    width: int
    height: int
    label_source: Path
    total_pixels: int
    ignored_feature_pixels: int
    ignored_nodata_labels: int
    unknown_label_counts: Mapping[int, int]

    def __post_init__(self) -> None:
        if self.X.ndim != 2:
            raise DataValidationError(f"Sample features must be 2D, got {self.X.shape}")
        if self.y.ndim != 1:
            raise DataValidationError(f"Sample labels must be 1D, got {self.y.shape}")
        if self.X.shape[0] != self.y.size:
            raise DataValidationError(
                f"Feature rows ({self.X.shape[0]}) do not match labels ({self.y.size})"
            )
        if self.X.shape[1] != len(self.feature_names):
            raise DataValidationError(
                f"Feature columns ({self.X.shape[1]}) do not match "
                f"{len(self.feature_names)} feature names"
            )
        if self.rows.shape != self.y.shape or self.cols.shape != self.y.shape:
            raise DataValidationError("Row/column coordinates must match label length")
        if not np.isfinite(self.X).all():
            raise DataValidationError("Sample features contain non-finite values")
        invalid_codes = set(np.unique(self.y).tolist()) - set(VALID_CLASS_CODES)
        if invalid_codes:
            raise DataValidationError(
                f"Sample labels contain invalid WorldCover codes: {sorted(invalid_codes)}"
            )
        if self.width <= 0 or self.height <= 0:
            raise DataValidationError("Feature grid dimensions must be positive")
        if self.total_pixels != self.width * self.height:
            raise DataValidationError(
                f"total_pixels ({self.total_pixels}) does not match grid "
                f"{self.width}x{self.height}"
            )

    @property
    def n_samples(self) -> int:
        return int(self.y.size)

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def class_counts(self) -> dict[int, int]:
        codes, counts = np.unique(self.y, return_counts=True)
        return {int(code): int(count) for code, count in zip(codes, counts, strict=True)}

    @property
    def class_distribution(self) -> list[dict[str, object]]:
        """Return per-class counts and percentages over valid samples."""
        counts = self.class_counts
        total = self.n_samples
        return [
            {
                "code": code,
                "name": get_class_name(code),
                "count": count,
                "percentage": (count / total) * 100.0 if total else 0.0,
            }
            for code, count in sorted(counts.items())
        ]

    @property
    def missing_classes(self) -> list[int]:
        """Return recognised WorldCover classes absent from these samples."""
        present = set(self.class_counts)
        return sorted(set(ESA_WORLDCOVER_CLASSES) - present)

    @property
    def imbalance_ratio(self) -> float:
        """Return largest-class count divided by smallest-class count."""
        counts = self.class_counts
        if not counts:
            return 0.0
        return max(counts.values()) / min(counts.values())

    def balanced_sample_weights(self) -> np.ndarray:
        """Return inverse-frequency sample weights summing to ``n_samples``.

        Each class receives equal total weight. This complements the Random
        Forest ``class_weight`` setting for estimators that accept
        ``sample_weight``.
        """
        codes, inverse, counts = np.unique(
            self.y,
            return_inverse=True,
            return_counts=True,
        )
        weights = self.n_samples / (len(codes) * counts[inverse])
        return weights.astype(np.float64, copy=False)

    def take(self, indices: np.ndarray) -> LabeledSamples:
        """Return a subset by positional indices."""
        positions = np.asarray(indices, dtype=np.int64)
        if positions.ndim != 1:
            raise DataValidationError("Sample indices must be one-dimensional")
        if positions.size == 0:
            raise DataValidationError("A sample subset must not be empty")
        if positions.min() < 0 or positions.max() >= self.n_samples:
            raise DataValidationError("Sample index is out of range")
        return LabeledSamples(
            X=self.X[positions],
            y=self.y[positions],
            rows=self.rows[positions],
            cols=self.cols[positions],
            feature_names=self.feature_names,
            transform=self.transform,
            crs=self.crs,
            width=self.width,
            height=self.height,
            label_source=self.label_source,
            total_pixels=self.total_pixels,
            ignored_feature_pixels=self.ignored_feature_pixels,
            ignored_nodata_labels=self.ignored_nodata_labels,
            unknown_label_counts=self.unknown_label_counts,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "label_source": str(self.label_source),
            "feature_names": list(self.feature_names),
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "total_pixels": self.total_pixels,
            "ignored_feature_pixels": self.ignored_feature_pixels,
            "ignored_nodata_labels": self.ignored_nodata_labels,
            "unknown_label_counts": {
                str(code): count for code, count in sorted(self.unknown_label_counts.items())
            },
            "class_distribution": self.class_distribution,
            "missing_class_codes": self.missing_classes,
            "imbalance_ratio": self.imbalance_ratio,
            "grid": {
                "crs": self.crs.to_string(),
                "transform": list(self.transform)[:6],
                "width": self.width,
                "height": self.height,
            },
        }


def generate_labeled_samples(
    features: FeatureStack,
    label_path: str | Path,
    *,
    label_band_index: int = 1,
) -> LabeledSamples:
    """Align labels to features and return only fully valid labeled pixels.

    Pixels are excluded when any feature is invalid, when the label is NoData or
    non-finite after alignment, or when the label code is not an ESA WorldCover
    class. Unknown and NoData label counts are recorded for auditability.

    Raises:
        DataValidationError: If no valid labeled samples remain.
        RasterError: If the label raster cannot be read or aligned.
    """
    aligned = align_labels_to_features(
        features,
        label_path,
        label_band_index=label_band_index,
    )
    codes, finite = _as_label_codes(aligned)

    recognised = np.isin(codes, list(VALID_CLASS_CODES))
    nodata = ~finite | (codes == NODATA_CLASS_CODE)
    unknown = finite & ~nodata & ~recognised
    valid_labels = finite & recognised

    combined = features.valid_mask & valid_labels
    if not bool(combined.any()):
        raise DataValidationError(
            "No valid labeled samples remain after feature and class validation"
        )

    rows, cols = np.nonzero(combined)
    X = features.data[:, rows, cols].T.astype(np.float32, copy=False)
    y = codes[rows, cols].astype(np.int16, copy=False)

    unknown_codes, unknown_counts = np.unique(codes[unknown], return_counts=True)
    unknown_mapping = MappingProxyType(
        {
            int(code): int(count)
            for code, count in zip(unknown_codes, unknown_counts, strict=True)
        }
    )

    samples = LabeledSamples(
        X=X,
        y=y,
        rows=rows.astype(np.int64, copy=False),
        cols=cols.astype(np.int64, copy=False),
        feature_names=features.feature_names,
        transform=features.transform,
        crs=features.crs,
        width=features.width,
        height=features.height,
        label_source=aligned.source_metadata.path,
        total_pixels=features.total_pixel_count,
        ignored_feature_pixels=features.total_pixel_count - features.valid_pixel_count,
        ignored_nodata_labels=int(np.count_nonzero(nodata)),
        unknown_label_counts=unknown_mapping,
    )

    logger.info(
        "Generated %d labeled samples across %d classes from %s",
        samples.n_samples,
        len(samples.class_counts),
        samples.label_source,
    )
    return samples
