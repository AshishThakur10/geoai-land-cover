"""Windowed land-cover classification and georeferenced GeoTIFF output."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.errors import RasterioIOError
from rasterio.windows import Window

from config.land_cover_classes import (
    NODATA_CLASS_CODE,
    get_class_colors,
    get_class_legend,
    get_class_name,
    is_valid_class_code,
)
from src.data.raster_loader import ensure_georeferenced, open_raster, read_metadata
from src.features.feature_extraction import FeatureStack
from src.training.model_manager import LoadedModel
from src.training.random_forest import TrainedRandomForest
from src.utils.errors import DataValidationError, ModelError, RasterError
from src.utils.file_utils import ensure_directory
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

ClassifierModel = TrainedRandomForest | LoadedModel

SUPPORTED_OUTPUT_SUFFIXES: frozenset[str] = frozenset({".tif", ".tiff"})
CLASSIFICATION_DESCRIPTION = "Land-cover class code"


@dataclass(frozen=True)
class ClassificationResult:
    """Metadata and class counts for a written classification raster."""

    output_path: Path
    feature_names: tuple[str, ...]
    class_codes: tuple[int, ...]
    class_counts: Mapping[int, int]
    crs: CRS
    transform: Affine
    width: int
    height: int
    nodata: int
    window_size: int

    @property
    def total_pixels(self) -> int:
        return self.width * self.height

    @property
    def valid_pixels(self) -> int:
        return int(sum(self.class_counts.values()))

    @property
    def nodata_pixels(self) -> int:
        return self.total_pixels - self.valid_pixels

    @property
    def class_distribution(self) -> list[dict[str, object]]:
        total = self.valid_pixels
        return [
            {
                "code": code,
                "name": get_class_name(code),
                "count": int(self.class_counts.get(code, 0)),
                "percentage": (
                    (int(self.class_counts.get(code, 0)) / total) * 100.0
                    if total
                    else 0.0
                ),
            }
            for code in self.class_codes
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "output_path": str(self.output_path),
            "feature_names": list(self.feature_names),
            "crs": self.crs.to_string(),
            "transform": list(self.transform)[:6],
            "width": self.width,
            "height": self.height,
            "nodata": self.nodata,
            "window_size": self.window_size,
            "total_pixels": self.total_pixels,
            "valid_pixels": self.valid_pixels,
            "nodata_pixels": self.nodata_pixels,
            "class_distribution": self.class_distribution,
            "legend": get_class_legend(),
        }


def _positive_window_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(f"window_size must be an integer, got {value!r}")
    if value < 1:
        raise DataValidationError(f"window_size must be >= 1, got {value}")
    return value


def _valid_nodata(model: ClassifierModel, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(f"nodata must be an integer, got {value!r}")
    if not 0 <= value <= 255:
        raise DataValidationError(f"nodata must be in [0, 255], got {value}")
    if is_valid_class_code(value) or value in tuple(int(code) for code in model.classes):
        raise DataValidationError(f"nodata {value} overlaps a land-cover class code")
    return value


def _model_class_codes(model: ClassifierModel) -> tuple[int, ...]:
    codes = tuple(int(code) for code in model.classes)
    if not codes:
        raise ModelError("Classifier contains no class codes")
    invalid = [code for code in codes if not is_valid_class_code(code)]
    if invalid:
        raise ModelError(f"Classifier contains invalid land-cover class codes: {invalid}")
    return codes


def _validate_feature_contract(
    model: ClassifierModel,
    feature_names: Sequence[str],
) -> tuple[str, ...]:
    names = tuple(str(name) for name in feature_names)
    expected = tuple(str(name) for name in model.feature_names)
    if names != expected:
        raise ModelError(
            f"Feature names/order do not match the model: expected {expected}, got {names}"
        )
    return names


def _resolve_raster_feature_names(
    model: ClassifierModel,
    descriptions: Sequence[str | None],
    feature_names: Sequence[str] | None,
) -> tuple[str, ...]:
    expected = tuple(str(name) for name in model.feature_names)
    if len(descriptions) != len(expected):
        raise ModelError(
            f"Feature raster has {len(descriptions)} bands but the model expects "
            f"{len(expected)} features"
        )
    if feature_names is not None:
        names = tuple(str(name) for name in feature_names)
        return _validate_feature_contract(model, names)

    cleaned = tuple(str(description).strip() if description else "" for description in descriptions)
    if all(cleaned):
        return _validate_feature_contract(model, cleaned)
    if not any(cleaned):
        logger.warning(
            "Feature raster has no band descriptions; assuming model feature order %s",
            expected,
        )
        return expected
    raise DataValidationError(
        "Feature raster has incomplete band descriptions; supply feature_names explicitly"
    )


def _rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _write_output_metadata(dataset: rasterio.io.DatasetWriter, nodata: int) -> None:
    dataset.set_band_description(1, CLASSIFICATION_DESCRIPTION)
    dataset.update_tags(
        CLASSIFICATION="ESA WorldCover-compatible land-cover class codes",
        CLASS_LEGEND=json.dumps(get_class_legend(), separators=(",", ":"), sort_keys=True),
        NODATA_CLASS_CODE=str(nodata),
    )
    colormap = dict.fromkeys(range(256), (0, 0, 0, 0))
    for code, color in get_class_colors().items():
        colormap[code] = (*_rgb(color), 255)
    dataset.write_colormap(1, colormap)


def _profile(
    *,
    width: int,
    height: int,
    crs: CRS,
    transform: Affine,
    nodata: int,
) -> dict[str, object]:
    return {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
    }


def _validate_output_path(output_path: str | Path) -> Path:
    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() not in SUPPORTED_OUTPUT_SUFFIXES:
        raise DataValidationError(
            f"Unsupported classification type {destination.suffix!r}; expected .tif or .tiff"
        )
    ensure_directory(destination.parent)
    return destination


def _predict_valid_pixels(
    model: ClassifierModel,
    data: np.ndarray,
    valid_mask: np.ndarray,
    nodata: int,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.full(valid_mask.shape, nodata, dtype=np.uint8)
    if not bool(valid_mask.any()):
        return output, np.empty(0, dtype=np.uint8)

    X = data[:, valid_mask].T.astype(np.float32, copy=False)
    try:
        predictions = np.asarray(model.predict(X))
    except (DataValidationError, ModelError):
        raise
    except Exception as exc:
        raise ModelError(f"Classification prediction failed: {exc}") from exc

    if predictions.ndim != 1 or predictions.size != X.shape[0]:
        raise ModelError(
            f"Classifier returned predictions with shape {predictions.shape} for {X.shape[0]} rows"
        )
    allowed = set(_model_class_codes(model))
    if not np.isin(predictions, list(allowed)).all():
        invalid = sorted(set(predictions.tolist()) - allowed)
        raise ModelError(f"Classifier returned invalid class codes: {invalid}")

    codes = predictions.astype(np.uint8, copy=False)
    output[valid_mask] = codes
    return output, codes


def _class_result(
    model: ClassifierModel,
    *,
    destination: Path,
    feature_names: tuple[str, ...],
    counts: Mapping[int, int],
    nodata: int,
    window_size: int,
    width: int,
    height: int,
    crs: CRS,
    transform: Affine,
) -> ClassificationResult:
    class_codes = _model_class_codes(model)
    class_counts = MappingProxyType(
        {code: int(counts.get(code, 0)) for code in class_codes}
    )
    result = ClassificationResult(
        output_path=destination,
        feature_names=feature_names,
        class_codes=class_codes,
        class_counts=class_counts,
        crs=crs,
        transform=transform,
        width=width,
        height=height,
        nodata=nodata,
        window_size=window_size,
    )
    logger.info(
        "Wrote classification to %s (%d classified pixels, %d NoData)",
        destination,
        result.valid_pixels,
        result.nodata_pixels,
    )
    return result


def classify_feature_stack(
    model: ClassifierModel,
    features: FeatureStack,
    output_path: str | Path,
    *,
    window_size: int = 512,
    nodata: int = NODATA_CLASS_CODE,
) -> ClassificationResult:
    """Classify an in-memory feature stack in windows and write a GeoTIFF."""
    size = _positive_window_size(window_size)
    resolved_nodata = _valid_nodata(model, nodata)
    feature_names = _validate_feature_contract(model, features.feature_names)
    destination = _validate_output_path(output_path)
    counts: dict[int, int] = {}

    try:
        with rasterio.open(
            destination,
            "w",
            **_profile(
                width=features.width,
                height=features.height,
                crs=features.crs,
                transform=features.transform,
                nodata=resolved_nodata,
            ),
        ) as dataset:
            _write_output_metadata(dataset, resolved_nodata)
            for row_off in range(0, features.height, size):
                for col_off in range(0, features.width, size):
                    height = min(size, features.height - row_off)
                    width = min(size, features.width - col_off)
                    window = Window(col_off, row_off, width, height)
                    data = features.data[
                        :,
                        row_off : row_off + height,
                        col_off : col_off + width,
                    ]
                    valid = features.valid_mask[
                        row_off : row_off + height,
                        col_off : col_off + width,
                    ] & np.all(np.isfinite(data), axis=0)
                    output, codes = _predict_valid_pixels(
                        model,
                        data,
                        valid,
                        resolved_nodata,
                    )
                    if codes.size:
                        unique, window_counts = np.unique(codes, return_counts=True)
                        for code, count in zip(unique, window_counts, strict=True):
                            counts[int(code)] = counts.get(int(code), 0) + int(count)
                    dataset.write(output, 1, window=window)
    except (RasterioIOError, OSError, ValueError) as exc:
        raise RasterError(f"Unable to write classification {destination}: {exc}") from exc

    return _class_result(
        model,
        destination=destination,
        feature_names=feature_names,
        counts=counts,
        nodata=resolved_nodata,
        window_size=size,
        width=features.width,
        height=features.height,
        crs=features.crs,
        transform=features.transform,
    )


def classify_feature_raster(
    model: ClassifierModel,
    feature_path: str | Path,
    output_path: str | Path,
    *,
    feature_names: Sequence[str] | None = None,
    window_size: int = 512,
    nodata: int = NODATA_CLASS_CODE,
) -> ClassificationResult:
    """Read a feature GeoTIFF in windows, classify it, and write a GeoTIFF."""
    size = _positive_window_size(window_size)
    resolved_nodata = _valid_nodata(model, nodata)
    metadata = read_metadata(feature_path)
    ensure_georeferenced(metadata)
    names = _resolve_raster_feature_names(model, metadata.descriptions, feature_names)
    destination = _validate_output_path(output_path)
    counts: dict[int, int] = {}

    try:
        with open_raster(feature_path) as source, rasterio.open(
            destination,
            "w",
            **_profile(
                width=metadata.width,
                height=metadata.height,
                crs=metadata.crs,
                transform=metadata.transform,
                nodata=resolved_nodata,
            ),
        ) as dataset:
            _write_output_metadata(dataset, resolved_nodata)
            for row_off in range(0, metadata.height, size):
                for col_off in range(0, metadata.width, size):
                    height = min(size, metadata.height - row_off)
                    width = min(size, metadata.width - col_off)
                    window = Window(col_off, row_off, width, height)
                    data = np.ma.filled(
                        np.ma.asarray(source.read(window=window, masked=True), dtype=np.float32),
                        np.nan,
                    )
                    valid = np.all(np.isfinite(data), axis=0)
                    output, codes = _predict_valid_pixels(
                        model,
                        data,
                        valid,
                        resolved_nodata,
                    )
                    if codes.size:
                        unique, window_counts = np.unique(codes, return_counts=True)
                        for code, count in zip(unique, window_counts, strict=True):
                            counts[int(code)] = counts.get(int(code), 0) + int(count)
                    dataset.write(output, 1, window=window)
    except (RasterioIOError, OSError, ValueError) as exc:
        raise RasterError(f"Unable to classify feature raster {feature_path}: {exc}") from exc

    return _class_result(
        model,
        destination=destination,
        feature_names=names,
        counts=counts,
        nodata=resolved_nodata,
        window_size=size,
        width=metadata.width,
        height=metadata.height,
        crs=metadata.crs,
        transform=metadata.transform,
    )
