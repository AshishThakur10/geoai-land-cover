from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import from_origin

from config.land_cover_classes import VALID_CLASS_CODES
from src.features.feature_extraction import FeatureStack, build_feature_stack
from src.preprocessing.raster_alignment import TargetGrid, align_raster_to_grid
from src.training.sample_generator import (
    LabeledSamples,
    align_labels_to_features,
    generate_labeled_samples,
)
from src.utils.errors import DataValidationError, RasterError
from tests.fixtures import (
    synthetic_worldcover_codes,
    write_synthetic_raster,
    write_synthetic_worldcover,
)


def _feature_stack(tmp_path: Path) -> FeatureStack:
    target = write_synthetic_raster(tmp_path / "target.tif", count=4, nodata=0)
    grid = TargetGrid.from_raster(target)
    bands = {}
    for key, index in (("B02", 1), ("B03", 2), ("B04", 3), ("B08", 4)):
        path = write_synthetic_raster(tmp_path / f"{key}.tif", count=4, nodata=0)
        bands[key] = align_raster_to_grid(
            path,
            grid,
            band_index=index,
            resampling="nearest",
            name=key,
        )
    return build_feature_stack(bands)


def _manual_samples(
    *,
    rows: np.ndarray,
    cols: np.ndarray,
    y: np.ndarray,
) -> LabeledSamples:
    X = np.column_stack([rows, cols, rows + cols]).astype(np.float32)
    return LabeledSamples(
        X=X,
        y=np.asarray(y, dtype=np.int16),
        rows=np.asarray(rows, dtype=np.int64),
        cols=np.asarray(cols, dtype=np.int64),
        feature_names=("B02", "B03", "NDVI"),
        transform=from_origin(500_000, 5_000_000, 10, 10),
        crs=CRS.from_epsg(32633),
        width=40,
        height=40,
        label_source=Path("synthetic_worldcover.tif"),
        total_pixels=1600,
        ignored_feature_pixels=0,
        ignored_nodata_labels=0,
        unknown_label_counts=MappingProxyType({}),
    )


def test_align_labels_to_features_uses_exact_feature_grid(tmp_path: Path) -> None:
    features = _feature_stack(tmp_path)
    labels = write_synthetic_worldcover(tmp_path / "worldcover.tif")

    aligned = align_labels_to_features(features, labels)

    assert aligned.crs == features.crs
    assert aligned.transform == features.transform
    assert aligned.data.shape == (features.height, features.width)
    assert aligned.resampling == "nearest"


def test_align_labels_resamples_coarse_worldcover_to_feature_grid(tmp_path: Path) -> None:
    features = _feature_stack(tmp_path)
    labels = write_synthetic_worldcover(
        tmp_path / "worldcover_20m.tif",
        width=20,
        height=20,
        transform=from_origin(500_000, 5_000_000, 20, 20),
    )

    aligned = align_labels_to_features(features, labels)

    assert aligned.data.shape == (40, 40)
    assert aligned.source_metadata.resolutions == (20.0, 20.0)
    assert aligned.valid_pixel_count == 1600


def test_generate_labeled_samples_keeps_valid_pixels_and_classes(tmp_path: Path) -> None:
    features = _feature_stack(tmp_path)
    labels = write_synthetic_worldcover(tmp_path / "worldcover.tif")

    samples = generate_labeled_samples(features, labels)

    assert samples.n_samples == 1598
    assert samples.n_features == 6
    assert samples.X.shape == (1598, 6)
    assert np.isfinite(samples.X).all()
    assert set(np.unique(samples.y)).issubset(VALID_CLASS_CODES)
    assert samples.class_counts == {10: 399, 30: 400, 40: 400, 80: 399}
    assert samples.ignored_feature_pixels == 2
    assert samples.ignored_nodata_labels == 0
    assert samples.label_source == labels
    assert (samples.width, samples.height) == (40, 40)
    assert samples.total_pixels == 1600


def test_generate_labeled_samples_records_unknown_and_nodata_labels(tmp_path: Path) -> None:
    features = _feature_stack(tmp_path)
    codes = synthetic_worldcover_codes()
    codes[0, 1] = 0
    codes[0, 2] = 255
    codes[1, 0] = 99
    labels = write_synthetic_worldcover(tmp_path / "worldcover.tif", codes=codes)

    samples = generate_labeled_samples(features, labels)

    assert samples.n_samples == 1595
    assert samples.ignored_nodata_labels == 1
    assert dict(samples.unknown_label_counts) == {99: 1, 255: 1}
    assert samples.to_dict()["unknown_label_counts"] == {"99": 1, "255": 1}


def test_labeled_sample_metadata_and_class_distribution(tmp_path: Path) -> None:
    samples = generate_labeled_samples(
        _feature_stack(tmp_path),
        write_synthetic_worldcover(tmp_path / "worldcover.tif"),
    )

    metadata = samples.to_dict()
    distribution = samples.class_distribution

    assert metadata["n_samples"] == 1598
    assert metadata["n_features"] == 6
    assert metadata["grid"] == {
        "crs": "EPSG:32633",
        "transform": list(from_origin(500_000, 5_000_000, 10, 10))[:6],
        "width": 40,
        "height": 40,
    }
    assert metadata["missing_class_codes"] == [20, 50, 60, 70, 90, 95, 100]
    assert metadata["imbalance_ratio"] == pytest.approx(400 / 399)
    assert distribution[0] == {
        "code": 10,
        "name": "Tree Cover",
        "count": 399,
        "percentage": pytest.approx(399 / 1598 * 100),
    }
    assert sum(entry["count"] for entry in distribution) == 1598


def test_balanced_sample_weights_equalise_class_totals(tmp_path: Path) -> None:
    samples = generate_labeled_samples(
        _feature_stack(tmp_path),
        write_synthetic_worldcover(tmp_path / "worldcover.tif"),
    )

    weights = samples.balanced_sample_weights()

    assert weights.shape == (samples.n_samples,)
    assert weights.sum() == pytest.approx(samples.n_samples)
    for code in samples.class_counts:
        assert weights[samples.y == code].sum() == pytest.approx(1598 / 4)


def test_generate_labeled_samples_rejects_no_valid_samples(tmp_path: Path) -> None:
    features = _feature_stack(tmp_path)
    labels = write_synthetic_worldcover(
        tmp_path / "worldcover.tif",
        codes=np.zeros((40, 40), dtype=np.uint8),
    )

    with pytest.raises(DataValidationError, match="No valid labeled samples"):
        generate_labeled_samples(features, labels)


def test_generate_labeled_samples_propagates_label_errors(tmp_path: Path) -> None:
    features = _feature_stack(tmp_path)
    labels = write_synthetic_worldcover(tmp_path / "worldcover.tif")
    no_crs = write_synthetic_worldcover(tmp_path / "no_crs.tif", crs=None)

    with pytest.raises(DataValidationError, match="out of range"):
        generate_labeled_samples(features, labels, label_band_index=2)
    with pytest.raises(RasterError, match="no CRS"):
        generate_labeled_samples(features, no_crs)


def test_labeled_samples_validate_internal_arrays() -> None:
    rows = np.array([0, 0], dtype=np.int64)
    cols = np.array([0, 1], dtype=np.int64)
    y = np.array([10, 40], dtype=np.int16)
    X = np.ones((2, 3), dtype=np.float32)
    common = {
        "rows": rows,
        "cols": cols,
        "feature_names": ("a", "b", "c"),
        "transform": Affine.identity(),
        "crs": CRS.from_epsg(4326),
        "width": 2,
        "height": 2,
        "label_source": Path("labels.tif"),
        "total_pixels": 4,
        "ignored_feature_pixels": 0,
        "ignored_nodata_labels": 0,
        "unknown_label_counts": MappingProxyType({}),
    }

    with pytest.raises(DataValidationError, match="must be 2D"):
        LabeledSamples(X=np.ones(2, dtype=np.float32), y=y, **common)
    with pytest.raises(DataValidationError, match="do not match labels"):
        LabeledSamples(X=X, y=np.array([10], dtype=np.int16), **common)
    with pytest.raises(DataValidationError, match="invalid WorldCover codes"):
        LabeledSamples(X=X, y=np.array([10, 255], dtype=np.int16), **common)
    with pytest.raises(DataValidationError, match="non-finite"):
        LabeledSamples(X=np.array([[1, 2, np.nan], [1, 2, 3]], dtype=np.float32), y=y, **common)
    with pytest.raises(DataValidationError, match="does not match grid"):
        LabeledSamples(X=X, y=y, **{**common, "total_pixels": 3})
    with pytest.raises(DataValidationError, match="one-dimensional"):
        _manual_samples(rows=rows, cols=cols, y=y).take(np.array([[0, 1]]))
    with pytest.raises(DataValidationError, match="must not be empty"):
        _manual_samples(rows=rows, cols=cols, y=y).take(np.array([], dtype=np.int64))
    with pytest.raises(DataValidationError, match="out of range"):
        _manual_samples(rows=rows, cols=cols, y=y).take(np.array([5]))
