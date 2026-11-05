from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from src.training.dataset_split import save_split_manifest, split_labeled_samples
from src.training.sample_generator import LabeledSamples
from src.utils.errors import DataValidationError


def _full_grid_samples() -> LabeledSamples:
    rows, cols = np.mgrid[0:40, 0:40]
    rows = rows.ravel().astype(np.int64)
    cols = cols.ravel().astype(np.int64)
    y = np.full(rows.size, 10, dtype=np.int16)
    y[(rows < 20) & (cols >= 20)] = 40
    y[(rows >= 20) & (cols < 20)] = 30
    y[(rows >= 20) & (cols >= 20)] = 80
    X = np.column_stack([rows, cols, rows + cols]).astype(np.float32)

    return LabeledSamples(
        X=X,
        y=y,
        rows=rows,
        cols=cols,
        feature_names=("B02", "B03", "NDVI"),
        transform=Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 5_000_000.0),
        crs=CRS.from_epsg(32633),
        width=40,
        height=40,
        label_source=Path("synthetic_worldcover.tif"),
        total_pixels=1600,
        ignored_feature_pixels=0,
        ignored_nodata_labels=0,
        unknown_label_counts=MappingProxyType({}),
    )


def _blocks(samples: LabeledSamples, block_size: int) -> set[tuple[int, int]]:
    return set(
        zip(
            (samples.rows // block_size).tolist(),
            (samples.cols // block_size).tolist(),
            strict=True,
        )
    )


def test_split_is_pixel_disjoint_and_covers_all_samples() -> None:
    samples = _full_grid_samples()

    split = split_labeled_samples(
        samples,
        block_size=10,
        validation_fraction=0.25,
        random_seed=42,
    )

    assert split.train.n_samples == 1200
    assert split.validation.n_samples == 400
    assert split.train.n_samples + split.validation.n_samples == samples.n_samples
    assert split.pixel_disjoint
    assert set(split.train_blocks).isdisjoint(split.validation_blocks)
    assert len(split.train_blocks) + len(split.validation_blocks) == 16


def test_split_assigns_whole_blocks_to_one_side() -> None:
    samples = _full_grid_samples()

    split = split_labeled_samples(samples, block_size=10, validation_fraction=0.25)

    assert _blocks(split.train, 10) == set(split.train_blocks)
    assert _blocks(split.validation, 10) == set(split.validation_blocks)


def test_split_is_reproducible_with_same_seed() -> None:
    samples = _full_grid_samples()

    first = split_labeled_samples(samples, block_size=10, validation_fraction=0.3, random_seed=7)
    second = split_labeled_samples(samples, block_size=10, validation_fraction=0.3, random_seed=7)

    assert first.validation_blocks == second.validation_blocks
    assert np.array_equal(first.train.rows, second.train.rows)
    assert np.array_equal(first.validation.cols, second.validation.cols)


def test_split_metadata_records_configuration_and_class_distribution() -> None:
    split = split_labeled_samples(
        _full_grid_samples(),
        block_size=10,
        validation_fraction=0.25,
        random_seed=42,
    )

    metadata = split.to_dict()

    assert metadata["config"] == {
        "block_size": 10,
        "validation_fraction": 0.25,
        "random_seed": 42,
    }
    assert metadata["pixel_disjoint"] is True
    assert metadata["blocks"]["train_count"] == len(split.train_blocks)
    assert metadata["blocks"]["validation_count"] == len(split.validation_blocks)
    assert metadata["train"]["n_samples"] == 1200
    assert metadata["validation"]["n_samples"] == 400
    assert sum(
        entry["count"] for entry in metadata["train"]["class_distribution"]
    ) == 1200
    assert sum(
        entry["count"] for entry in metadata["validation"]["class_distribution"]
    ) == 400


def test_save_split_manifest_writes_json(tmp_path: Path) -> None:
    split = split_labeled_samples(_full_grid_samples(), random_seed=11)
    output = tmp_path / "nested" / "split.json"

    result = save_split_manifest(split, output)

    assert result == output.resolve()
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["config"]["random_seed"] == 11
    assert manifest["train"]["n_samples"] == split.train.n_samples
    assert manifest["validation"]["n_samples"] == split.validation.n_samples


def test_split_rejects_invalid_parameters() -> None:
    samples = _full_grid_samples()

    with pytest.raises(DataValidationError, match="block_size must be an integer"):
        split_labeled_samples(samples, block_size=10.5)  # type: ignore[arg-type]
    with pytest.raises(DataValidationError, match="block_size must be >= 1"):
        split_labeled_samples(samples, block_size=0)
    with pytest.raises(DataValidationError, match="between 0 and 1"):
        split_labeled_samples(samples, validation_fraction=0.0)
    with pytest.raises(DataValidationError, match="between 0 and 1"):
        split_labeled_samples(samples, validation_fraction=1.0)
    with pytest.raises(DataValidationError, match="random_seed must be an integer"):
        split_labeled_samples(samples, random_seed=1.5)  # type: ignore[arg-type]
    with pytest.raises(DataValidationError, match="random_seed must be >= 0"):
        split_labeled_samples(samples, random_seed=-1)
    with pytest.raises(DataValidationError, match="Unsupported manifest"):
        save_split_manifest(split_labeled_samples(samples), Path("split.csv"))


def test_split_requires_at_least_two_occupied_blocks() -> None:
    rows = np.array([0, 1, 2], dtype=np.int64)
    cols = np.array([0, 1, 2], dtype=np.int64)
    samples = LabeledSamples(
        X=np.column_stack([rows, cols]).astype(np.float32),
        y=np.array([10, 10, 40], dtype=np.int16),
        rows=rows,
        cols=cols,
        feature_names=("B02", "NDVI"),
        transform=Affine.identity(),
        crs=CRS.from_epsg(32633),
        width=40,
        height=40,
        label_source=Path("labels.tif"),
        total_pixels=1600,
        ignored_feature_pixels=0,
        ignored_nodata_labels=0,
        unknown_label_counts=MappingProxyType({}),
    )

    with pytest.raises(DataValidationError, match="At least two occupied"):
        split_labeled_samples(samples, block_size=10)
