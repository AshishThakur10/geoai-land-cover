"""Spatially separated train/validation splits and split manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.training.sample_generator import LabeledSamples
from src.utils.errors import DataValidationError
from src.utils.file_utils import ensure_directory
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_BLOCK_SIZE = 10
DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_RANDOM_SEED = 42


def _validate_block_size(block_size: int) -> int:
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise DataValidationError(f"block_size must be an integer, got {block_size!r}")
    if block_size < 1:
        raise DataValidationError(f"block_size must be >= 1, got {block_size}")
    return block_size


def _validate_fraction(validation_fraction: float) -> float:
    fraction = float(validation_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise DataValidationError(
            f"validation_fraction must be between 0 and 1, got {validation_fraction!r}"
        )
    return fraction


def _block_coordinates(
    samples: LabeledSamples,
    block_size: int,
) -> np.ndarray:
    return np.stack(
        [samples.rows // block_size, samples.cols // block_size],
        axis=1,
    ).astype(np.int64, copy=False)


@dataclass(frozen=True)
class SpatialSplit:
    """Block-separated train/validation subsets.

    Whole blocks are assigned to exactly one side, so no pixel can appear in both
    subsets. Adjacent blocks can still be close; add a buffer or larger blocks if
    your analysis requires a strict distance between splits.
    """

    train: LabeledSamples
    validation: LabeledSamples
    block_size: int
    validation_fraction: float
    random_seed: int
    train_blocks: tuple[tuple[int, int], ...]
    validation_blocks: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if set(self.train_blocks) & set(self.validation_blocks):
            raise DataValidationError("Train and validation blocks must be disjoint")
        if not self.train_blocks or not self.validation_blocks:
            raise DataValidationError("Both train and validation blocks must be non-empty")
        if self.train.feature_names != self.validation.feature_names:
            raise DataValidationError("Train and validation feature names must match")
        if (self.train.width, self.train.height) != (
            self.validation.width,
            self.validation.height,
        ):
            raise DataValidationError("Train and validation grids must match")

    @property
    def pixel_disjoint(self) -> bool:
        """Return ``True`` when no spatial coordinate is shared."""
        train_pixels = set(zip(self.train.rows.tolist(), self.train.cols.tolist(), strict=True))
        validation_pixels = set(
            zip(self.validation.rows.tolist(), self.validation.cols.tolist(), strict=True)
        )
        return train_pixels.isdisjoint(validation_pixels)

    def to_dict(self) -> dict[str, object]:
        return {
            "config": {
                "block_size": self.block_size,
                "validation_fraction": self.validation_fraction,
                "random_seed": self.random_seed,
            },
            "blocks": {
                "train_count": len(self.train_blocks),
                "validation_count": len(self.validation_blocks),
                "train": [list(block) for block in self.train_blocks],
                "validation": [list(block) for block in self.validation_blocks],
            },
            "train": self.train.to_dict(),
            "validation": self.validation.to_dict(),
            "pixel_disjoint": self.pixel_disjoint,
        }


def split_labeled_samples(
    samples: LabeledSamples,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> SpatialSplit:
    """Split samples by spatial blocks.

    Blocks containing at least one sample are shuffled with a seeded generator;
    the requested fraction is assigned to validation. Every sample in a block
    stays on the same side, preventing the same pixel (and its immediate
    neighbourhood) from leaking across the split.

    Raises:
        DataValidationError: For invalid parameters or fewer than two occupied
            blocks.
    """
    size = _validate_block_size(block_size)
    fraction = _validate_fraction(validation_fraction)
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise DataValidationError(f"random_seed must be an integer, got {random_seed!r}")
    if random_seed < 0:
        raise DataValidationError(f"random_seed must be >= 0, got {random_seed}")

    block_coordinates = _block_coordinates(samples, size)
    unique_blocks, inverse = np.unique(block_coordinates, axis=0, return_inverse=True)
    block_count = len(unique_blocks)
    if block_count < 2:
        raise DataValidationError(
            f"At least two occupied spatial blocks are required; found {block_count}. "
            "Use a smaller block_size or provide more samples."
        )

    validation_block_count = int(round(block_count * fraction))
    validation_block_count = max(1, min(block_count - 1, validation_block_count))

    rng = np.random.default_rng(random_seed)
    shuffled = rng.permutation(block_count)
    validation_ids = set(shuffled[:validation_block_count].tolist())
    validation_mask = np.fromiter(
        (int(block_id) in validation_ids for block_id in inverse),
        dtype=bool,
        count=samples.n_samples,
    )

    train_blocks = tuple(
        tuple(int(value) for value in unique_blocks[index])
        for index in range(block_count)
        if index not in validation_ids
    )
    validation_blocks = tuple(
        tuple(int(value) for value in unique_blocks[index])
        for index in sorted(validation_ids)
    )

    split = SpatialSplit(
        train=samples.take(np.flatnonzero(~validation_mask)),
        validation=samples.take(np.flatnonzero(validation_mask)),
        block_size=size,
        validation_fraction=fraction,
        random_seed=random_seed,
        train_blocks=train_blocks,
        validation_blocks=validation_blocks,
    )

    train_classes = set(split.train.class_counts)
    validation_classes = set(split.validation.class_counts)
    train_only = sorted(train_classes - validation_classes)
    validation_only = sorted(validation_classes - train_classes)
    if train_only or validation_only:
        logger.warning(
            "Spatial split class coverage differs; train-only=%s validation-only=%s",
            train_only,
            validation_only,
        )

    logger.info(
        "Spatial split: %d train samples in %d blocks, %d validation samples in %d blocks",
        split.train.n_samples,
        len(split.train_blocks),
        split.validation.n_samples,
        len(split.validation_blocks),
    )
    return split


def save_split_manifest(split: SpatialSplit, output_path: str | Path) -> Path:
    """Write split configuration, class distributions, and block IDs as JSON."""
    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() != ".json":
        raise DataValidationError(
            f"Unsupported manifest type {destination.suffix!r}; expected .json"
        )
    ensure_directory(destination.parent)
    destination.write_text(
        json.dumps(split.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote split manifest to %s", destination)
    return destination
