"""Windowed land-cover change detection and transition accounting."""

from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from config.land_cover_classes import is_valid_class_code
from src.analysis.area_utils import (
    AreaPlan,
    resolve_area_plan,
    valid_values,
    validate_class_nodata,
)
from src.analysis.transition_matrix import (
    CHANGE_CAVEAT,
    TransitionMatrix,
)
from src.data.data_metadata import RasterMetadata
from src.data.raster_loader import (
    assert_same_grid,
    ensure_georeferenced,
    open_raster,
    read_metadata,
)
from src.utils.errors import DataValidationError, RasterError
from src.utils.file_utils import ensure_directory
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_OUTPUT_SUFFIXES: frozenset[str] = frozenset({".tif", ".tiff"})
SUPPORTED_ALIGNMENT: frozenset[str] = frozenset({"strict", "reproject"})
UNCHANGED_VALUE = 0
CHANGED_VALUE = 1
CHANGE_NODATA = 255
CHANGE_DESCRIPTION = "Land-cover change (0 unchanged, 1 changed)"
CHANGE_LEGEND = (
    {"value": UNCHANGED_VALUE, "name": "Unchanged", "color": "#D9D9D9"},
    {"value": CHANGED_VALUE, "name": "Changed", "color": "#DC1E1E"},
)


@dataclass(frozen=True)
class ChangeDetectionResult:
    """Metadata, summary, and transition matrix for a change analysis."""

    earlier_path: Path
    later_path: Path
    output_path: Path
    earlier_band_index: int
    later_band_index: int
    crs: CRS
    transform: Affine
    width: int
    height: int
    nodata: int
    alignment: str
    window_size: int
    matrix: TransitionMatrix

    @property
    def total_pixels(self) -> int:
        return self.width * self.height

    @property
    def valid_pixels(self) -> int:
        return self.matrix.valid_pixels

    @property
    def nodata_pixels(self) -> int:
        return self.total_pixels - self.valid_pixels

    @property
    def changed_pixels(self) -> int:
        return self.matrix.changed_pixels

    @property
    def unchanged_pixels(self) -> int:
        return self.matrix.unchanged_pixels

    @property
    def changed_percentage(self) -> float:
        return self.matrix.changed_percentage

    @property
    def total_area_km2(self) -> float:
        return self.matrix.total_area_km2

    @property
    def changed_area_km2(self) -> float:
        return self.matrix.changed_area_km2

    def to_dict(self) -> dict[str, object]:
        return {
            "earlier_path": str(self.earlier_path),
            "later_path": str(self.later_path),
            "output_path": str(self.output_path),
            "earlier_band_index": self.earlier_band_index,
            "later_band_index": self.later_band_index,
            "crs": self.crs.to_string(),
            "transform": list(self.transform)[:6],
            "width": self.width,
            "height": self.height,
            "nodata": self.nodata,
            "alignment": self.alignment,
            "window_size": self.window_size,
            "values": {
                "unchanged": UNCHANGED_VALUE,
                "changed": CHANGED_VALUE,
                "nodata": self.nodata,
            },
            "change_legend": list(CHANGE_LEGEND),
            "total_pixels": self.total_pixels,
            "valid_pixels": self.valid_pixels,
            "nodata_pixels": self.nodata_pixels,
            "changed_pixels": self.changed_pixels,
            "unchanged_pixels": self.unchanged_pixels,
            "changed_percentage": self.changed_percentage,
            "total_area_km2": self.total_area_km2,
            "changed_area_km2": self.changed_area_km2,
            "transition_matrix": self.matrix.to_dict(),
            "scientific_caveat": CHANGE_CAVEAT,
        }


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(f"{name} must be an integer, got {value!r}")
    if value < 1:
        raise DataValidationError(f"{name} must be >= 1, got {value}")
    return value


def _validate_alignment(value: str) -> str:
    alignment = str(value).strip().lower()
    if alignment not in SUPPORTED_ALIGNMENT:
        allowed = ", ".join(sorted(SUPPORTED_ALIGNMENT))
        raise DataValidationError(
            f"Unsupported alignment mode {value!r}; expected {allowed}"
        )
    return alignment


def _validate_output_path(output_path: str | Path) -> Path:
    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() not in SUPPORTED_OUTPUT_SUFFIXES:
        raise DataValidationError(
            f"Unsupported change raster type {destination.suffix!r}; expected .tif or .tiff"
        )
    ensure_directory(destination.parent)
    return destination


def _validate_classification(
    metadata: RasterMetadata,
    *,
    band_index: int,
    label: str,
) -> None:
    ensure_georeferenced(metadata)
    if band_index > metadata.count:
        raise DataValidationError(
            f"{label} band index {band_index} is out of range for "
            f"{metadata.path.name} ({metadata.count} band(s))"
        )
    dtype = np.dtype(metadata.dtypes[band_index - 1])
    if not np.issubdtype(dtype, np.integer):
        raise DataValidationError(
            f"{label} classification band must use an integer dtype, got {dtype}"
        )


def _validate_codes(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    label: str,
) -> None:
    observed = np.unique(values[valid])
    unknown = [int(code) for code in observed if not is_valid_class_code(int(code))]
    if unknown:
        raise DataValidationError(
            f"{label} classification contains unknown class codes: {unknown}"
        )


def _read_window(
    dataset: rasterio.io.DatasetReader,
    *,
    band_index: int,
    window: Window,
    nodata: int | float | None,
) -> tuple[np.ndarray, np.ndarray]:
    values = dataset.read(band_index, window=window)
    return values, valid_values(values, nodata)


def _write_change_metadata(
    dataset: rasterio.io.DatasetWriter,
    *,
    earlier_path: Path,
    later_path: Path,
    earlier_band_index: int,
    later_band_index: int,
    alignment: str,
) -> None:
    dataset.set_band_description(1, CHANGE_DESCRIPTION)
    dataset.update_tags(
        EARLIER_SOURCE=earlier_path.name,
        LATER_SOURCE=later_path.name,
        EARLIER_BAND=str(earlier_band_index),
        LATER_BAND=str(later_band_index),
        ALIGNMENT=alignment,
        UNCHANGED_VALUE=str(UNCHANGED_VALUE),
        CHANGED_VALUE=str(CHANGED_VALUE),
        NODATA_VALUE=str(CHANGE_NODATA),
        CHANGE_LEGEND=json.dumps(CHANGE_LEGEND, separators=(",", ":")),
        SCIENTIFIC_CAVEAT=CHANGE_CAVEAT,
    )
    colormap = dict.fromkeys(range(256), (0, 0, 0, 0))
    colormap[UNCHANGED_VALUE] = (217, 217, 217, 255)
    colormap[CHANGED_VALUE] = (220, 30, 30, 255)
    dataset.write_colormap(1, colormap)


def _profile(
    *,
    width: int,
    height: int,
    crs: CRS,
    transform: Affine,
) -> dict[str, object]:
    return {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": CHANGE_NODATA,
        "compress": "deflate",
    }


def _pair_counts(
    earlier_values: np.ndarray,
    later_values: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.any(valid):
        empty = np.empty((0, 2), dtype=np.int64)
        return empty, np.empty(0, dtype=np.int64)
    pairs = np.column_stack((earlier_values[valid], later_values[valid]))
    unique, counts = np.unique(pairs, axis=0, return_counts=True)
    return unique.astype(np.int64, copy=False), counts.astype(np.int64, copy=False)


def _accumulate_counts(
    pairs: np.ndarray,
    counts: np.ndarray,
    *,
    row_area_km2: float,
    counts_by_pair: dict[tuple[int, int], int],
    areas_by_pair: dict[tuple[int, int], float],
) -> None:
    for pair, count in zip(pairs, counts, strict=True):
        key = (int(pair[0]), int(pair[1]))
        count_int = int(count)
        counts_by_pair[key] = counts_by_pair.get(key, 0) + count_int
        if row_area_km2:
            areas_by_pair[key] = areas_by_pair.get(key, 0.0) + (
                count_int * row_area_km2
            )


def _build_matrix(
    *,
    earlier_path: Path,
    later_path: Path,
    earlier_classes: set[int],
    later_classes: set[int],
    counts_by_pair: dict[tuple[int, int], int],
    areas_by_pair: dict[tuple[int, int], float],
    plan: AreaPlan,
    valid_pixels: int,
    changed_pixels: int,
) -> TransitionMatrix:
    classes = tuple(sorted(earlier_classes | later_classes))
    if not classes:
        raise DataValidationError(
            "Classifications contain no valid land-cover classes in common"
        )

    counts: dict[int, dict[int, int]] = {
        from_code: {
            to_code: int(counts_by_pair.get((from_code, to_code), 0))
            for to_code in classes
        }
        for from_code in classes
    }
    areas: dict[int, dict[int, float]] = {
        from_code: {
            to_code: float(areas_by_pair.get((from_code, to_code), 0.0))
            for to_code in classes
        }
        for from_code in classes
    }
    if plan.area_method == "projected":
        assert plan.pixel_area_km2 is not None
        areas = {
            from_code: {
                to_code: count * plan.pixel_area_km2
                for to_code, count in counts[from_code].items()
            }
            for from_code in classes
        }

    return TransitionMatrix(
        earlier_path=earlier_path,
        later_path=later_path,
        classes=classes,
        counts=counts,
        areas_km2=areas,
        area_method=plan.area_method,
        pixel_area_km2=plan.pixel_area_km2,
        valid_pixels=valid_pixels,
        changed_pixels=changed_pixels,
    )


def detect_changes(
    earlier_path: str | Path,
    later_path: str | Path,
    output_path: str | Path,
    *,
    earlier_band_index: int = 1,
    later_band_index: int = 1,
    alignment: str = "strict",
    earlier_nodata: int | float | None = None,
    later_nodata: int | float | None = None,
    window_size: int = 1024,
) -> ChangeDetectionResult:
    """Compare two classifications and write a georeferenced change raster.

    ``alignment="strict"`` requires identical CRS, transform, and dimensions.
    ``alignment="reproject"`` warps the later classification onto the earlier
    grid using nearest-neighbour resampling so class codes are preserved.
    """
    earlier_band = _positive_integer(earlier_band_index, "earlier_band_index")
    later_band = _positive_integer(later_band_index, "later_band_index")
    size = _positive_integer(window_size, "window_size")
    mode = _validate_alignment(alignment)
    destination = _validate_output_path(output_path)

    earlier_metadata = read_metadata(earlier_path)
    later_metadata = read_metadata(later_path)
    _validate_classification(
        earlier_metadata,
        band_index=earlier_band,
        label="Earlier",
    )
    _validate_classification(
        later_metadata,
        band_index=later_band,
        label="Later",
    )
    if mode == "strict":
        try:
            assert_same_grid(earlier_metadata, later_metadata)
        except DataValidationError as exc:
            raise DataValidationError(
                f"{exc}; use alignment='reproject' to warp the later "
                "classification onto the earlier grid"
            ) from exc

    earlier_resolved_nodata = validate_class_nodata(
        earlier_metadata.nodata if earlier_nodata is None else earlier_nodata,
        label="earlier_nodata",
    )
    later_resolved_nodata = validate_class_nodata(
        later_metadata.nodata if later_nodata is None else later_nodata,
        label="later_nodata",
    )
    crs = earlier_metadata.crs
    assert crs is not None
    plan = resolve_area_plan(earlier_metadata)
    width = earlier_metadata.width
    height = earlier_metadata.height
    counts_by_pair: dict[tuple[int, int], int] = {}
    areas_by_pair: dict[tuple[int, int], float] = {}
    earlier_classes: set[int] = set()
    later_classes: set[int] = set()
    valid_pixels = 0
    changed_pixels = 0

    try:
        with ExitStack() as stack:
            earlier_source = stack.enter_context(open_raster(earlier_path))
            later_source = stack.enter_context(open_raster(later_path))
            if mode == "reproject":
                later_dataset = stack.enter_context(
                    WarpedVRT(
                        later_source,
                        crs=crs,
                        transform=earlier_metadata.transform,
                        width=width,
                        height=height,
                        src_nodata=later_resolved_nodata,
                        nodata=CHANGE_NODATA,
                        resampling=Resampling.nearest,
                        init_dest_nodata=True,
                    )
                )
                later_effective_nodata: int | float | None = CHANGE_NODATA
            else:
                later_dataset = later_source
                later_effective_nodata = later_resolved_nodata

            with rasterio.open(destination, "w", **_profile(
                width=width,
                height=height,
                crs=crs,
                transform=earlier_metadata.transform,
            )) as output:
                _write_change_metadata(
                    output,
                    earlier_path=earlier_metadata.path,
                    later_path=later_metadata.path,
                    earlier_band_index=earlier_band,
                    later_band_index=later_band,
                    alignment=mode,
                )

                for row_off in range(0, height, size):
                    for col_off in range(0, width, size):
                        window_height = min(size, height - row_off)
                        window_width = min(size, width - col_off)
                        window = Window(col_off, row_off, window_width, window_height)
                        earlier_values, earlier_valid = _read_window(
                            earlier_source,
                            band_index=earlier_band,
                            window=window,
                            nodata=earlier_resolved_nodata,
                        )
                        later_values, later_valid = _read_window(
                            later_dataset,
                            band_index=later_band,
                            window=window,
                            nodata=later_effective_nodata,
                        )
                        _validate_codes(
                            earlier_values,
                            earlier_valid,
                            label="Earlier",
                        )
                        _validate_codes(
                            later_values,
                            later_valid,
                            label="Later",
                        )

                        pair_valid = earlier_valid & later_valid
                        changed = pair_valid & (earlier_values != later_values)
                        change_values = np.full(pair_valid.shape, CHANGE_NODATA, dtype=np.uint8)
                        change_values[pair_valid] = UNCHANGED_VALUE
                        change_values[changed] = CHANGED_VALUE
                        output.write(change_values, 1, window=window)

                        valid_pixels += int(np.count_nonzero(pair_valid))
                        changed_pixels += int(np.count_nonzero(changed))
                        earlier_classes.update(
                            int(code)
                            for code in np.unique(earlier_values[earlier_valid])
                        )
                        later_classes.update(
                            int(code)
                            for code in np.unique(later_values[later_valid])
                        )

                        if plan.area_method == "projected":
                            pairs, counts = _pair_counts(
                                earlier_values,
                                later_values,
                                pair_valid,
                            )
                            _accumulate_counts(
                                pairs,
                                counts,
                                row_area_km2=0.0,
                                counts_by_pair=counts_by_pair,
                                areas_by_pair=areas_by_pair,
                            )
                        else:
                            for local_row in range(window_height):
                                row_valid = pair_valid[local_row]
                                if not np.any(row_valid):
                                    continue
                                pairs, counts = _pair_counts(
                                    earlier_values[local_row],
                                    later_values[local_row],
                                    row_valid,
                                )
                                _accumulate_counts(
                                    pairs,
                                    counts,
                                    row_area_km2=plan.row_pixel_area_km2(
                                        earlier_metadata.transform,
                                        row_off + local_row,
                                        width,
                                    ),
                                    counts_by_pair=counts_by_pair,
                                    areas_by_pair=areas_by_pair,
                                )
    except (RasterioIOError, OSError, ValueError) as exc:
        raise RasterError(f"Unable to detect land-cover changes: {exc}") from exc

    if valid_pixels == 0:
        raise DataValidationError(
            "Classifications contain no pixels valid in both rasters"
        )

    matrix = _build_matrix(
        earlier_path=earlier_metadata.path,
        later_path=later_metadata.path,
        earlier_classes=earlier_classes,
        later_classes=later_classes,
        counts_by_pair=counts_by_pair,
        areas_by_pair=areas_by_pair,
        plan=plan,
        valid_pixels=valid_pixels,
        changed_pixels=changed_pixels,
    )
    result = ChangeDetectionResult(
        earlier_path=earlier_metadata.path,
        later_path=later_metadata.path,
        output_path=destination,
        earlier_band_index=earlier_band,
        later_band_index=later_band,
        crs=crs,
        transform=earlier_metadata.transform,
        width=width,
        height=height,
        nodata=CHANGE_NODATA,
        alignment=mode,
        window_size=size,
        matrix=matrix,
    )
    logger.info(
        "Detected %d changed pixels (%.2f%%) across %d classes in %s",
        result.changed_pixels,
        result.changed_percentage,
        len(matrix.classes),
        destination,
    )
    return result
