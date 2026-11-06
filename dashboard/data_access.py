"""Dashboard file discovery, upload validation, and metadata loading."""

from __future__ import annotations

import json
import stat
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import uuid4

import numpy as np
from rasterio.enums import Resampling

from config.settings import Settings
from src.data.raster_loader import open_raster, read_metadata
from src.data.vector_loader import SUPPORTED_VECTOR_SUFFIXES
from src.utils.errors import DataValidationError
from src.utils.file_utils import ensure_directory, human_readable_size, safe_filename

RASTER_SUFFIXES: frozenset[str] = frozenset({".tif", ".tiff"})
VECTOR_UPLOAD_SUFFIXES: frozenset[str] = SUPPORTED_VECTOR_SUFFIXES | {".zip"}
JSON_SUFFIXES: frozenset[str] = frozenset({".json"})
MAX_JSON_BYTES = 20 * 1024 * 1024
ACQUISITION_KEYS = (
    "acquisition_date",
    "acquisition_datetime",
    "datetime",
    "date",
    "sensing_time",
    "time",
)


@dataclass(frozen=True)
class FileRecord:
    """A discoverable dashboard input or output file."""

    path: Path
    size_bytes: int
    modified_utc: datetime

    @property
    def size_label(self) -> str:
        return human_readable_size(self.size_bytes)

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "size": self.size_label,
            "modified_utc": self.modified_utc.isoformat(),
        }


def upload_dir(settings: Settings) -> Path:
    """Return and create the dashboard upload directory."""
    return ensure_directory(settings.data_dir / "uploads")


def max_upload_bytes(settings: Settings) -> int:
    return settings.max_upload_mb * 1024 * 1024


def _validate_payload(filename: str, data: bytes, *, suffixes: Iterable[str], maximum: int) -> str:
    name = safe_filename(filename)
    suffix = Path(name).suffix.lower()
    allowed = {value.lower() for value in suffixes}
    if suffix not in allowed:
        options = ", ".join(sorted(allowed))
        raise DataValidationError(
            f"Unsupported upload type {suffix or '(none)'}; expected {options}"
        )
    if not data:
        raise DataValidationError(f"Uploaded file is empty: {name}")
    if len(data) > maximum:
        raise DataValidationError(
            f"Upload exceeds the {human_readable_size(maximum)} limit "
            f"({human_readable_size(len(data))})"
        )
    return name


def _store_payload(filename: str, data: bytes, directory: Path) -> Path:
    safe_name = safe_filename(filename)
    target_dir = ensure_directory(directory)
    suffix = Path(safe_name).suffix.lower()
    stem = Path(safe_name).stem
    destination = target_dir / f"{stem}_{uuid4().hex[:8]}{suffix}"
    try:
        destination.write_bytes(data)
    except OSError as exc:
        raise DataValidationError(f"Unable to store upload {safe_name}: {exc}") from exc
    return destination


def store_upload(
    filename: str,
    data: bytes,
    *,
    directory: Path,
    suffixes: Iterable[str],
    maximum_bytes: int,
) -> Path:
    """Validate and persist an uploaded file without trusting its filename."""
    name = _validate_payload(filename, data, suffixes=suffixes, maximum=maximum_bytes)
    return _store_payload(name, data, directory)


def _safe_zip_members(archive: zipfile.ZipFile, maximum_bytes: int) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    total = 0
    for member in members:
        name = PurePosixPath(member.filename.replace("\\", "/"))
        if name.is_absolute() or ".." in name.parts:
            raise DataValidationError(f"Archive contains an unsafe path: {member.filename}")
        mode = member.external_attr >> 16
        if mode and stat.S_ISLNK(mode):
            raise DataValidationError(f"Archive contains a symbolic link: {member.filename}")
        if member.is_dir():
            continue
        total += member.file_size
        if member.file_size > maximum_bytes or total > maximum_bytes:
            raise DataValidationError("Archive expands beyond the configured upload limit")
    return members


def extract_vector_archive(archive_path: Path, *, maximum_bytes: int) -> Path:
    """Extract a zipped vector dataset and return its single primary layer."""
    try:
        with zipfile.ZipFile(archive_path) as archive:
            _safe_zip_members(archive, maximum_bytes)
            target = ensure_directory(
                archive_path.with_name(f"{archive_path.stem}_{uuid4().hex[:8]}")
            )
            archive.extractall(target)
    except zipfile.BadZipFile as exc:
        raise DataValidationError(f"Invalid ZIP archive: {archive_path.name}") from exc
    except OSError as exc:
        raise DataValidationError(f"Unable to extract {archive_path.name}: {exc}") from exc

    candidates = sorted(
        path
        for path in target.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_VECTOR_SUFFIXES
    )
    if not candidates:
        raise DataValidationError("ZIP archive contains no supported vector layer")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates[:5])
        raise DataValidationError(
            f"ZIP archive must contain exactly one vector layer; found: {names}"
        )
    return candidates[0]


def prepare_vector_upload(
    filename: str,
    data: bytes,
    settings: Settings,
) -> Path:
    """Persist a vector upload, extracting zipped Shapefiles when needed."""
    maximum = max_upload_bytes(settings)
    stored = store_upload(
        filename,
        data,
        directory=upload_dir(settings),
        suffixes=VECTOR_UPLOAD_SUFFIXES,
        maximum_bytes=maximum,
    )
    if stored.suffix.lower() == ".zip":
        return extract_vector_archive(stored, maximum_bytes=maximum)
    return stored


def prepare_raster_upload(filename: str, data: bytes, settings: Settings) -> Path:
    """Persist a GeoTIFF upload."""
    return store_upload(
        filename,
        data,
        directory=upload_dir(settings),
        suffixes=RASTER_SUFFIXES,
        maximum_bytes=max_upload_bytes(settings),
    )


def prepare_json_upload(filename: str, data: bytes, settings: Settings) -> Path:
    """Persist a JSON report/model-metadata upload."""
    return store_upload(
        filename,
        data,
        directory=upload_dir(settings),
        suffixes=JSON_SUFFIXES,
        maximum_bytes=max_upload_bytes(settings),
    )


def discover_files(
    directories: Sequence[Path],
    *,
    suffixes: Iterable[str],
    recursive: bool = True,
) -> list[FileRecord]:
    """List files under existing directories, newest first."""
    allowed = {value.lower() for value in suffixes}
    seen: set[Path] = set()
    records: list[FileRecord] = []
    for directory in directories:
        root = Path(directory)
        if not root.is_dir():
            continue
        iterator = root.rglob("*") if recursive else root.glob("*")
        for path in iterator:
            try:
                if not path.is_file() or path.suffix.lower() not in allowed:
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                info = resolved.stat()
            except OSError:
                continue
            records.append(
                FileRecord(
                    path=resolved,
                    size_bytes=info.st_size,
                    modified_utc=datetime.fromtimestamp(info.st_mtime, tz=UTC),
                )
            )
    return sorted(records, key=lambda item: (item.modified_utc, str(item.path)), reverse=True)


def raster_files(settings: Settings, *, include_classifications: bool = False) -> list[FileRecord]:
    directories = [
        settings.processed_imagery_dir,
        settings.sentinel2_dir,
        upload_dir(settings),
    ]
    if include_classifications:
        directories.append(settings.classifications_dir)
    return discover_files(directories, suffixes=RASTER_SUFFIXES)


def classification_files(settings: Settings) -> list[FileRecord]:
    return discover_files(
        [settings.classifications_dir, upload_dir(settings)],
        suffixes=RASTER_SUFFIXES,
    )


def change_files(settings: Settings) -> list[FileRecord]:
    return discover_files(
        [settings.change_detection_dir, upload_dir(settings)],
        suffixes=RASTER_SUFFIXES,
    )


def vector_files(settings: Settings) -> list[FileRecord]:
    return discover_files(
        [settings.aoi_dir, upload_dir(settings)],
        suffixes=SUPPORTED_VECTOR_SUFFIXES,
    )


def json_files(settings: Settings) -> list[FileRecord]:
    return discover_files(
        [settings.outputs_dir, settings.models_dir, upload_dir(settings)],
        suffixes=JSON_SUFFIXES,
    )


def read_json_object(path: str | Path, *, maximum_bytes: int = MAX_JSON_BYTES) -> dict[str, object]:
    """Read a bounded JSON object."""
    source = Path(path).expanduser().resolve()
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise DataValidationError(f"Unable to inspect JSON file {source}: {exc}") from exc
    if size > maximum_bytes:
        raise DataValidationError(
            f"JSON file exceeds the {human_readable_size(maximum_bytes)} limit: {source.name}"
        )
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"Unable to read JSON file {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise DataValidationError(f"JSON file must contain an object: {source.name}")
    return value


def _matching_json(
    directory_files: Sequence[FileRecord],
    required: Iterable[str],
) -> list[FileRecord]:
    matches: list[FileRecord] = []
    for record in directory_files:
        try:
            value = read_json_object(record.path)
        except DataValidationError:
            continue
        if all(key in value for key in required):
            matches.append(record)
    return matches


def evaluation_reports(settings: Settings) -> list[FileRecord]:
    """Find JSON reports containing evaluation sections."""
    return _matching_json(json_files(settings), ("metrics", "confusion_matrix", "per_class"))


def model_metadata_files(settings: Settings) -> list[FileRecord]:
    """Find JSON files containing model metadata."""
    return _matching_json(json_files(settings), ("format_version", "feature_names"))


def read_evaluation_report(path: str | Path) -> dict[str, object]:
    """Read and minimally validate an evaluation report."""
    report = read_json_object(path)
    required = ("metrics", "confusion_matrix", "per_class")
    missing = [key for key in required if key not in report]
    if missing:
        raise DataValidationError(
            f"Evaluation report is missing required sections: {', '.join(missing)}"
        )
    if not isinstance(report["metrics"], Mapping):
        raise DataValidationError("Evaluation report metrics must be an object")
    return report


def read_model_metadata(path: str | Path) -> dict[str, object]:
    """Read and minimally validate model metadata."""
    metadata = read_json_object(path)
    if "format_version" not in metadata or "feature_names" not in metadata:
        raise DataValidationError("Model metadata must contain format_version and feature_names")
    return metadata


def raster_tags(path: str | Path) -> dict[str, str]:
    """Read dataset-level raster tags."""
    from src.data.raster_loader import open_raster

    with open_raster(path) as dataset:
        return {str(key): str(value) for key, value in dataset.tags().items()}


def acquisition_date(tags: Mapping[str, str]) -> str | None:
    """Extract an acquisition date from actual raster tags, never guessed."""
    normalized = {key.lower(): value for key, value in tags.items()}
    for key in ACQUISITION_KEYS:
        value = normalized.get(key)
        if value and value.strip():
            return value.strip()
    return None


def raster_summary(path: str | Path) -> dict[str, object]:
    """Return JSON-safe raster metadata plus tags for display and reports."""
    metadata = read_metadata(path)
    summary = metadata.to_dict()
    summary["tags"] = raster_tags(path)
    summary["acquisition_date"] = acquisition_date(summary["tags"])
    return summary


def read_band_preview(
    path: str | Path,
    band_index: int = 1,
    *,
    max_size: int = 1400,
) -> np.ma.MaskedArray:
    """Read a downsampled raster band for dashboard display."""
    metadata = read_metadata(path)
    if band_index < 1 or band_index > metadata.count:
        raise DataValidationError(
            f"Band index {band_index} is out of range for {metadata.path.name}"
        )
    if max_size < 1:
        raise DataValidationError(f"max_size must be >= 1, got {max_size}")
    scale = min(1.0, max_size / max(metadata.width, metadata.height))
    height = max(1, int(round(metadata.height * scale)))
    width = max(1, int(round(metadata.width * scale)))
    with open_raster(path) as dataset:
        return dataset.read(
            band_index,
            out_shape=(height, width),
            masked=True,
            resampling=Resampling.bilinear,
        )


__all__ = [
    "ACQUISITION_KEYS",
    "FileRecord",
    "JSON_SUFFIXES",
    "RASTER_SUFFIXES",
    "VECTOR_UPLOAD_SUFFIXES",
    "acquisition_date",
    "change_files",
    "classification_files",
    "discover_files",
    "evaluation_reports",
    "extract_vector_archive",
    "json_files",
    "max_upload_bytes",
    "model_metadata_files",
    "prepare_json_upload",
    "prepare_raster_upload",
    "prepare_vector_upload",
    "raster_files",
    "raster_summary",
    "raster_tags",
    "read_band_preview",
    "read_evaluation_report",
    "read_json_object",
    "read_model_metadata",
    "store_upload",
    "upload_dir",
    "vector_files",
]
