"""Application settings loaded from environment variables.

Values are read from variables prefixed with ``GEOAI_``. A local ``.env`` file
is loaded when :func:`get_settings` is used, but tests and library callers can
use :func:`load_settings` with an explicit mapping for deterministic behaviour.

Relative paths are always resolved against the project root, never against the
current working directory.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from src.utils.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PREFIX = "GEOAI_"

VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
VALID_CLASS_WEIGHTS = frozenset({"balanced", "balanced_subsample"})


def _read(environ: Mapping[str, str], key: str, default: str | None) -> str | None:
    return environ.get(f"{ENV_PREFIX}{key}", default)


def _resolve_path(value: str | None, project_root: Path) -> Path | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _parse_int(
    environ: Mapping[str, str],
    key: str,
    default: int,
    *,
    minimum: int | None = None,
) -> int:
    raw = _read(environ, key, str(default))
    try:
        value = int(str(raw))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{ENV_PREFIX}{key} must be an integer, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{ENV_PREFIX}{key} must be >= {minimum}, got {value}")
    return value


def _parse_optional_int(
    environ: Mapping[str, str],
    key: str,
    minimum: int | None = None,
) -> int | None:
    raw = _read(environ, key, "")
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"{ENV_PREFIX}{key} must be an integer or empty, got {raw!r}"
        ) from exc
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{ENV_PREFIX}{key} must be >= {minimum}, got {value}")
    return value


def _parse_float(
    environ: Mapping[str, str],
    key: str,
    default: float,
    *,
    minimum_exclusive: float | None = None,
) -> float:
    raw = _read(environ, key, str(default))
    try:
        value = float(str(raw))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{ENV_PREFIX}{key} must be a number, got {raw!r}") from exc
    if minimum_exclusive is not None and value <= minimum_exclusive:
        raise ConfigurationError(f"{ENV_PREFIX}{key} must be > {minimum_exclusive}, got {value}")
    return value


def _parse_log_level(environ: Mapping[str, str]) -> str:
    raw = _read(environ, "LOG_LEVEL", "INFO")
    level = str(raw).strip().upper()
    if level not in VALID_LOG_LEVELS:
        allowed = ", ".join(sorted(VALID_LOG_LEVELS))
        raise ConfigurationError(f"{ENV_PREFIX}LOG_LEVEL must be one of {allowed}, got {raw!r}")
    return level


def _parse_class_weight(environ: Mapping[str, str]) -> str | None:
    raw = _read(environ, "CLASS_WEIGHT", "balanced_subsample")
    value = str(raw).strip()
    if not value or value.lower() == "none":
        return None
    if value not in VALID_CLASS_WEIGHTS:
        allowed = ", ".join(sorted(VALID_CLASS_WEIGHTS))
        raise ConfigurationError(
            f"{ENV_PREFIX}CLASS_WEIGHT must be one of {allowed}, 'none', or empty; got {raw!r}"
        )
    return value


@dataclass(frozen=True)
class Settings:
    """Immutable application configuration."""

    project_root: Path
    log_level: str
    log_file: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    outputs_dir: Path
    sentinel2_dir: Path
    worldcover_dir: Path
    aoi_dir: Path
    default_aoi: Path | None
    random_seed: int
    n_estimators: int
    max_depth: int | None
    class_weight: str | None
    target_resolution: float
    max_upload_mb: int

    @property
    def classifications_dir(self) -> Path:
        return self.outputs_dir / "classifications"

    @property
    def change_detection_dir(self) -> Path:
        return self.outputs_dir / "change_detection"

    @property
    def reports_dir(self) -> Path:
        return self.outputs_dir / "reports"

    @property
    def statistics_dir(self) -> Path:
        return self.outputs_dir / "statistics"

    @property
    def features_dir(self) -> Path:
        return self.processed_dir / "features"

    @property
    def labels_dir(self) -> Path:
        return self.processed_dir / "labels"

    @property
    def processed_imagery_dir(self) -> Path:
        return self.processed_dir / "imagery"

    @property
    def logs_dir(self) -> Path:
        return self.log_file.parent

    def managed_directories(self) -> tuple[Path, ...]:
        """Return all directories the application expects to exist."""
        return (
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.models_dir,
            self.outputs_dir,
            self.sentinel2_dir,
            self.worldcover_dir,
            self.aoi_dir,
            self.processed_imagery_dir,
            self.features_dir,
            self.labels_dir,
            self.classifications_dir,
            self.change_detection_dir,
            self.reports_dir,
            self.statistics_dir,
            self.logs_dir,
        )

    def ensure_directories(self) -> None:
        """Create all managed directories, including parents."""
        for directory in self.managed_directories():
            directory.mkdir(parents=True, exist_ok=True)


def load_settings(
    environ: Mapping[str, str] | None = None,
    project_root: Path | None = None,
) -> Settings:
    """Build settings from an environment mapping.

    Args:
        environ: Mapping to read values from. Defaults to ``os.environ``.
        project_root: Root used to resolve relative paths. Defaults to the
            repository root containing this package.
    """
    source = os.environ if environ is None else environ
    root = (project_root or PROJECT_ROOT).resolve()

    data_dir = _resolve_path(_read(source, "DATA_DIR", "data"), root)
    assert data_dir is not None  # default is non-empty

    raw_dir = _resolve_path(_read(source, "RAW_DIR", str(data_dir / "raw")), root)
    processed_dir = _resolve_path(_read(source, "PROCESSED_DIR", str(data_dir / "processed")), root)
    models_dir = _resolve_path(_read(source, "MODELS_DIR", str(data_dir / "models")), root)
    outputs_dir = _resolve_path(_read(source, "OUTPUTS_DIR", str(data_dir / "outputs")), root)
    sentinel2_dir = _resolve_path(
        _read(source, "SENTINEL2_DIR", str((raw_dir or data_dir) / "sentinel2")), root
    )
    worldcover_dir = _resolve_path(
        _read(source, "WORLDCOVER_DIR", str((raw_dir or data_dir) / "worldcover")), root
    )
    aoi_dir = _resolve_path(_read(source, "AOI_DIR", str((raw_dir or data_dir) / "aoi")), root)
    log_file = _resolve_path(_read(source, "LOG_FILE", "logs/geoai.log"), root)
    assert log_file is not None

    assert raw_dir is not None
    assert processed_dir is not None
    assert models_dir is not None
    assert outputs_dir is not None
    assert sentinel2_dir is not None
    assert worldcover_dir is not None
    assert aoi_dir is not None

    return Settings(
        project_root=root,
        log_level=_parse_log_level(source),
        log_file=log_file,
        data_dir=data_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        models_dir=models_dir,
        outputs_dir=outputs_dir,
        sentinel2_dir=sentinel2_dir,
        worldcover_dir=worldcover_dir,
        aoi_dir=aoi_dir,
        default_aoi=_resolve_path(_read(source, "DEFAULT_AOI", ""), root),
        random_seed=_parse_int(source, "RANDOM_SEED", 42, minimum=0),
        n_estimators=_parse_int(source, "N_ESTIMATORS", 150, minimum=1),
        max_depth=_parse_optional_int(source, "MAX_DEPTH", minimum=1),
        class_weight=_parse_class_weight(source),
        target_resolution=_parse_float(source, "TARGET_RESOLUTION", 10.0, minimum_exclusive=0.0),
        max_upload_mb=_parse_int(source, "MAX_UPLOAD_MB", 200, minimum=1),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings, loading ``.env`` once (without overriding env vars)."""
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
    return load_settings()
