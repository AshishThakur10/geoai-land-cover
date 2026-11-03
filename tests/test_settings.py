from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import get_settings, load_settings
from src.utils.errors import ConfigurationError


def test_defaults_are_resolved_against_project_root(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    settings = load_settings({}, project_root=root)

    assert settings.project_root == root
    assert settings.data_dir == root / "data"
    assert settings.raw_dir == root / "data" / "raw"
    assert settings.processed_dir == root / "data" / "processed"
    assert settings.models_dir == root / "data" / "models"
    assert settings.outputs_dir == root / "data" / "outputs"
    assert settings.sentinel2_dir == root / "data" / "raw" / "sentinel2"
    assert settings.worldcover_dir == root / "data" / "raw" / "worldcover"
    assert settings.aoi_dir == root / "data" / "raw" / "aoi"
    assert settings.log_file == root / "logs" / "geoai.log"
    assert settings.default_aoi is None
    assert settings.log_level == "INFO"
    assert settings.random_seed == 42
    assert settings.n_estimators == 150
    assert settings.max_depth is None
    assert settings.class_weight == "balanced_subsample"
    assert settings.target_resolution == 10.0
    assert settings.max_upload_mb == 200


def test_environment_overrides(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    settings = load_settings(
        {
            "GEOAI_LOG_LEVEL": "debug",
            "GEOAI_LOG_FILE": "custom/logs/app.log",
            "GEOAI_RANDOM_SEED": "7",
            "GEOAI_N_ESTIMATORS": "25",
            "GEOAI_MAX_DEPTH": "12",
            "GEOAI_CLASS_WEIGHT": "balanced",
            "GEOAI_TARGET_RESOLUTION": "20",
            "GEOAI_MAX_UPLOAD_MB": "50",
            "GEOAI_DEFAULT_AOI": "data/raw/aoi/study.geojson",
        },
        project_root=root,
    )

    assert settings.log_level == "DEBUG"
    assert settings.log_file == root / "custom" / "logs" / "app.log"
    assert settings.random_seed == 7
    assert settings.n_estimators == 25
    assert settings.max_depth == 12
    assert settings.class_weight == "balanced"
    assert settings.target_resolution == 20.0
    assert settings.max_upload_mb == 50
    assert settings.default_aoi == root / "data" / "raw" / "aoi" / "study.geojson"


def test_derived_paths_follow_data_directory(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    settings = load_settings({"GEOAI_DATA_DIR": "custom_data"}, project_root=root)

    assert settings.data_dir == root / "custom_data"
    assert settings.raw_dir == root / "custom_data" / "raw"
    assert settings.sentinel2_dir == root / "custom_data" / "raw" / "sentinel2"
    assert settings.classifications_dir == root / "custom_data" / "outputs" / "classifications"


def test_absolute_paths_are_preserved(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    absolute = tmp_path / "absolute_data"
    settings = load_settings({"GEOAI_DATA_DIR": str(absolute)}, project_root=root)

    assert settings.data_dir == absolute


def test_empty_default_aoi_is_none(tmp_path: Path) -> None:
    settings = load_settings({"GEOAI_DEFAULT_AOI": "   "}, project_root=tmp_path)
    assert settings.default_aoi is None


def test_none_class_weight(tmp_path: Path) -> None:
    settings = load_settings({"GEOAI_CLASS_WEIGHT": "none"}, project_root=tmp_path)
    assert settings.class_weight is None


@pytest.mark.parametrize(
    ("environ", "message"),
    [
        ({"GEOAI_LOG_LEVEL": "verbose"}, "LOG_LEVEL"),
        ({"GEOAI_RANDOM_SEED": "abc"}, "RANDOM_SEED"),
        ({"GEOAI_N_ESTIMATORS": "0"}, "N_ESTIMATORS"),
        ({"GEOAI_MAX_DEPTH": "0"}, "MAX_DEPTH"),
        ({"GEOAI_MAX_DEPTH": "deep"}, "MAX_DEPTH"),
        ({"GEOAI_CLASS_WEIGHT": "auto"}, "CLASS_WEIGHT"),
        ({"GEOAI_TARGET_RESOLUTION": "0"}, "TARGET_RESOLUTION"),
        ({"GEOAI_MAX_UPLOAD_MB": "-1"}, "MAX_UPLOAD_MB"),
    ],
)
def test_invalid_values_raise_configuration_error(
    tmp_path: Path, environ: dict[str, str], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        load_settings(environ, project_root=tmp_path)


def test_ensure_directories_creates_all_managed_paths(tmp_path: Path) -> None:
    settings = load_settings({}, project_root=tmp_path)
    settings.ensure_directories()

    for directory in settings.managed_directories():
        assert directory.is_dir(), directory


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
