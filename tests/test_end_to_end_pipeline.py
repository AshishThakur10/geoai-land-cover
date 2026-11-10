from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS

from config.settings import load_settings
from dashboard import data_access
from src.analysis.area_statistics import compute_area_statistics, save_area_statistics_csv
from src.analysis.change_detection import detect_changes
from src.analysis.transition_matrix import save_transition_matrix_csv
from src.classification.predictor import classify_feature_raster
from src.features.feature_extraction import FeatureStack, save_feature_stack
from src.reports.report_generator import analysis_report_json, build_analysis_report
from src.training.dataset_split import save_split_manifest, split_labeled_samples
from src.training.evaluation import evaluate_random_forest, save_evaluation_report
from src.training.model_manager import ModelRunMetadata, load_model, save_model
from src.training.random_forest import RandomForestConfig, train_random_forest
from src.training.sample_generator import generate_labeled_samples
from tests.fixtures import (
    RASTER_CRS,
    default_transform,
    synthetic_ml_samples,
    synthetic_worldcover_codes,
    write_synthetic_aoi_geojson,
    write_synthetic_worldcover,
)


def _synthetic_feature_stack(codes: np.ndarray) -> FeatureStack:
    samples = synthetic_ml_samples()
    means = {
        int(code): samples.X[samples.y == code].mean(axis=0)
        for code in np.unique(codes)
    }
    rng = np.random.default_rng(2026)
    data = np.empty((samples.n_features, *codes.shape), dtype=np.float32)
    for code, mean in means.items():
        mask = codes == code
        data[:, mask] = mean[:, np.newaxis] + rng.normal(
            0.0, 0.02, size=(samples.n_features, int(mask.sum()))
        )
    data[:, 0, 0] = np.nan
    data[:, 5, 7] = np.nan
    valid = np.all(np.isfinite(data), axis=0)
    return FeatureStack(
        data=data,
        feature_names=samples.feature_names,
        transform=default_transform(),
        crs=CRS.from_string(RASTER_CRS),
        valid_mask=valid,
        band_names=("B02", "B03", "B04", "B08"),
    )


def test_synthetic_pipeline_to_dashboard_outputs(tmp_path: Path) -> None:
    settings = load_settings(
        {
            "GEOAI_DATA_DIR": str(tmp_path / "data"),
            "GEOAI_LOG_FILE": str(tmp_path / "logs" / "geoai.log"),
        },
        project_root=tmp_path,
    )
    settings.ensure_directories()

    aoi = write_synthetic_aoi_geojson(settings.aoi_dir / "study.geojson")
    codes = synthetic_worldcover_codes()
    labels = write_synthetic_worldcover(settings.worldcover_dir / "worldcover.tif", codes=codes)
    features = _synthetic_feature_stack(codes)
    feature_path = save_feature_stack(features, settings.processed_imagery_dir / "features.tif")

    samples = generate_labeled_samples(features, labels)
    split = split_labeled_samples(
        samples,
        block_size=10,
        validation_fraction=0.5,
        random_seed=42,
    )
    save_split_manifest(split, settings.reports_dir / "split_manifest.json")
    trained = train_random_forest(
        split.train,
        RandomForestConfig(n_estimators=51, max_depth=8, random_state=42, n_jobs=1),
    )
    evaluation = evaluate_random_forest(trained, split.validation)
    evaluation_path = save_evaluation_report(
        evaluation,
        settings.reports_dir / "evaluation.json",
    )
    artifacts = save_model(
        trained,
        settings.models_dir / "random_forest.joblib",
        run_metadata=ModelRunMetadata(
            input_paths=(str(feature_path),),
            aoi_path=str(aoi),
            label_path=str(labels),
            study_area="Synthetic integration fixture",
            notes="Synthetic test data; not a real satellite observation.",
        ),
        evaluation=evaluation,
    )

    loaded = load_model(artifacts.model_path)
    classification = classify_feature_raster(
        loaded,
        feature_path,
        settings.classifications_dir / "land_cover_2021.tif",
        window_size=16,
    )
    statistics = compute_area_statistics(classification.output_path)
    statistics_path = save_area_statistics_csv(
        statistics,
        settings.statistics_dir / "land_cover_2021_area.csv",
    )

    assert evaluation.accuracy >= 0.8
    assert classification.valid_pixels == 1598
    assert statistics.valid_pixels == classification.valid_pixels
    assert statistics.total_area_km2 == pytest.approx(0.1598)
    assert statistics_path.is_file()

    with rasterio.open(classification.output_path) as source:
        earlier = source.read(1)
        profile = source.profile
    later = earlier.copy()
    changed_mask = np.zeros_like(earlier, dtype=bool)
    changed_mask[:10, :10] = earlier[:10, :10] == 10
    later[changed_mask] = 40
    later_path = settings.classifications_dir / "land_cover_2024.tif"
    with rasterio.open(later_path, "w", **profile) as destination:
        destination.write(later, 1)

    change = detect_changes(
        classification.output_path,
        later_path,
        settings.change_detection_dir / "land_cover_change.tif",
        window_size=16,
    )
    transition_path = save_transition_matrix_csv(
        change.matrix,
        settings.change_detection_dir / "transition_matrix.csv",
    )

    expected_changed = int(changed_mask.sum())
    assert expected_changed > 0
    assert change.matrix.changed_pixels == expected_changed
    assert transition_path.is_file()
    with transition_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0][0:4] == [
        "from_code",
        "from_land_cover_type",
        "to_code",
        "to_land_cover_type",
    ]

    report = build_analysis_report(
        study_area={"aoi_path": str(aoi), "study_area": "Synthetic integration fixture"},
        imagery=data_access.raster_summary(feature_path),
        classification=statistics.to_dict(),
        model_evaluation=evaluation.to_dict(),
        change_detection=change.to_dict(),
    )
    report_path = settings.reports_dir / "analysis_report.json"
    report_path.write_text(analysis_report_json(report), encoding="utf-8")
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))

    assert saved_report["sections"]["classification"]["valid_pixels"] == 1598
    assert saved_report["sections"]["model_evaluation"]["metrics"]["accuracy"] >= 0.8
    assert saved_report["sections"]["change_detection"]["changed_pixels"] == expected_changed
    assert evaluation_path.is_file()

    assert {record.path.name for record in data_access.classification_files(settings)} >= {
        "land_cover_2021.tif",
        "land_cover_2024.tif",
    }
    assert [record.path for record in data_access.change_files(settings)] == [
        change.output_path.resolve()
    ]
    assert evaluation_path.resolve() in {
        record.path for record in data_access.evaluation_reports(settings)
    }
    assert artifacts.metadata_path.resolve() in {
        record.path for record in data_access.model_metadata_files(settings)
    }
    assert aoi.resolve() in {record.path for record in data_access.vector_files(settings)}
    assert feature_path.resolve() in {
        record.path for record in data_access.raster_files(settings)
    }
