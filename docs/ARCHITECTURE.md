# Architecture

## 1. Overview

The system is a layered, modular geospatial application. Each layer depends on
the layer beneath it, and cross-cutting concerns (configuration, logging,
errors, files, validation) live in `config/` and `src/utils/`.

```text
Layer 6  UI / Dashboard          app.py, dashboard/
Layer 5  Visualization           src/visualization/, src/reports/
Layer 4  Analysis                src/analysis/
Layer 3  Machine Learning        src/training/, src/classification/
Layer 2  Geospatial Processing   src/preprocessing/, src/features/
Layer 1  Data Input              src/data/
Cross    Config & Utilities      config/, src/utils/
```

## 2. Layer responsibilities

### Layer 1 — Data input (`src/data/`)

- `raster_loader.py` — open GeoTIFFs, read metadata, validate bands and NoData.
- `vector_loader.py` — read GeoJSON/Shapefile/GeoPackage, validate geometries.
- `dataset_validator.py` — check CRS compatibility, grid alignment, extent.
- `data_metadata.py` — typed metadata records for reproducibility.

### Layer 2 — Geospatial processing (`src/preprocessing/`, `src/features/`)

- `raster_alignment.py` — reproject/resample bands to a common grid.
- `cloud_masking.py` — apply SCL/QA masks when available.
- `raster_clipping.py` — clip to the AOI while preserving georeferencing.
- `image_stacking.py` — build a band stack with a shared transform/CRS.
- `spectral_indices.py` — safe NDVI, NDWI, optional NDBI.
- `feature_extraction.py` — assemble the model feature matrix.

All outputs preserve CRS, transform, width, height, and NoData semantics.

### Layer 3 — Machine learning (`src/training/`, `src/classification/`)

- `sample_generator.py` — align WorldCover labels with features.
- `dataset_split.py` — spatially separated train/validation blocks.
- `random_forest.py` — configured Random Forest classifier.
- `evaluation.py` — overall accuracy, precision, recall, F1, per-class metrics,
  confusion matrix, feature importance.
- `model_manager.py` — save/load model plus reproducibility metadata.
- `predictor.py` — windowed/chunked prediction writing a georeferenced raster.

### Layer 4 — Analysis (`src/analysis/`)

- `area_statistics.py` — pixel counts, projected/geodesic areas, percentages.
- `change_detection.py` — align two classifications and identify transitions.
- `transition_matrix.py` — class-to-class area matrix and summaries.

### Layer 5 — Visualization (`src/visualization/`, `src/reports/`)

- Raster previews, spectral-index maps, classification legends, charts,
  Folium maps, CSV/report generation.

### Layer 6 — UI (`dashboard/`, `app.py`)

- Streamlit landing page in Phase 1; multi-section dashboard in Phase 8.

## 3. Cross-cutting design

### Configuration (`config/settings.py`)

- `GEOAI_*` environment variables, loaded from `.env` without overriding real
  environment variables.
- `load_settings(environ, project_root)` is pure and testable;
  `get_settings()` is cached for application use.
- Relative paths resolve against the repository root, not the current working
  directory.
- `Settings.ensure_directories()` creates all managed directories.
- Invalid values raise `ConfigurationError` with the variable name.

### Land-cover classes (`config/land_cover_classes.py`)

- Single source of truth for the 11 ESA WorldCover 2021 codes, names, and
  official colours.
- Read-only mappings prevent accidental mutation.
- Helpers: `get_class_name`, `get_class_code`, `is_valid_class_code`,
  `get_class_legend`, `get_class_colors`.
- Unknown codes are excluded from training in Phase 4.

### Logging (`src/utils/logging_config.py`)

- Console handler plus optional 5 MB rotating file handler (3 backups).
- Idempotent; Streamlit reruns do not duplicate handlers.
- `normalize_level` validates level names.
- `reset_logging()` exists for test isolation.

### Errors (`src/utils/errors.py`)

- `GeoAIError`
- `ConfigurationError`
- `DataValidationError`
- `RasterError`
- `VectorError`
- `ModelError`

Layers raise these with actionable messages instead of leaking raw exceptions.

### Files and validation (`src/utils/file_utils.py`, `src/utils/validation.py`)

- Directory creation with clear failures.
- Upload suffix/size validation and safe filename generation.
- Human-readable byte sizes for UI messages.

## 4. Data flow (target pipeline)

```text
AOI upload
  -> geometry validation and CRS detection
  -> Sentinel-2 band loading and metadata validation
  -> alignment/resampling to a common grid
  -> cloud/NoData masking
  -> AOI clipping
  -> multispectral stack + spectral indices
  -> WorldCover label alignment
  -> valid pixel sampling
  -> spatial train/validation split
  -> Random Forest training
  -> evaluation reports and feature importance
  -> windowed classification -> georeferenced GeoTIFF
  -> area statistics -> change detection -> dashboard/exports
```

## 5. Key decisions and trade-offs

| Decision | Rationale |
|---|---|
| `src/` as the import package | Follows the requirements document; imports are `from src...` |
| Dataclasses + `python-dotenv` | Lightweight, explicit, testable; no config framework |
| No system GDAL dependency | Rasterio/GeoPandas wheels bundle GDAL and PROJ |
| Custom exception hierarchy | Callers can distinguish config, data, raster, vector, and model failures |
| Read-only class registry | Prevents accidental mutation across dashboard reruns |
| Synthetic test fixtures | Tests run quickly without downloading large datasets |
| No data or results in Git | Large artefacts and fabricated results are explicitly avoided |

## 6. Non-functional requirements mapping

- **Performance:** windowed raster processing in Phase 6; cached settings.
- **Reliability:** validation helpers, typed errors, rotating logs.
- **Code quality:** modular packages, type hints, Ruff, focused tests.
- **Reproducibility:** fixed seed in settings; model metadata in Phase 5;
  environment and dependency declarations in `pyproject.toml`.
- **Security:** `.env` ignored, upload suffix/size checks, no secrets in source.

## 7. Testing strategy

- Unit tests use synthetic data only and run without network access.
- Phase 1 covers settings, class registry, logging, file/validation helpers,
  dependency imports, and Python version.
- Later phases add synthetic GeoTIFF/GeoJSON fixtures for loaders, alignment,
  indices, training, prediction, area statistics, and change detection.
- Git ignores data, models, logs, and generated outputs.
