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

- `data_metadata.py` — frozen `RasterMetadata`/`VectorMetadata` records with
  serialisable dictionaries.
- `raster_loader.py` — context-managed GeoTIFF access, metadata inspection,
  single/all-band reads, validity masks, grid comparison.
- `vector_loader.py` — GeoJSON/Shapefile/GeoPackage loading, geometry
  validation and repair, reprojection, geodesic area, metadata.
- `dataset_validator.py` — AOI/raster CRS alignment and overlap validation.

**Implemented in Phase 2.** Raster paths, suffixes, CRS presence, band ranges,
corrupt files, empty layers, invalid geometries, and non-overlapping AOIs all
produce actionable errors.

### Layer 2 — Geospatial processing (`src/preprocessing/`, `src/features/`)

- `raster_clipping.py` — **implemented Phase 2.** Clips every band to an AOI
  with `rasterio.mask`, preserving CRS/transform/descriptions and honouring
  source NoData. When no NoData is defined, the output carries an internal
  validity mask instead of a misleading fill value.
- `raster_alignment.py` — **implemented Phase 3.** Defines a frozen
  `TargetGrid`, reprojects/resamples one band or many bands onto it, and writes
  georeferenced `float32` GeoTIFFs. Masked/NoData pixels become `NaN`.
- `image_stacking.py` — **implemented Phase 3.** Validates a shared grid and
  stacks aligned bands into a `(bands, height, width)` array with a common
  validity mask; exports multi-band GeoTIFFs.
- `spectral_indices.py` — **implemented Phase 3.** Safe normalized differences
  with zero-denominator and non-finite handling: NDVI, NDWI, and optional NDBI.
- `feature_extraction.py` — **implemented Phase 3.** Canonicalises band
  names/aliases, orders bands, adds indices, tracks per-pixel validity, exposes
  an `(n_valid_pixels, n_features)` sample matrix, and writes feature stacks.
- `cloud_masking.py` — deferred until SCL/QA input is explicitly supported.

All outputs preserve CRS, transform, width, height, and NoData semantics.

### Configuration — Sentinel-2 bands

- `config/sentinel2_bands.py` is the single source of truth for B02/B03/B04/
  B08/B11/B12: names, native resolution, nominal wavelength, and aliases.
- `resolve_band_key` accepts canonical keys, common aliases (`red`, `nir`,
  `swir1`), and case variants.
- `infer_band_key` extracts supported band codes from Sentinel-2 filenames but
  returns `None` rather than guessing.
- Core feature bands are B02/B03/B04/B08; B11/B12 are optional. NDBI is enabled
  only when B11 is available.

### Visualization — implemented Phases 2–3

- `raster_visualizer.py` — `plot_band`, `plot_rgb`, and `save_band_preview`.
- `index_visualizer.py` — fixed-scale index maps, valid-pixel distributions,
  and PNG previews.
- Figure creation attaches an Agg canvas explicitly, so previews work without a
  display and inside Streamlit.
- No fabricated imagery is generated; helpers only render caller-supplied data.

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

- Phase 2: raster band/RGB previews and PNG export.
- Later: spectral-index maps, classification legends, charts, Folium maps,
  CSV/report generation.

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

### Phase 2 geospatial rules

- Raster reads always go through `open_raster`, which validates the path and
  guarantees the dataset is closed.
- `RasterMetadata.is_georeferenced` requires both a CRS and a non-identity
  transform.
- AOIs are always reprojected to the raster CRS before clipping; missing CRS
  values are rejected rather than guessed.
- Invalid geometries are repaired with `shapely.make_valid`; unrepaired
  geometries are dropped with a warning, and an empty result is an error.
- Clipping uses a masked array internally. Source NoData is preserved; if the
  source has no NoData, the output stores an internal mask.
- Area helpers use geodesic calculations on WGS84, never degrees-as-metres.
- `valid_pixel_count` counts spatial pixels valid across all bands.

### Phase 3 spectral-feature rules

- Georeferencing always comes from the target grid; source CRS/transform are
  only used as reprojection inputs.
- Alignment output is `float32` with `NaN` NoData so masked pixels cannot
  collide with valid integer values. Valid pixels are always finite.
- Resampling names are validated against Rasterio's supported methods. Default
  is `bilinear` for reflectance and indices; callers should use `nearest` for
  categorical labels.
- `TargetGrid.from_bounds(..., resolution)` snaps outward to a resolution
  multiple. Resolution units follow the CRS, so geographic grids require an
  explicit, meaningful degree value.
- Normalized differences compute in `float32` under suppressed NumPy error
  state. Zero denominators, non-finite inputs, and masked inputs produce masked
  outputs, never infinities.
- NDVI and NDWI are always computed from core bands; NDBI is included only when
  B11 is available unless explicitly requested.
- The feature stack's validity mask is the intersection of finite values across
  every band and index. `as_sample_matrix()` returns only those pixels, which
  prevents scikit-learn from receiving `NaN` or masked rows.
- B02/B03/B04/B08 are required; B11/B12 are optional. Duplicate aliases for the
  same band are rejected rather than silently overwritten.

## 5. Key decisions and trade-offs

| Decision | Rationale |
|---|---|
| `src/` as the import package | Follows the requirements document; imports are `from src...` |
| Dataclasses + `python-dotenv` | Lightweight, explicit, testable; no config framework |
| No system GDAL dependency | Rasterio/GeoPandas wheels bundle GDAL and PROJ |
| Custom exception hierarchy | Callers can distinguish config, data, raster, vector, and model failures |
| Read-only class registry | Prevents accidental mutation across dashboard reruns |
| Reference-band-derived target grid | One explicit alignment choice instead of silently mixing resolutions |
| `float32` + `NaN` for features | Preserves masked semantics and matches scikit-learn expectations |
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
- Phase 2 covers synthetic GeoTIFF and GeoJSON/Shapefile fixtures for metadata,
  band reads, NoData handling, geometry repair, reprojection, CRS validation,
  clipping, georeferenced output, and raster previews.
- Phase 3 covers band-key resolution and filename inference, target-grid
  construction, same-grid and cross-resolution alignment, geographic-to-
  projected reprojection, masked-NoData propagation, multispectral stacks,
  zero-denominator index masking, NDBI availability rules, feature ordering,
  sample-matrix validity, statistics, and index visualisations.
- Later phases extend the same fixture approach to training, prediction, area
  statistics, and change detection.
- Git ignores data, models, logs, and generated outputs.
