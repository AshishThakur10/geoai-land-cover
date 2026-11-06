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

- `sample_generator.py` — **implemented Phase 4.** Aligns WorldCover labels to
  the exact feature grid with nearest resampling, excludes NoData/unknown
  classes and invalid features, retains spatial coordinates, and records class
  distributions, ignored-pixel counts, missing classes, and inverse-frequency
  sample weights.
- `dataset_split.py` — **implemented Phase 4.** Assigns whole spatial blocks to
  train or validation with a seeded shuffle and writes a JSON manifest with
  block IDs, configuration, class distributions, and disjointness checks.
- `random_forest.py` — **implemented Phase 5.** Validates configuration,
  builds a seeded scikit-learn Random Forest, trains on a `LabeledSamples`
  object, rejects fewer than two samples/classes, and exposes predictions plus
  training provenance.
- `evaluation.py` — **implemented Phase 5.** Held-out accuracy, balanced
  accuracy, macro/weighted precision/recall/F1, per-class support and metrics,
  confusion matrix, feature importance, and JSON report persistence.
- `model_manager.py` — **implemented Phase 5.** Saves the fitted model with a
  `.joblib` file and JSON metadata sidecar; loads both and validates the format,
  estimator type, feature count/order, and class contract.
- `predictor.py` — **implemented Phase 6.** Classifies an in-memory
  `FeatureStack` or reads a feature GeoTIFF in windows, writes a `uint8`
  georeferenced class raster with defined NoData, class legend tags, palette,
  and per-class pixel counts.

### Layer 4 — Analysis (`src/analysis/`)

- `area_utils.py` — shared NoData validation, pixel-validity masks, and
  projected/geodesic area calculations.
- `area_statistics.py` — **implemented Phase 6.** Windowed per-class pixel
  counts, percentages, projected or WGS84 geodesic areas, and CSV export.
- `change_detection.py` — **implemented Phase 7.** Validates or reprojects two
  classifications onto the earlier grid, writes a windowed change raster, and
  accumulates changed-area and transition statistics.
- `transition_matrix.py` — **implemented Phase 7.** Class-to-class pixel/area
  matrix, changed-transition summaries, scientific caveat, and CSV export.

### Layer 5 — Visualization (`src/visualization/`, `src/reports/`)

- Phases 2–3: raster band/RGB previews, spectral-index maps, and PNG export.
- Phase 8: classification legends, charts, Folium maps, and report generation.

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
  -> model + reproducibility metadata persistence
  -> model reload and windowed classification -> georeferenced GeoTIFF
  -> per-class pixel counts/areas -> statistics CSV
  -> aligned change raster + transition matrix + changed-area summary
  -> dashboard/exports
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

### Phase 4 label and split rules

- Label rasters are aligned to the feature stack's CRS, transform, width, and
  height. Only nearest-neighbour resampling is exposed for categorical labels.
- Label `0`, non-finite aligned values, and codes outside the 11 ESA
  WorldCover classes are excluded from samples. Unknown-code and NoData counts
  are retained for auditability rather than silently discarded.
- A sample is valid only when every feature is finite and the label is valid;
  `X` never contains `NaN` and `y` never contains an unknown code.
- Class counts, percentages, missing classes, and the max/min imbalance ratio
  are derived from the retained samples. `balanced_sample_weights()` assigns
  equal total weight per present class.
- Splitting is done on spatial blocks, not individual pixels: every sample in a
  block goes to exactly one side. This removes pixel overlap and reduces
  fine-grained spatial leakage, but adjacent blocks can still be nearby.
- The block size, validation fraction, seed, train/validation block IDs,
  per-split class distributions, and `pixel_disjoint` flag are recorded in a
  JSON manifest for reproducibility.
- WorldCover remains a reference label source with label noise and temporal
  mismatch risks; split manifests do not turn it into ground truth.

### Phase 5 model rules

- Models are fitted only on `split.train`; `split.validation` is reserved for
  held-out evaluation. Training accuracy is never reported as validation
  accuracy.
- Random Forest configuration comes from `RandomForestConfig.from_settings()`
  or explicit values. The seed is recorded with the model; tests verify that the
  same seed and samples reproduce the same predictions.
- `use_balanced_weights` is explicit rather than implicit. When enabled, model
  metadata records the choice so class weighting is reproducible.
- Evaluation derives its label set from the union of observed and predicted
  valid WorldCover codes. An explicit label list may include absent classes,
  whose support is reported as zero. Division by zero yields zero metrics rather
  than `NaN` or an exception.
- Metrics are calculated from actual prediction arrays. No fallback, synthetic,
  or hard-coded accuracy values are used.
- Feature importances are emitted only when their count matches the feature
  names and all values are finite.
- A saved model consists of a `.joblib` model plus a `.metadata.json` sidecar.
  The sidecar records format version, creation timestamp, model checksum, feature
  order, classes, parameters, seed, training counts and label source, optional
  input paths/dates/AOI, evaluation metrics, and software versions.
- `load_model()` treats the metadata as a contract: an unsupported format,
  non-Random-Forest estimator, feature mismatch, or class mismatch is rejected.
  Joblib loading is restricted to trusted files because deserialisation can
  execute arbitrary code.

### Phase 6 classification and area rules

- Classification accepts either a fitted `TrainedRandomForest` or a persisted
  `LoadedModel`; both must use the exact feature names and order seen in
  training. Raster band descriptions provide the feature order, or the caller
  must supply `feature_names` explicitly.
- Feature rasters are read and classified in square windows. The in-memory
  `FeatureStack` path applies the same windowing logic. Memory use is bounded by
  the window size and feature count rather than total raster size.
- A pixel is classified only when every feature is finite after masked/NoData
  handling. Invalid pixels become the configured NoData code. NoData must be an
  integer in `[0, 255]` and must not overlap a recognised class code.
- Output is a single-band `uint8` GeoTIFF with the source CRS, transform, width,
  and height unchanged. Only model classes and NoData may occur. The dataset
  carries an ESA WorldCover legend tag, a palette, and a class-code description.
- Predictions returned by the estimator are checked against the model's declared
  classes before writing; unexpected codes fail rather than enter the map.
- Area statistics require an integer classification band. Unknown class codes
  are reported as an error, not silently dropped.
- For projected CRSs, pixel area comes from the absolute transform determinant
  multiplied by the CRS linear-unit conversion factor. For geographic CRSs,
  WGS84 geodesic cell areas are computed row by row; raw degrees are never
  treated as metres.
- Rotated geographic grids are rejected because the row-wise geodesic
  simplification would be inaccurate. Reproject them to a projected CRS first.
- Percentages use valid classified pixels as the denominator; NoData is excluded.
  CSV output uses the same counts/areas as the in-memory result.

### Phase 7 change-detection rules

- Both inputs must be single-band integer classification rasters with a CRS and
  recognised class codes. NoData values cannot overlap valid class codes.
- `alignment="strict"` requires identical CRS, transform, width, and height.
  `alignment="reproject"` warps the later classification onto the earlier grid
  with nearest-neighbour resampling; categorical classes are never interpolated.
- Reprojection uses a lazy `WarpedVRT`, so later-raster windows are read and
  warped on demand. Strict comparisons read matching windows directly.
- Only pixels valid in both classifications participate. A pixel is changed when
  the earlier and later class codes differ. Invalid pixels become change NoData.
- Change rasters are `uint8`: `0` unchanged, `1` changed, `255` NoData. They
  preserve the earlier CRS, transform, width, and height and carry a change
  description, legend tag, palette, alignment mode, and scientific caveat.
- The transition matrix includes the sorted union of classes observed in either
  valid classification. Rows are earlier classes, columns are later classes;
  diagonal cells are unchanged pixels and off-diagonal cells are transitions.
- Transition areas apply the same projected or WGS84 geodesic calculations as
  Phase 6. Percentages use both-valid pixels as the denominator; changed
  percentages use changed pixels.
- All generated summaries include the explicit caveat that map differences are
  not proof of real-world land-cover change.

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
| Nearest-only label resampling | Prevents invalid class codes from interpolation |
| Block-based spatial split | Prevents pixel leakage without expensive geometry buffering |
| Balanced sample weights + class distribution metadata | Makes imbalance explicit and reproducible |
| Train/evaluate on separate spatial splits | Avoids presenting training performance as validation evidence |
| Joblib model + JSON metadata sidecar | Keeps the binary artifact small while making provenance human-readable |
| Feature/class contract checks on load | Prevents silently predicting with mismatched inputs or labels |
| Windowed classification reads/writes | Bounds memory use for large imagery |
| `uint8` class raster with NoData `0` | Compact, directly usable in GIS with an explicit invalid value |
| Geodesic fallback for geographic rasters | Reports meaningful areas without treating degrees as metres |
| Strict alignment by default, explicit nearest reprojection | Prevents silently warping categorical maps while still supporting mismatched grids |
| Change raster values `0`/`1`/`255` | Unambiguous unchanged, changed, and invalid states |
| Full class union for transition matrices | Represents classes that appear or disappear between dates |
| Synthetic test fixtures | Tests run quickly without downloading large datasets |
| No data or results in Git | Large artefacts and fabricated results are explicitly avoided |

## 6. Non-functional requirements mapping

- **Performance:** windowed feature reading, classification, statistics, and
  aligned change detection; cached settings.
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
- Phase 4 covers WorldCover label alignment, NoData/unknown-class exclusion,
  class distributions and imbalance metadata, balanced sample weights, block-
  based disjoint splits, reproducibility, and JSON manifest serialisation.
- Phase 5 covers configuration validation, seeded Random Forest training,
  deterministic prediction, insufficient sample/class failures, known-value
  metric calculations, absent classes, confusion matrices, feature importance,
  held-out evaluation, save/load prediction identity, metadata contents, and
  corrupted or mismatched artifact rejection.
- Phase 6 covers in-memory and raster-window classification, NoData propagation,
  feature/class contract failures, deterministic class maps, grid/CRS/transform
  preservation, legend tags and palette, projected pixel-area calculations,
  geographic geodesic areas, unknown-code rejection, CSV export, and invalid
  area-input failures.
- Phase 7 covers known transition matrices and areas, window-size independence,
  strict-grid rejection, nearest reprojection onto the earlier grid, geographic
  geodesic transition areas, change-raster values and metadata, CSV export,
  invalid inputs, NoData exclusion, and the scientific caveat.
- Git ignores data, models, logs, and generated outputs.
