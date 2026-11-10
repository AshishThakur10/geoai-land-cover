# Phased Implementation Plan

Development proceeds phase by phase. A phase is complete only when its code,
tests, and documentation pass. Do not start the next phase while a critical
error remains unresolved.

## Phase 1 — Project setup (complete)

**Deliverables**

- Directory structure and importable packages.
- Python 3.11 environment with declared dependencies.
- Environment-driven settings and ESA WorldCover class registry.
- Logging, error hierarchy, file and validation utilities.
- Minimal Streamlit landing page.
- README, architecture, data-source, user, and plan documentation.
- Unit tests for all Phase 1 modules.

**Acceptance criteria**

- `uv pip install -e ".[dev]"` succeeds.
- All mandatory geospatial and ML libraries import.
- `pytest -q` passes.
- `ruff check .` passes.
- `compileall` succeeds.
- Streamlit starts headlessly and the health endpoint returns `ok`.
- No data, models, or secrets are committed.

## Phase 2 — GIS fundamentals and data processing (complete)

**Tasks**

- Load GeoTIFFs and read metadata, CRS, transform, bounds, NoData, dtypes.
- Load GeoJSON/Shapefile and validate/repair geometries.
- Detect and reproject vector CRS.
- Clip rasters to the AOI while preserving georeferencing.
- Validate NoData handling.
- Preview raster bands.

**Acceptance criteria**

- Synthetic raster and vector fixtures load successfully.
- Invalid raster, invalid geometry, and incompatible CRS produce meaningful
  `GeoAIError` subclasses.
- Clipped output retains the correct CRS, transform, and dimensions.
- Tests cover valid and invalid inputs.

**Delivered modules**

- `src/data/data_metadata.py`
- `src/data/raster_loader.py`
- `src/data/vector_loader.py`
- `src/data/dataset_validator.py`
- `src/preprocessing/raster_clipping.py`
- `src/visualization/raster_visualizer.py`
- `tests/fixtures.py` plus five Phase 2 test modules

**Verified behaviour**

- 40×40 synthetic GeoTIFF clipped to a 100 m × 100 m AOI yields a 10×10
  output with the expected transform and EPSG:32633 CRS.
- Source NoData is preserved; sources without NoData produce an internal mask.
- Invalid bowtie geometries are repaired with `shapely.make_valid`.
- WGS84 AOIs are reprojected to the raster CRS; non-overlapping AOIs are
  rejected.
- Geodesic area of a synthetic 1 km × 1 km polygon is approximately 1 km².

## Phase 3 — Spectral feature extraction (complete)

**Tasks**

- Load Sentinel-2 B02/B03/B04/B08; optionally B11/B12.
- Resample bands to a consistent grid and document the target resolution.
- Build a multispectral stack with preserved georeferencing.
- Calculate NDVI, NDWI, and optional NDBI with zero-denominator handling.
- Visualize indices and distributions.

**Acceptance criteria**

- Aligned bands share CRS, transform, width, and height.
- NDVI and NDWI lie in `[-1, 1]` for valid pixels.
- NDBI is enabled only when SWIR is available.
- Division by zero and NoData produce masked values, not infinities.

**Delivered modules**

- `config/sentinel2_bands.py`
- `src/preprocessing/raster_alignment.py`
- `src/preprocessing/image_stacking.py`
- `src/features/spectral_indices.py`
- `src/features/feature_extraction.py`
- `src/visualization/index_visualizer.py`
- `tests/test_sentinel2_bands.py` plus five Phase 3 test modules

**Verified behaviour**

- Band keys and aliases resolve consistently; unsupported bands and filename
  codes are rejected or ignored rather than guessed.
- Same-grid alignment preserves values; cross-resolution and geographic-to-
  projected alignment produce the target CRS, transform, dimensions, and
  `float32` output.
- Source NoData becomes `NaN` and propagates into the shared feature mask.
- Multispectral stacks reject mismatched grids and preserve band order and
  descriptions on export.
- NDVI/NDWI always compute from core bands; NDBI appears only when B11 exists.
- Zero denominators and non-finite inputs are masked; no infinities are
  produced.
- The feature stack exposes only finite valid pixels to modelling and rejects
  missing core bands or duplicate aliases.

## Phase 4 — Training dataset preparation (complete)

**Tasks**

- Load and map ESA WorldCover 2021 codes.
- Align labels with the feature grid.
- Exclude invalid/NoData pixels and unknown classes.
- Handle class imbalance.
- Create spatially separated training/validation sets.

**Acceptance criteria**

- Every sample has a valid class code and complete feature vector.
- No pixel appears in both training and validation.
- Class distribution and split metadata are recorded.
- Tests verify alignment, masking, and split disjointness.

**Delivered modules**

- `src/training/sample_generator.py`
- `src/training/dataset_split.py`
- `tests/test_sample_generator.py`
- `tests/test_dataset_split.py`
- Synthetic WorldCover fixtures in `tests/fixtures.py`

**Verified behaviour**

- WorldCover labels align to the exact feature grid with nearest resampling;
  coarse 20 m labels resample correctly to a 10 m feature grid.
- Label NoData, non-finite aligned values, unknown codes, and pixels with any
  invalid feature are excluded and counted.
- Retained samples always have a valid WorldCover code and a fully finite
  feature vector.
- Class counts, percentages, missing classes, and imbalance ratio are derived
  from retained samples; inverse-frequency weights equalise total class weight.
- Whole spatial blocks are assigned to one side, so train and validation have
  no overlapping pixels.
- Split configuration, block IDs, class distributions, and disjointness are
  serialised to a reproducible JSON manifest.

## Phase 5 — Machine-learning development (complete)

**Tasks**

- Train a Random Forest with configurable parameters.
- Evaluate overall accuracy, precision, recall, F1, confusion matrix, and
  per-class support.
- Generate feature importance.
- Save the model and reproducibility metadata (inputs, dates, AOI, features,
  parameters, seed, metrics, software versions).

**Acceptance criteria**

- Model reloads and reproduces predictions.
- Metrics are computed from the held-out spatial split.
- No fabricated or hard-coded metrics.
- Tests use synthetic data and small forests.

**Delivered modules**

- `src/training/random_forest.py`
- `src/training/evaluation.py`
- `src/training/model_manager.py`
- `tests/test_random_forest.py`
- `tests/test_evaluation.py`
- `tests/test_model_manager.py`
- `synthetic_ml_samples()` fixture in `tests/fixtures.py`

**Verified behaviour**

- Random Forest configuration rejects invalid tree counts, depths, leaf sizes,
  feature strategies, class weights, seeds, and worker counts.
- The same seed and synthetic samples reproduce the same predictions.
- Training rejects fewer than two samples or fewer than two classes and records
  feature names/order, class codes, counts, parameters, and label source.
- Known synthetic predictions produce the expected accuracy, balanced accuracy,
  macro/weighted metrics, confusion matrix, per-class support, and feature
  importance; absent classes are reported with zero support.
- Held-out validation is evaluated separately from training data using small
  synthetic forests.
- Saving and reloading reproduces predictions and preserves feature order,
  classes, parameters, seed, run metadata, metrics, checksum, and versions.
- Corrupt files, unsupported paths, metadata-format mismatches, non-Random-
  Forest estimators, and feature/class contract violations are rejected.
- Evaluation reports and model metadata serialise to valid JSON.

## Phase 6 — Land-cover classification (complete)

**Tasks**

- Load a trained model.
- Predict in windows/chunks.
- Write a georeferenced GeoTIFF with class codes and defined NoData.
- Compute pixel counts, areas (projected CRS or geodesic), and percentages.
- Export statistics CSV.

**Acceptance criteria**

- Output CRS, transform, width, and height match the input feature grid.
- Only valid class codes or NoData appear in the raster.
- Areas use metres, not degrees.
- Tests verify transform/CRS preservation and deterministic output.

**Delivered modules**

- `src/classification/predictor.py`
- `src/analysis/area_statistics.py`
- `tests/test_classification.py`
- `tests/test_area_statistics.py`

**Verified behaviour**

- In-memory and raster-path classification use bounded windows, including
  partial edge windows and window sizes smaller than the raster.
- Saved models and trained models enforce feature names/order; NoData never
  overlaps a class code; invalid feature pixels become NoData.
- Output GeoTIFFs are `uint8`, preserve CRS/transform/width/height, contain only
  model classes or NoData, and carry a class legend tag, palette, description,
  and class counts.
- Classification is deterministic across window sizes and save/load cycles.
- Projected area statistics use transform determinant and CRS linear-unit
  conversion; geographic statistics use WGS84 geodesic cell areas.
- Percentages exclude NoData, unknown class codes fail explicitly, and CSV
  values match the in-memory statistics.
- Missing CRS, rotated geographic grids, non-integer bands, invalid band/window
  arguments, and unsupported output suffixes fail with actionable errors.

## Phase 7 — Change detection (complete)

**Tasks**

- Load two classified rasters.
- Validate alignment; reproject/align when necessary.
- Compare pixels and generate a change raster.
- Build a class-to-class transition matrix with areas.
- Summaries of changed area and percentage.

**Acceptance criteria**

- Known synthetic transitions produce the expected matrix.
- Misaligned inputs are either aligned or rejected with a clear error.
- Reports include the scientific caveat that differences are not proof of
  real-world change.

**Delivered modules**

- `src/analysis/change_detection.py`
- `src/analysis/transition_matrix.py`
- `src/analysis/area_utils.py` (shared area/NoData logic extracted from Phase 6)
- `tests/test_change_detection.py`

**Verified behaviour**

- Known synthetic transitions produce exact pixel counts and transition areas,
  with unchanged cells on the diagonal and changed cells off-diagonal.
- `alignment="strict"` rejects CRS, transform, and dimension mismatches with an
  actionable reproject hint; `alignment="reproject"` warps the later raster onto
  the earlier grid with nearest-neighbour resampling.
- Change rasters use `uint8` values `0` unchanged, `1` changed, and `255`
  NoData, preserve the earlier grid, and include description, legend, palette,
  alignment, and caveat metadata.
- Only both-valid pixels are analysed; changed pixels and changed area are
  counted against both-valid and changed denominators respectively.
- Transition matrices contain the sorted union of observed classes, projected
  or WGS84 geodesic areas, and complete CSV export.
- Window sizes do not change pixel counts, areas, or output values.
- Invalid alignment modes, band indices, dtypes, class codes, NoData overrides,
  output suffixes, missing CRS, and rasters with no overlap fail clearly.
- All result and matrix dictionaries include the scientific change caveat.

## Phase 8 — Interactive dashboard (complete)

**Tasks**

- Multi-section Streamlit dashboard: home, study area, imagery, spectral
  analysis, classification, model evaluation, change detection, exports.
- Folium maps and Plotly/Matplotlib charts.
- File uploads with suffix/size validation.
- Download buttons for GeoTIFF, CSV, confusion matrix, classification report,
  and analysis summary.

**Acceptance criteria**

- Dashboard starts without errors.
- Sections use real outputs from the pipeline.
- Invalid uploads show actionable messages.
- Exported files preserve georeferencing and match on-screen statistics.

**Delivered modules**

- `app.py`
- `dashboard/common.py`
- `dashboard/data_access.py`
- `dashboard/sections/` (all eight sections)
- `src/visualization/map_visualizer.py`
- `src/visualization/charts.py`
- `src/reports/report_generator.py`
- `tests/test_dashboard_data_access.py`
- `tests/test_map_visualizer.py`
- `tests/test_charts_reports.py`
- `tests/test_dashboard_app.py`

**Verified behaviour**

- All eight sections load without errors with both empty and populated
  synthetic project directories.
- Dashboard file discovery covers imagery, classifications, change outputs,
  study areas, model metadata, and evaluation reports; newer files sort first.
- Upload suffix, empty-payload, and configured-size limits are enforced before
  storage; stored names are generated rather than caller-controlled.
- Zipped vector uploads reject absolute paths, parent traversal, symbolic links,
  expansion beyond the upload limit, and archives without exactly one supported
  vector layer.
- Study-area activation exposes feature count, approximate WGS84 geodesic area,
  CRS, and geometry types; clearing it removes the active context.
- Raster views report dimensions, band count, pixel size, CRS, band
  descriptions, and acquisition dates only when present in raster tags.
- Spectral indices are computed through the Phase 3 pipeline and are not
  displayed against a changed source or band mapping until recomputed.
- Classification maps use the class legend and palette; area tables and charts
  use the same `AreaStatistics` object as CSV exports.
- Evaluation views validate report structure and render metrics, confusion
  matrix, per-class metrics, and feature importance from the JSON report.
- Change detection uses the Phase 7 API, including strict/reproject alignment,
  and displays changed area, changed share, transition heatmap, and full matrix.
- Exports provide WGS84 GeoJSON for the active AOI, source GeoTIFFs, CSV tables,
  and JSON/Markdown analysis reports. Reports include only available real
  sections and carry scientific caveats.
- AppTest covers navigation, empty-state handling, study-area activation,
  imagery, spectral computation, evaluation, change detection, and export
  controls.
- Invalid uploads and pipeline/display failures produce actionable UI errors
  instead of tracebacks.

## Phase 9 — Final testing, documentation, and release readiness (in progress)

**Tasks**

- Expand unit and integration tests, including invalid inputs.
- Test CRS alignment, classification outputs, and change detection.
- Complete installation, data-acquisition, and usage documentation.
- Add screenshots and example outputs from real data runs.
- Final README polish and GitHub readiness check.

**Progress**

- Added `tests/test_end_to_end_pipeline.py`, which runs a synthetic project
  through feature extraction, label sampling, spatial splitting, training,
  evaluation, model persistence, classification, area statistics, change
  detection, transition export, report generation, and dashboard discovery.
- Added invalid-archive, symbolic-link, empty-archive, malformed evaluation
  report, malformed model metadata, and malformed CSV-serialization tests.
- Study Area now displays the source-CRS bounding box alongside feature count,
  area, CRS, and geometry types.
- Added `.github/workflows/ci.yml` to run Ruff and the test suite with coverage.
- Updated stale AOI/data-acquisition wording and the bounding-box user guidance.

**Remaining**

- Add screenshots and example outputs from a real data run when openly licensed
  data is available; do not fabricate these artefacts.
- The repository remains intentionally unlicensed. Add an explicit licence
  before redistribution or external contribution; do not describe it as
  open-source until then.

**Acceptance criteria**

- Full test suite passes.
- Lint passes.
- Documentation is complete and accurate.
- No data, models, outputs, or secrets are tracked by Git.

## Verification commands

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest --cov=src --cov=config --cov=dashboard --cov-report=term-missing
.venv/bin/python -m ruff check .
.venv/bin/python -m compileall -q src config dashboard tests app.py
.venv/bin/python -m streamlit run app.py --server.headless true
```

## Verification record

### Phase 1

- Python 3.11.16 (uv-managed)
- 69 packages installed, including Rasterio 1.4.4 (GDAL 3.10.3), GeoPandas
  1.2.0, Scikit-learn 1.9.1, Streamlit 1.66.1
- 59 Phase 1 tests passed
- Ruff: all checks passed
- `compileall`: OK
- Streamlit health endpoint: `ok`

### Phase 2

- 111 total tests passed (52 new Phase 2 tests)
- Synthetic GeoTIFF fixtures cover multi-band data, NoData, missing CRS, and
  corrupt files
- Synthetic GeoJSON/Shapefile fixtures cover CRS reprojection and geometry
  repair
- Raster clipping verified for CRS, transform, dimensions, band descriptions,
  NoData, internal masks, and georeferenced GeoTIFF output
- Raster previews verified for band, RGB, PNG export, and invalid bands
- Ruff: all checks passed
- `compileall`: OK

### Phase 3

- 185 total tests passed (74 new Phase 3 tests)
- Overall coverage: 94% statement/branch coverage
- Synthetic fixtures cover same-grid alignment, 20 m → 10 m resampling,
  geographic → projected reprojection, NoData propagation, mismatched grids,
  zero-denominator indices, NDBI availability, feature ordering, sample-matrix
  validity, and index visualisation
- Verified NDVI/NDWI stay in `[-1, 1]` for valid pixels and masked values never
  become infinities
- Verified exported aligned rasters and feature stacks preserve CRS, transform,
  dimensions, `float32` dtype, band descriptions, and `NaN` NoData
- Ruff: all checks passed
- `compileall`: OK

### Phase 4

- 201 total tests passed (16 new Phase 4 tests)
- Overall coverage: 94% statement/branch coverage
- Synthetic WorldCover fixtures cover four classes, same-grid and 20 m → 10 m
  label alignment, explicit NoData/unknown codes, and feature/label mask
  intersections
- Verified retained samples contain only valid ESA WorldCover codes and finite
  feature vectors
- Verified class distributions, percentages, missing classes, imbalance ratio,
  and balanced sample weights
- Verified block-based train/validation splits are pixel-disjoint, deterministic
  under the same seed, and cover every sample exactly once
- Verified JSON manifests record split configuration, block IDs, class
  distributions, and the disjointness check
- Ruff: all checks passed
- `compileall`: OK

### Phase 5

- 238 total tests passed (37 new Phase 5 tests)
- Overall coverage: 93% statement/branch coverage
- Synthetic ML fixture covers four land-cover classes and six features across
  200 samples
- Verified seeded training reproducibility, balanced-weight mode, and explicit
  failures for insufficient samples/classes and invalid prediction inputs
- Verified accuracy, balanced accuracy, macro/weighted precision/recall/F1,
  per-class support, confusion matrix, absent classes, and feature importance
  against known synthetic predictions
- Verified held-out evaluation uses validation samples, not training samples
- Verified `.joblib` save/load prediction identity and JSON metadata containing
  feature order, classes, parameters, seed, training counts, run provenance,
  metrics, model checksum, and software versions
- Verified rejection of unsupported suffixes, missing/corrupt files, metadata
  format mismatches, non-Random-Forest estimators, and feature/class mismatches
- Ruff: all checks passed
- `compileall`: OK

### Phase 6

- 255 total tests passed (17 new Phase 6 tests)
- Overall coverage: 93% statement/branch coverage
- Synthetic feature stacks cover four classes, invalid pixels, partial windows,
  and both in-memory and saved-feature-raster classification
- Verified class maps match expected synthetic classes, remain identical across
  window sizes, and preserve CRS, transform, width, height, `uint8` dtype, and
  NoData `0`
- Verified loaded-model classification reproduces in-memory predictions and that
  feature order, NoData, and class-code contracts are enforced
- Verified GeoTIFF legend tags, palette, and per-class pixel counts
- Verified projected areas from 10 m pixels, geographic geodesic cell areas,
  percentages excluding NoData, unknown-code rejection, and CSV serialisation
- Ruff: all checks passed
- `compileall`: OK

### Phase 7

- 264 total tests passed (9 new Phase 7 tests)
- Overall coverage: 93% statement/branch coverage
- Change detection: 97% coverage; transition matrix: 96% coverage
- Known synthetic transitions produce exact pixel counts, matrix cells, changed
  area, and changed percentage
- Verified change-raster CRS, transform, dimensions, `uint8` values, NoData,
  description, legend, palette, and scientific-caveat tags
- Verified window-size-independent counts, areas, and output values
- Verified strict-grid rejection, nearest-neighbour reprojection onto the
  earlier grid, and source-NoData propagation through warping
- Verified WGS84 geodesic transition areas for geographic rasters
- Verified complete transition CSV export and explicit caveat in result/matrix
  dictionaries
- Verified rejection of unsupported alignment/output/band/dtype/class/NoData
  inputs and classifications with no both-valid pixels
- Ruff: all checks passed
- `compileall`: OK

### Phase 8

- 290 total tests passed (26 new Phase 8 tests)
- Overall coverage across `src`, `config`, and `dashboard`: 89%
- Dashboard coverage: `data_access.py` 87%, `common.py` 83%, section modules
  65–83%, map visualiser 87%, charts 89%, report generator 88%
- AppTest verified navigation, empty and populated states, study-area
  activation, imagery metadata, spectral computation, evaluation, change
  detection, and export controls without exceptions
- Verified upload size/suffix checks, generated storage names, zipped-vector
  traversal/symlink/expansion/single-layer rejection, newest-first discovery,
  raster tag/acquisition handling, and JSON report/model metadata reads
- Verified Folium classification/continuous/change/boundary maps, Plotly chart
  construction, CSV report serialisation, and JSON/Markdown analysis reports
- Study-area exports serialise the active AOI as complete WGS84 GeoJSON; source
  rasters are downloaded unchanged, and area/transition CSVs serialise the
  displayed statistics objects
- Confirmed spectral results are hidden after a source or band-mapping change
  until recomputed
- Streamlit headless health endpoint: `ok`
- Ruff: all checks passed
- `compileall`: OK

### Phase 9 (in progress)

- 294 total tests passed (4 new Phase 9 tests)
- Overall coverage across `src`, `config`, and `dashboard`: 90%
- `dashboard/data_access.py` 91%, report generator 89%, map visualiser 87%
- End-to-end synthetic test covers feature extraction, WorldCover sampling,
  spatially blocked splitting, Random Forest training/evaluation, model
  persistence/reload, windowed classification, area statistics, change
  detection, transition CSV export, analysis-report JSON, and dashboard output
  discovery
- Added rejection tests for invalid ZIP data, empty archives, symbolic links,
  malformed evaluation reports, malformed model metadata, malformed per-class
  metrics, and non-integer confusion-matrix values
- Study Area displays the source-CRS bounding box; AppTest verifies it appears
  alongside the existing AOI metadata
- CI workflow YAML parsed successfully; it runs Ruff and pytest with coverage on
  push/PR
- Streamlit headless health endpoint: `ok`
- Ruff: all checks passed
- `compileall`: OK
- No data, model, output, log, or obvious secret files are tracked; only
  `.gitkeep` placeholders remain under data/log paths
