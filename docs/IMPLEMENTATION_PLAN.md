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

## Phase 4 — Training dataset preparation (next)

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

## Phase 5 — Machine-learning development

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

## Phase 6 — Land-cover classification

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

## Phase 7 — Change detection

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

## Phase 8 — Interactive dashboard

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

## Phase 9 — Testing and documentation

**Tasks**

- Expand unit and integration tests, including invalid inputs.
- Test CRS alignment, classification outputs, and change detection.
- Complete installation, data-acquisition, and usage documentation.
- Add screenshots and example outputs from real data runs.
- Final README polish and GitHub readiness check.

**Acceptance criteria**

- Full test suite passes.
- Lint passes.
- Documentation is complete and accurate.
- No data, models, outputs, or secrets are tracked by Git.

## Verification commands

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest --cov=src --cov=config --cov-report=term-missing
.venv/bin/python -m ruff check .
.venv/bin/python -m compileall -q src config tests app.py
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
