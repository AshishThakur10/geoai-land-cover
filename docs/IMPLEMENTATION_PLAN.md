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

## Phase 2 — GIS fundamentals and data processing

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

## Phase 3 — Spectral feature extraction

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

## Phase 4 — Training dataset preparation

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

## Phase 1 verification record

- Python 3.11.16 (uv-managed)
- 69 packages installed, including Rasterio 1.4.4 (GDAL 3.10.3), GeoPandas
  1.2.0, Scikit-learn 1.9.1, Streamlit 1.66.1
- 59 Phase 1 tests passed
- Ruff: all checks passed
- `compileall`: OK
- Streamlit health endpoint: `ok`
