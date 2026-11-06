# User Guide

## Current status

Phase 7 (change detection) is complete. The project can load
GeoTIFFs and GeoJSON/Shapefile boundaries, inspect metadata and CRS values,
repair invalid geometries, reproject AOIs, clip rasters with preserved
georeferencing and NoData, align and resample Sentinel-2 bands to a shared grid,
build a multispectral stack, compute NDVI/NDWI/optional NDBI, export a
model-ready feature stack, align ESA WorldCover labels, exclude and audit
invalid pixels, report class distributions and balanced weights, create
spatially separated train/validation blocks with JSON manifests, train a
configurable Random Forest, evaluate it on held-out samples, inspect feature
importance, save/load the model with reproducibility metadata, classify feature
rasters in windows, write georeferenced land-cover GeoTIFFs with legends,
calculate projected or geodesic class areas with CSV export, compare two
classifications, write a georeferenced change raster, build class-to-class
transition matrices with areas, and export transition tables. The full dashboard
is not implemented yet.

## 1. Requirements

- Python 3.11 (the project is pinned to 3.11.x)
- macOS, Linux, or Windows with a wheel-supported architecture
- `uv` (recommended) or `python3.11 -m venv`
- Git (optional, for version control)

No separate GDAL installation is normally needed: Rasterio and GeoPandas
binary wheels bundle GDAL and PROJ.

## 2. Installation

From the repository root:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Or with standard tooling:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

Shortcuts:

```bash
make install
```

## 3. Configuration

```bash
cp .env.example .env
```

Important variables:

| Variable | Default | Purpose |
|---|---|---|
| `GEOAI_LOG_LEVEL` | `INFO` | Console/file log level |
| `GEOAI_LOG_FILE` | `logs/geoai.log` | Rotating log file path |
| `GEOAI_DATA_DIR` | `data` | Root data directory |
| `GEOAI_DEFAULT_AOI` | empty | Optional default AOI path |
| `GEOAI_RANDOM_SEED` | `42` | Reproducibility seed |
| `GEOAI_N_ESTIMATORS` | `150` | Random Forest trees |
| `GEOAI_MAX_DEPTH` | empty | Optional Random Forest depth limit |
| `GEOAI_CLASS_WEIGHT` | `balanced_subsample` | Class weighting |
| `GEOAI_TARGET_RESOLUTION` | `10` | Target grid resolution (metres) |
| `GEOAI_MAX_UPLOAD_MB` | `200` | Upload size limit |

Relative paths resolve against the repository root. `.env` is ignored by Git;
never commit credentials or local paths.

## 4. Running

```bash
source .venv/bin/activate
streamlit run app.py
```

Open <http://localhost:8501>. The Phase 1 page shows project status, configured
paths, and the ESA WorldCover class legend.

## 5. Using the geospatial and spectral pipeline

```python
from src.data.raster_loader import read_band, read_metadata
from src.data.vector_loader import geodesic_area_km2, load_aoi
from src.preprocessing.raster_clipping import clip_raster_to_aoi
from src.visualization.raster_visualizer import plot_rgb, save_band_preview

# 1. Inspect a raster without reading all pixels
metadata = read_metadata("data/raw/sentinel2/scene.tif")
print(metadata.to_dict())

# 2. Load and validate an AOI in the raster CRS
aoi = load_aoi("data/raw/aoi/study.geojson", target_crs=metadata.crs)
print(f"AOI area: {geodesic_area_km2(aoi):.2f} km²")

# 3. Clip all bands and write a georeferenced GeoTIFF
result = clip_raster_to_aoi(
    "data/raw/sentinel2/scene.tif",
    aoi,
    "data/processed/imagery/scene_clipped.tif",
)
print(result.crs, result.transform, result.width, result.height)
print(f"Valid pixels: {result.valid_pixel_count}/{result.total_pixel_count}")

# 4. Preview data
figure = plot_rgb(
    read_band(result.output_path, 3),
    read_band(result.output_path, 2),
    read_band(result.output_path, 1),
)
save_band_preview(result.output_path, "data/outputs/reports/band1.png")
```

All paths above are examples; provide your own local files. Tests use synthetic
fixtures and do not download data.

### Spectral features

```python
from src.features.feature_extraction import build_feature_stack, save_feature_stack
from src.features.spectral_indices import compute_spectral_indices
from src.preprocessing.raster_alignment import TargetGrid, align_bands
from src.visualization.index_visualizer import (
    plot_index_distribution,
    save_index_preview,
)

band_paths = {
    "B02": "data/raw/sentinel2/B02.tif",
    "B03": "data/raw/sentinel2/B03.tif",
    "B04": "data/raw/sentinel2/B04.tif",
    "B08": "data/raw/sentinel2/B08.tif",
    # Optional SWIR bands enable NDBI and add features:
    # "B11": "data/raw/sentinel2/B11.tif",
    # "B12": "data/raw/sentinel2/B12.tif",
}

# 1. Align every band to one target grid. Use a projected CRS in metres.
grid = TargetGrid.from_raster(band_paths["B04"])
aligned = align_bands(band_paths, reference_band="B04", resampling="bilinear")

# 2. Build a feature stack (band order: B02, B03, B04, B08, optional B11/B12,
#    then NDVI, NDWI, optional NDBI).
features = build_feature_stack(aligned)
print(features.feature_names)
print(features.statistics())

# 3. Extract only spatially valid pixels for scikit-learn.
X, rows, cols = features.as_sample_matrix()
print(X.shape)  # (n_valid_pixels, n_features)

# 4. Export a georeferenced Float32 GeoTIFF and NDVI visualisations.
save_feature_stack(features, "data/processed/features/scene_features.tif")
indices = compute_spectral_indices(
    {key: raster.data for key, raster in aligned.items()}
)
save_index_preview(indices["NDVI"], "data/outputs/reports/ndvi.png", name="NDVI")
plot_index_distribution(indices["NDVI"], name="NDVI")

# Optional: request NDBI explicitly when B11 is present.
compute_spectral_indices(
    {key: raster.data for key, raster in aligned.items()},
    include_ndbi=True,
)
```

Alignment converts data to `float32`, keeps CRS/transform/width/height from the
target grid, and represents masked or NoData pixels as `NaN`. Normalized
difference calculations mask zero denominators and non-finite inputs, so NDVI
and NDWI never contain infinities. NDVI and NDWI are always in `[-1, 1]` for
valid pixels; NDBI requires B11.

Notes:

- B02/B03/B04/B08 are natively 10 m; B11/B12 are natively 20 m. When SWIR is
  used, the default example still targets the 10 m B04 grid and resamples SWIR
  with the configured method. Document the chosen target resolution for any
  published analysis.
- Use `resampling="nearest"` for classification labels. Use `"bilinear"` or
  another continuous method for reflectance bands and indices, unless your
  analysis explicitly requires otherwise.
- If source bands sit on different grids, alignment derives the target grid
  from the reference band. To use a custom grid, build a `TargetGrid` from
  bounds and pass it to `align_raster_to_grid`.

### Labels and spatial splits

```python
from src.training.dataset_split import save_split_manifest, split_labeled_samples
from src.training.sample_generator import generate_labeled_samples

# 1. Align WorldCover labels to the exact feature grid with nearest resampling.
samples = generate_labeled_samples(
    features,
    "data/raw/worldcover/ESA_WorldCover_10m_2021_v200.tif",
)

print(samples.n_samples, samples.n_features)
print(samples.class_counts)
print(samples.class_distribution)
print(samples.missing_classes)
print(samples.imbalance_ratio)

# 2. Optional inverse-frequency weights for estimators accepting sample_weight.
weights = samples.balanced_sample_weights()

# 3. Assign whole spatial blocks to train or validation.
split = split_labeled_samples(
    samples,
    block_size=10,
    validation_fraction=0.2,
    random_seed=42,
)
print(split.train.X.shape, split.train.y.shape)
print(split.validation.X.shape, split.validation.y.shape)
print(split.pixel_disjoint)

# 4. Record configuration and class distributions as JSON.
save_split_manifest(split, "data/processed/labels/split_manifest.json")
```

Rules:

- Label rasters are resampled with `nearest` only. Never use bilinear or cubic
  interpolation for class codes.
- A sample is kept only when every feature is finite, the label is finite, and
  the code is one of the 11 ESA WorldCover classes.
- Label `0`, missing/outside-coverage pixels, and unknown codes are excluded.
  NoData and unknown-code counts are recorded in `samples.to_dict()`.
- `class_counts` and `class_distribution` describe the retained samples;
  `missing_classes` lists recognised classes with no samples.
- `balanced_sample_weights()` gives each present class equal total weight and
  sums to the number of samples. It complements the Random Forest
  `class_weight` setting rather than replacing the spatial split.
- Spatial splitting operates on blocks of pixels. Every sample in a block goes
  to exactly one side, so no pixel appears in both train and validation.
  Adjacent blocks can still be near each other; use larger blocks if a strict
  buffer between splits is required.
- The split seed, fraction, block size, block IDs, sample counts, and per-class
  distributions are written to the manifest.

WorldCover is a reference label source, not unquestioned ground truth. Label
noise and class/temporal mismatch remain limitations of any metrics derived
from it.

### Model training, evaluation, and persistence

```python
from config.settings import get_settings
from src.training.evaluation import evaluate_random_forest, save_evaluation_report
from src.training.model_manager import ModelRunMetadata, load_model, save_model
from src.training.random_forest import RandomForestConfig, train_random_forest

# 1. Train only on the training split. Other RF parameters use safe defaults;
#    tree count, depth, class weight, and seed come from GEOAI_* settings.
config = RandomForestConfig.from_settings(get_settings())
trained = train_random_forest(split.train, config)
print(trained.to_dict())

# 2. Evaluate on the held-out spatial validation split.
metrics = evaluate_random_forest(trained, split.validation)
print(metrics.accuracy, metrics.balanced_accuracy)
print(metrics.confusion_matrix)
for item in metrics.feature_importances:
    print(item.name, item.importance)

# 3. Save the real report and the model plus metadata.
save_evaluation_report(metrics, "data/outputs/reports/evaluation.json")
artifacts = save_model(
    trained,
    "data/models/random_forest.joblib",
    run_metadata=ModelRunMetadata(
        input_paths=tuple(band_paths.values()),
        image_dates=("2021-07-01",),
        aoi_path="data/raw/aoi/study.geojson",
        label_path="data/raw/worldcover/ESA_WorldCover_10m_2021_v200.tif",
        study_area="Example study area",
        notes="Example run; replace with actual provenance",
    ),
    evaluation=metrics,
)
print(artifacts.metadata_path)

# 4. Reload and verify prediction identity before reuse.
loaded = load_model(artifacts.model_path)
reloaded_predictions = loaded.predict(split.validation.X)
assert (reloaded_predictions == trained.predict(split.validation.X)).all()
```

Rules:

- Train on `split.train` and evaluate on `split.validation`. Do not report
  training-set accuracy as validation performance.
- Metrics come from actual predictions. There are no fallback or invented
  scores.
- `use_balanced_weights=True` is optional and recorded in metadata. The Random
  Forest `class_weight` setting is independent and also recorded.
- Feature names and order in `split.train` and `split.validation` must match;
  prediction and evaluation reject mismatches.
- The `.joblib` file is the fitted estimator. The adjacent
  `.metadata.json` file records format version, checksum, features, classes,
  parameters, seed, training counts and label source, optional run provenance,
  metrics, and software versions.
- Only load `.joblib` files from trusted sources. Joblib deserialisation can
  execute arbitrary code.
- A model trained on synthetic test data is not evidence of real-world
  land-cover accuracy. Report metrics only for the actual data and split used.

### Classification and area statistics

```python
from src.analysis.area_statistics import (
    compute_area_statistics,
    save_area_statistics_csv,
)
from src.classification.predictor import (
    classify_feature_raster,
    classify_feature_stack,
)

# 1. Classify a saved feature GeoTIFF in windows.
classification = classify_feature_raster(
    loaded,
    "data/processed/features/scene_features.tif",
    "data/outputs/classifications/land_cover.tif",
    window_size=512,
)
print(classification.to_dict())

# Alternatively, classify an existing in-memory FeatureStack.
classification = classify_feature_stack(
    loaded,
    features,
    "data/outputs/classifications/land_cover_in_memory.tif",
    window_size=512,
)

# 2. Calculate per-class counts, areas, and percentages.
statistics = compute_area_statistics(classification.output_path, window_size=1024)
for item in statistics.classes:
    print(item.code, item.name, item.pixel_count, item.area_km2, item.percentage)

# 3. Export the table as CSV.
save_area_statistics_csv(
    statistics,
    "data/outputs/statistics/land_cover_area.csv",
)
```

Rules:

- Use the same feature names and order used for training. Band descriptions in
  the feature GeoTIFF are checked; pass `feature_names=` explicitly when the
  raster has missing or non-matching descriptions.
- `window_size` controls memory use, not output resolution. Classification reads
  and writes each window while preserving the full CRS, transform, width, and
  height.
- Any invalid feature pixel becomes NoData (`0` by default). NoData must not
  overlap a valid land-cover class code.
- Output class maps use `uint8` ESA WorldCover codes plus NoData. They include a
  class legend tag and palette, so GIS software can display class names/colours.
- Area statistics use projected pixel dimensions when the CRS is projected. For
  geographic CRSs, WGS84 geodesic cell areas replace degrees-as-metres. Rotated
  geographic grids must be reprojected to a projected CRS first.
- Percentages use valid classified pixels as the denominator; NoData is excluded.
- Classification maps are model predictions, not verified ground truth. Area
  totals describe the map produced by the model, not confirmed real-world land
  cover.

### Change detection and transition matrices

```python
from src.analysis.change_detection import detect_changes
from src.analysis.transition_matrix import save_transition_matrix_csv

result = detect_changes(
    "data/outputs/classifications/land_cover_2021.tif",
    "data/outputs/classifications/land_cover_2024.tif",
    "data/outputs/change/land_cover_change.tif",
    alignment="strict",
    window_size=1024,
)

print(result.changed_pixels)
print(result.changed_percentage)
print(result.changed_area_km2)
for item in result.matrix.changed_transitions:
    print(
        item.from_name,
        "->",
        item.to_name,
        item.pixel_count,
        item.area_km2,
    )

save_transition_matrix_csv(
    result.matrix,
    "data/outputs/change/transition_matrix.csv",
)
```

Rules:

- `alignment="strict"` is the default and requires both classifications to have
  the same CRS, transform, width, and height. Use `alignment="reproject"` to
  warp the later classification onto the earlier grid with nearest-neighbour
  resampling. Categorical class codes are never interpolated.
- Only pixels valid in both rasters are compared. If either raster is NoData,
  the change raster is NoData. Changed means the earlier and later class codes
  differ.
- Change-raster values are `0` unchanged, `1` changed, and `255` NoData. The
  earlier raster's CRS, transform, width, and height are preserved.
- Transition-matrix rows are earlier classes and columns are later classes.
  Diagonal cells are unchanged area; off-diagonal cells are transitions. The
  matrix uses the sorted union of classes observed in either classification.
- Transition areas use projected pixel dimensions or WGS84 geodesic cells, never
  raw degrees.
- `result.to_dict()` and `result.matrix.to_dict()` include the scientific caveat.
  The change GeoTIFF also stores it in the `SCIENTIFIC_CAVEAT` tag.
- Differences between classifications are not proof of real-world change. For a
  defensible comparison, use consistent preprocessing, compatible observation
  periods, the same class definitions, and preferably the same trained model.

## 6. Testing and quality checks

```bash
make test
make test-cov
make lint
```

Equivalent direct commands:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m compileall -q src config tests app.py
```

Unit tests use synthetic data only; no downloads are required.

## 7. Data

Place data under `data/raw/` as described in
[DATA_SOURCES.md](DATA_SOURCES.md). Data, models, logs, and outputs are ignored
by Git.

## 8. Troubleshooting

### `python3 --version` shows 3.9

Use `python3.11` explicitly or let `uv` create the environment:

```bash
uv venv --python 3.11 .venv
```

### A geospatial wheel fails to install

Confirm Python is 3.11 and the platform is supported by the wheel. If a source
build is attempted on macOS, install GDAL via Homebrew and retry:

```bash
brew install gdal
```

This project does not require Homebrew GDAL on typical macOS/Linux setups.

### Streamlit does not start

Check the log output for Python tracebacks. Run:

```bash
.venv/bin/python -m streamlit run app.py --server.headless true
```

### Settings look stale

`get_settings()` is cached. In notebooks or tests call
`get_settings.cache_clear()` after changing environment variables.

## 9. Known limitations

- No dashboard functionality beyond the Phase 1 landing page.
- No AOI, imagery, or reference data bundled.
- Change detection compares classified maps, not raw imagery. Seasonal,
  atmospheric, acquisition, preprocessing, model, and classification differences
  can appear as change.
- Reprojection uses nearest-neighbour resampling; changing grids can still alter
  the effective pixel support and should be documented in comparisons.
- No pre-trained models or classification results are bundled. Models and
  classification outputs you create are local artifacts and are ignored by Git.
- Alignment, stacking, sampling, splitting, training, evaluation, and feature
  construction are in-memory; classification and area statistics can process
  rasters in windows, but the upstream feature cube must still fit in memory.
- Cloud masking is not implemented until SCL/QA input is explicitly supported.
- Block splitting prevents pixel overlap but does not add a geographic buffer
  between adjacent blocks.
- Repository licence not yet selected.
