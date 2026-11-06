# GeoAI-Based Land Cover Classification and Change Detection System

A modular geospatial and machine-learning project for classifying Sentinel-2
Level-2A land cover and comparing classifications from different dates. The
initial reference labels come from ESA WorldCover 2021 and the initial model is
a Random Forest classifier.

> **Status: Phase 8 (interactive dashboard) complete.** The repository
> provides environment setup, configuration, logging, raster/vector loading with
> metadata and CRS handling, geometry validation/repair, reprojection, AOI
> clipping with NoData preservation, raster previews, band alignment and
> resampling, multispectral stacking, NDVI/NDWI/optional NDBI, a model-ready
> feature stack, WorldCover label alignment, audited valid-pixel sampling,
> class-distribution reporting, balanced sample weights, spatially separated
> train/validation blocks with JSON manifests, configurable Random Forest
> training, held-out evaluation, feature importance, model persistence with
> reproducibility metadata, windowed classification to georeferenced GeoTIFF,
> class legends, projected/geodesic area statistics with CSV export, aligned
> change maps, class-to-class transition matrices with areas, changed-area
> summaries, and an eight-section Streamlit dashboard with maps, charts, uploads,
> GeoTIFF/CSV/GeoJSON downloads, and analysis reports.

## Objectives

- Load and validate Sentinel-2 GeoTIFF imagery and AOI vector boundaries.
- Preserve coordinate reference systems, transforms, and georeferencing.
- Build spectral features (blue, green, red, NIR; optional SWIR) and indices
  (NDVI, NDWI, optional NDBI).
- Train and evaluate a Random Forest model using spatial train/validation
  splits to reduce leakage.
- Produce georeferenced land-cover maps and per-class area statistics.
- Compare two classification dates and generate a transition matrix.
- Provide an interactive Streamlit dashboard with maps, charts, and exports.

## Technology

- Python 3.11
- Rasterio, GeoPandas, Shapely, PyProj, Pyogrio
- NumPy, Pandas, SciPy
- Scikit-learn (Random Forest), Joblib
- Matplotlib, Seaborn, Plotly, Folium
- Streamlit and Streamlit-Folium

## Project structure

```text
geoai-land-cover/
├── app.py                     # Streamlit entry point and section navigation
├── config/
│   ├── settings.py            # Environment-driven application settings
│   ├── land_cover_classes.py  # ESA WorldCover class registry
│   └── sentinel2_bands.py     # Sentinel-2 band metadata and alias resolution
├── data/                      # Local data and artefacts (not committed)
├── dashboard/
│   ├── common.py              # Session state, cached loaders, UI helpers
│   ├── data_access.py         # Discovery, uploads, raster/JSON metadata
│   └── sections/              # Home, AOI, imagery, spectral, classification,
│                              # evaluation, change, and exports pages
├── docs/                      # Requirements, architecture, data, user guide
├── src/
│   ├── data/                  # Raster/vector loading, metadata, CRS validation
│   ├── preprocessing/         # AOI clipping, band alignment, image stacking
│   ├── features/              # Spectral indices and model feature extraction
│   ├── training/              # Sampling, splits, RF training/evaluation, models
│   ├── classification/        # Windowed prediction and classification rasters
│   ├── analysis/              # Area statistics, change detection, transitions
│   ├── visualization/         # Previews, dashboard charts, and Folium maps
│   ├── reports/               # Analysis report and table serialisation
│   └── utils/                 # Errors, logging, file and validation helpers
└── tests/                     # Unit tests (synthetic fixtures only)
```

## Setup

Prerequisites: Python 3.11 and either
[`uv`](https://docs.astral.sh/uv/) (recommended) or `python3.11 -m venv`.

### Option A: uv (recommended)

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Option B: standard venv + pip

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Rasterio and GeoPandas wheels bundle GDAL/PROJ on macOS, Linux, and Windows, so
a separate system GDAL installation is normally not required.

### Configuration

```bash
cp .env.example .env
```

Edit `.env` to change log levels, data directories, the optional default AOI,
the random seed, Random Forest defaults, target resolution, and upload limits.
Relative paths resolve against the project root. `.env` is never committed.

```bash
make install   # create .venv and install the project + dev tools
make test      # run tests
make test-cov  # run tests with coverage
make lint      # ruff
make run       # streamlit run app.py
```

## Commands

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest --cov=src --cov=config --cov=dashboard --cov-report=term-missing
.venv/bin/python -m ruff check .
.venv/bin/python -m compileall -q src config dashboard tests app.py
.venv/bin/python -m streamlit run app.py
```

The app serves at <http://localhost:8501> by default.

## Interactive dashboard

Start Streamlit and open <http://localhost:8501>. The dashboard has eight
sections:

- **Home** — inventory of imagery, classifications, change outputs, study
  areas, model metadata, and evaluation reports.
- **Study Area** — select or upload a GeoJSON/GeoPackage/zipped Shapefile,
  inspect features, area, CRS, and activate the boundary used by later sections.
- **Satellite Imagery** — inspect raster dimensions, CRS, pixel size, band
  descriptions, acquisition tags, and downsampled previews.
- **Spectral Analysis** — choose a band mapping, compute NDVI/NDWI and optional
  NDBI, and inspect index maps and distributions.
- **Classification** — open an existing or uploaded class raster, view its map,
  class distribution, area statistics, and model metadata.
- **Model Evaluation** — inspect held-out metrics, confusion matrix, per-class
  metrics, and feature importance from an evaluation JSON report.
- **Change Detection** — compare two classifications, run strict or reprojected
  change detection, and inspect changed area and the transition matrix.
- **Exports** — download the active AOI as WGS84 GeoJSON, classified/change
  GeoTIFFs, area/transition/evaluation CSVs, and a JSON or Markdown analysis
  report that includes only sections backed by real selected or computed data.

Uploads are size- and suffix-checked, stored under `data/uploads/`, and zipped
Shapefiles are extracted only after path traversal, symbolic-link, and
uncompressed-size checks. The dashboard does not train models or load joblib
files. It runs spectral indices and change detection through the same pipeline
code as the Python API; exported statistics are generated from those same
objects.

No dashboard section fabricates imagery or metrics. Model predictions, area
statistics, and change maps are derived products, not field-verified
observations. See the scientific limitations below.

### Example GIS pipeline (Phase 2)

```python
from src.data.raster_loader import read_band, read_metadata
from src.data.vector_loader import geodesic_area_km2, load_aoi
from src.preprocessing.raster_clipping import clip_raster_to_aoi
from src.visualization.raster_visualizer import save_band_preview

metadata = read_metadata("data/raw/sentinel2/scene.tif")
print(metadata.to_dict())

aoi = load_aoi("data/raw/aoi/study.geojson", target_crs=metadata.crs)
print(f"AOI area: {geodesic_area_km2(aoi):.2f} km²")

result = clip_raster_to_aoi(
    "data/raw/sentinel2/scene.tif",
    aoi,
    "data/processed/imagery/scene_clipped.tif",
)
print(f"Clipped grid: {result.width}x{result.height}, CRS={result.crs}")
print(f"Valid pixels: {result.valid_pixel_count}/{result.total_pixel_count}")

save_band_preview(
    "data/processed/imagery/scene_clipped.tif",
    "data/outputs/reports/scene_band1.png",
)
```

Replace the paths with real local files. No sample imagery is bundled.

### Example spectral feature pipeline (Phase 3)

```python
from src.features.feature_extraction import build_feature_stack, save_feature_stack
from src.features.spectral_indices import compute_spectral_indices
from src.preprocessing.raster_alignment import align_bands
from src.visualization.index_visualizer import save_index_preview

band_paths = {
    "B02": "data/raw/sentinel2/B02.tif",
    "B03": "data/raw/sentinel2/B03.tif",
    "B04": "data/raw/sentinel2/B04.tif",
    "B08": "data/raw/sentinel2/B08.tif",
}

# Align every band to B04's 10 m grid (the core bands are natively 10 m).
aligned = align_bands(band_paths, reference_band="B04", resampling="bilinear")

features = build_feature_stack(aligned)
save_feature_stack(features, "data/processed/features/scene_features.tif")

indices = compute_spectral_indices({key: raster.data for key, raster in aligned.items()})
save_index_preview(indices["NDVI"], "data/outputs/reports/ndvi.png", name="NDVI")

print(f"Features: {features.feature_names}")
print(f"Valid pixels: {features.valid_pixel_count}/{features.total_pixel_count}")
```

Add B11 (and optionally B12) to `band_paths` to include SWIR features and NDBI.
The example intentionally omits them to show the 10 m core configuration.

### Example labeled samples and spatial split (Phase 4)

```python
from src.training.dataset_split import save_split_manifest, split_labeled_samples
from src.training.sample_generator import generate_labeled_samples

# WorldCover labels are aligned to the exact feature grid with nearest resampling.
samples = generate_labeled_samples(
    features,
    "data/raw/worldcover/ESA_WorldCover_10m_2021_v200.tif",
)
print(samples.to_dict())  # class distribution, ignored pixels, imbalance ratio

split = split_labeled_samples(
    samples,
    block_size=10,          # whole 10x10-pixel blocks stay on one side
    validation_fraction=0.2,
    random_seed=42,
)
save_split_manifest(split, "data/processed/labels/split_manifest.json")

print(split.train.X.shape, split.train.y.shape)
print(split.validation.X.shape, split.validation.y.shape)
print(f"Pixel-disjoint: {split.pixel_disjoint}")
```

Labels must overlap the feature extent and use ESA WorldCover class codes.
NoData (`0`), non-finite aligned labels, unknown codes, and pixels with any
invalid feature are excluded and counted. The split manifest records the block
IDs, seed, fraction, sample counts, and per-class distributions.

### Example Random Forest training and persistence (Phase 5)

```python
from config.settings import get_settings
from src.training.evaluation import evaluate_random_forest, save_evaluation_report
from src.training.model_manager import ModelRunMetadata, load_model, save_model
from src.training.random_forest import RandomForestConfig, train_random_forest

config = RandomForestConfig.from_settings(get_settings())
trained = train_random_forest(split.train, config)

metrics = evaluate_random_forest(trained, split.validation)
save_evaluation_report(metrics, "data/outputs/reports/evaluation.json")
print(metrics.to_dict())  # real held-out metrics, not fabricated values

artifacts = save_model(
    trained,
    "data/models/random_forest.joblib",
    run_metadata=ModelRunMetadata(
        input_paths=tuple(band_paths.values()),
        image_dates=("2021-07-01",),
        aoi_path="data/raw/aoi/study.geojson",
        label_path="data/raw/worldcover/ESA_WorldCover_10m_2021_v200.tif",
        study_area="Example study area",
    ),
    evaluation=metrics,
)
print(artifacts.metadata_path)

reloaded = load_model(artifacts.model_path)
assert (reloaded.predict(split.validation.X) == trained.predict(split.validation.X)).all()
```

Model metadata records feature names/order, class codes, hyperparameters, seed,
training counts and label source, optional input paths/dates/AOI, held-out
metrics, and software versions. Only load `.joblib` files from trusted sources:
joblib deserialisation can execute code.

### Example windowed classification and area statistics (Phase 6)

```python
from src.analysis.area_statistics import (
    compute_area_statistics,
    save_area_statistics_csv,
)
from src.classification.predictor import classify_feature_raster

classification = classify_feature_raster(
    loaded,
    "data/processed/features/scene_features.tif",
    "data/outputs/classifications/land_cover.tif",
    window_size=512,
)
print(classification.to_dict())

statistics = compute_area_statistics(classification.output_path)
print(statistics.to_dict())
save_area_statistics_csv(
    statistics,
    "data/outputs/statistics/land_cover_area.csv",
)
```

The classification GeoTIFF preserves the feature grid's CRS, transform, width,
and height. Pixels with any invalid feature become the defined NoData code
(`0`), and only valid ESA WorldCover codes appear elsewhere. The dataset includes
a class legend tag and palette. Area statistics use projected pixel dimensions
when possible and WGS84 geodesic cell areas for geographic rasters; longitude and
latitude degrees are never treated as metres.

Classification maps are model outputs, not verified ground truth. Internal
validation metrics and area totals do not establish real-world accuracy.

### Example change detection and transition export (Phase 7)

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
print(result.to_dict())

save_transition_matrix_csv(
    result.matrix,
    "data/outputs/change/transition_matrix.csv",
)
```

The change raster uses `0` for unchanged, `1` for changed, and `255` for NoData
where either classification is invalid. It preserves the earlier raster's CRS,
transform, width, and height. With `alignment="reproject"`, the later raster is
warped onto the earlier grid using nearest-neighbour resampling; the default
`strict` mode rejects mismatched grids. Transition areas use the same projected
or WGS84 geodesic rules as area statistics.

A difference between classifications does not prove real-world land-cover
change. It may reflect seasonality, clouds, acquisition differences,
classification errors, or different models, preprocessing, or class
definitions.

## Data

No satellite imagery, reference rasters, or trained models are committed to this
repository. See [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) for:

- Sentinel-2 Level-2A acquisition from the Copernicus Data Space Ecosystem.
- ESA WorldCover 2021 v200 tile selection.
- AOI sources and licensing/attribution notes.

Unit tests use synthetic arrays and fixtures and do not require downloads.

## Scientific limitations

- ESA WorldCover is used as a **reference label source**, not unquestioned
  ground truth. Label noise propagates into model metrics.
- Sentinel-2 imagery and WorldCover labels must represent compatible periods
  and coverage. Do not pair 2021 labels with unrelated recent imagery and claim
  that the labels are current.
- Random pixel-level train/validation splits leak information because nearby
  pixels are spatially correlated. This project uses spatially separated blocks,
  but adjacent blocks can still be nearby.
- A difference between two classifications is not proof of real environmental
  change; it may reflect seasonality, clouds, acquisition differences, or model
  error.
- Internal validation metrics are not independent scientific validation.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Project setup, environment, configuration, logging, tests, docs | Complete |
| 2 | GIS fundamentals: raster/vector loading, CRS, clipping, NoData | Complete |
| 3 | Band alignment, multispectral stack, NDVI/NDWI/NDBI | Complete |
| 4 | WorldCover label alignment and spatial sampling | Complete |
| 5 | Random Forest training, evaluation, model persistence | Complete |
| 6 | Windowed classification, GeoTIFF output, area statistics | Complete |
| 7 | Change detection and transition matrices | Complete |
| 8 | Interactive Streamlit dashboard and exports | Complete |
| 9 | Expanded testing, documentation, screenshots, real-data examples | Next |

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for acceptance
criteria per phase.

## Attribution

- Sentinel-2 imagery: Copernicus Sentinel data.
- ESA WorldCover 2021: © ESA WorldCover project 2021, licensed under CC BY 4.0.
- Exact attribution and licence terms must be confirmed at download time.

## License

The repository licence has not been selected yet.
