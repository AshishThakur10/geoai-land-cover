# GeoAI-Based Land Cover Classification and Change Detection System

A modular geospatial and machine-learning project for classifying Sentinel-2
Level-2A land cover and comparing classifications from different dates. The
initial reference labels come from ESA WorldCover 2021 and the initial model is
a Random Forest classifier.

> **Status: Phase 1 (project setup) complete.** The repository currently provides
> the environment, configuration, class registry, logging, utilities, tests, and
> documentation. Raster processing, model training, classification, change
> detection, and the full dashboard are implemented in later phases.

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
├── app.py                     # Streamlit entry point (Phase 1 placeholder)
├── config/
│   ├── settings.py            # Environment-driven application settings
│   └── land_cover_classes.py  # ESA WorldCover class registry
├── data/                      # Local data and artefacts (not committed)
├── dashboard/                 # Dashboard pages (Phase 8)
├── docs/                      # Requirements, architecture, data, user guide
├── src/
│   ├── data/                  # Raster/vector loading and validation (Phase 2)
│   ├── preprocessing/         # Alignment, masking, clipping (Phase 2–3)
│   ├── features/              # Band stacking and spectral indices (Phase 3)
│   ├── training/              # Sampling, splitting, RF training (Phase 4–5)
│   ├── classification/        # Windowed prediction (Phase 6)
│   ├── analysis/              # Area statistics and change detection (Phase 6–7)
│   ├── visualization/         # Maps and charts (Phase 8)
│   ├── reports/               # Report generation (Phase 8)
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
.venv/bin/python -m ruff check .
.venv/bin/python -m compileall -q src config tests app.py
.venv/bin/python -m streamlit run app.py
```

The app serves at <http://localhost:8501> by default.

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
  pixels are spatially correlated. Later phases use spatially separated splits.
- A difference between two classifications is not proof of real environmental
  change; it may reflect seasonality, clouds, acquisition differences, or model
  error.
- Internal validation metrics are not independent scientific validation.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Project setup, environment, configuration, logging, tests, docs | Complete |
| 2 | GIS fundamentals: raster/vector loading, CRS, clipping, NoData | Next |
| 3 | Band alignment, multispectral stack, NDVI/NDWI/NDBI | Planned |
| 4 | WorldCover label alignment and spatial sampling | Planned |
| 5 | Random Forest training and evaluation | Planned |
| 6 | Windowed classification, GeoTIFF output, area statistics | Planned |
| 7 | Change detection and transition matrices | Planned |
| 8 | Interactive Streamlit dashboard and exports | Planned |
| 9 | Expanded testing, documentation, screenshots | Planned |

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for acceptance
criteria per phase.

## Attribution

- Sentinel-2 imagery: Copernicus Sentinel data.
- ESA WorldCover 2021: © ESA WorldCover project 2021, licensed under CC BY 4.0.
- Exact attribution and licence terms must be confirmed at download time.

## License

The repository licence has not been selected yet.
