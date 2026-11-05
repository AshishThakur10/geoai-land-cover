# User Guide

## Current status

Phase 3 (spectral feature extraction) is complete. The project can load
GeoTIFFs and GeoJSON/Shapefile boundaries, inspect metadata and CRS values,
repair invalid geometries, reproject AOIs, clip rasters with preserved
georeferencing and NoData, align and resample Sentinel-2 bands to a shared grid,
build a multispectral stack, compute NDVI/NDWI/optional NDBI, and export a
model-ready feature stack. Model training, classification, change detection,
and the full dashboard are not implemented yet.

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
- No trained models or classification results yet.
- Alignment and stacking are in-memory; windowed processing is introduced with
  classification in Phase 6.
- Cloud masking is not implemented until SCL/QA input is explicitly supported.
- Repository licence not yet selected.
