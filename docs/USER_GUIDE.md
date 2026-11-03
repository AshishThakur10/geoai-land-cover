# User Guide

## Current status

Phase 1 (project setup) is complete. The application can be installed, tested,
and started as a Streamlit landing page. It does not yet load imagery, train a
model, classify land cover, or perform change detection.

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

## 5. Testing and quality checks

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

## 6. Data

Place data under `data/raw/` as described in
[DATA_SOURCES.md](DATA_SOURCES.md). Data, models, logs, and outputs are ignored
by Git.

## 7. Troubleshooting

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

## 8. Known limitations

- No dashboard functionality beyond the Phase 1 landing page.
- No AOI, imagery, or reference data bundled.
- No trained models or classification results.
- Repository licence not yet selected.
