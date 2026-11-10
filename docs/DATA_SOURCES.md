# Data Sources

This repository does **not** include satellite imagery, reference rasters, AOI
files, trained models, or generated outputs. Download data locally into the
directories below. Large files are excluded by `.gitignore`.

## 1. Sentinel-2 Level-2A imagery

**Source:** Copernicus Data Space Ecosystem
<https://dataspace.copernicus.eu/>

Select:

- Sentinel-2, Level-2A (surface reflectance)
- Bands B02, B03, B04, B08 initially; B11 and B12 optional (20 m)
- Low cloud cover and a single acquisition date per experiment
- A period compatible with the reference labels (see Section 2)

Place GeoTIFFs under `data/raw/sentinel2/`. The initial pipeline reads local
files only; automatic acquisition is a future enhancement.

**Attribution:** Copernicus Sentinel data. Confirm the current licence and
attribution requirements on the Copernicus Data Space Ecosystem at download
time.

### Resolution note

B02/B03/B04/B08 are 10 m; B11/B12 are 20 m. Bands must be resampled to a common
target grid. The default target resolution is 10 m (`GEOAI_TARGET_RESOLUTION`).
Do not silently mix grids; alignment and resampling are implemented in Phase 3.

## 2. ESA WorldCover 2021

**Source:** <https://esa-worldcover.org/en>

Use the **2021 v200** classification map at approximately 10 m resolution.
Download only the tiles covering the AOI; do not download the global product.

Place tiles under `data/raw/worldcover/`.

### Class codes

| Code | Class |
|---:|---|
| 10 | Tree Cover |
| 20 | Shrubland |
| 30 | Grassland |
| 40 | Cropland |
| 50 | Built-up |
| 60 | Bare / Sparse Vegetation |
| 70 | Snow and Ice |
| 80 | Permanent Water Bodies |
| 90 | Herbaceous Wetland |
| 95 | Mangroves |
| 100 | Moss and Lichen |

Unrecognised codes (including 0/NoData) are excluded from training.

**Attribution:** © ESA WorldCover project 2021, licensed under CC BY 4.0.
Confirm the licence on the official site before redistribution.

### Temporal compatibility warning

WorldCover 2021 labels are only appropriate for Sentinel-2 imagery from a
compatible period (ideally 2021). Pairing 2021 labels with unrelated recent
imagery and presenting the labels as current is scientifically invalid.
Experiments must record acquisition dates and label provenance.

## 3. Area of Interest (AOI)

Supported formats: GeoJSON, GeoPackage, and Shapefile (upload a ZIP containing
the Shapefile sidecar files). Place files under `data/raw/aoi/` or upload them
through the dashboard Study Area section.

Possible boundary sources include national/regional open-data portals,
OpenStreetMap administrative boundaries (check the ODbL requirements), or a
hand-drawn polygon. Record the source, licence, and attribution for any AOI
used in an experiment.

Set `GEOAI_DEFAULT_AOI` in `.env` to an optional default boundary path. The
path may be left empty; no AOI is bundled with the repository.

## 4. Sample data

A small sample dataset (one Sentinel-2 subset, the corresponding WorldCover
tile, and one AOI) will be documented and added only when openly licensed data
can be included or downloaded reproducibly. Until then, unit tests use
synthetic arrays and fixtures, and real-data runs require the user to download
the files listed above.

Do not invent or commit fabricated satellite data as if it were real imagery.

## 5. Recommended first study area

Choose a small city- or district-level AOI (roughly tens to a few hundred square
kilometres) to keep processing manageable on a standard laptop. Ensure the AOI:

- lies fully inside one WorldCover tile,
- is covered by one Sentinel-2 Level-2A acquisition,
- contains a mix of land-cover classes for a meaningful experiment.

## 6. Data not to commit

- Sentinel-2 GeoTIFFs and subsets
- WorldCover tiles
- AOI files containing sensitive locations (optional)
- Trained models (`*.joblib`) and generated rasters (`*.tif`)
- Logs, caches, and `.env`

See `.gitignore` for the enforced patterns.
