# GeoAI-Based Land Cover Classification and Change Detection System

**Project Type:** Personal Project / Machine Learning / Geospatial Analysis  
**Status:** In Progress  
**Developer:** Ashish Thakur  
**Primary Language:** Python  
**Application Type:** Geospatial Analysis Platform with Interactive Dashboard  
**Target:** Portfolio-ready project suitable for GitHub and resume demonstration

---

# 1. Project Overview

## 1.1 Project Description

The GeoAI-Based Land Cover Classification and Change Detection System is a geospatial analysis application that uses satellite imagery and machine learning to identify, classify, visualize, and analyze different land-cover categories.

The application will process multispectral satellite imagery, particularly Sentinel-2 imagery, and use machine-learning algorithms to classify geographical areas into categories such as:

- Tree Cover
- Shrubland
- Grassland
- Cropland
- Built-up Areas
- Bare or Sparse Vegetation
- Water Bodies
- Wetlands
- Mangroves
- Snow and Ice
- Moss and Lichen

The system will also provide functionality to compare land-cover maps from different periods and identify possible changes in land cover.

The objective is to demonstrate how GeoAI, remote sensing, GIS, and machine learning can support environmental monitoring, urban development analysis, vegetation monitoring, and climate-resilience research.

## 1.2 Main Objectives

1. Understand and process satellite imagery.
2. Load, inspect, and preprocess raster datasets.
3. Work with vector geographical boundaries.
4. Extract spectral features from multispectral satellite images.
5. Train a machine-learning model for land-cover classification.
6. Generate classified land-cover maps.
7. Calculate the area covered by each land-cover category.
8. Compare land-cover classifications from different years.
9. Visualize geospatial information through an interactive dashboard.
10. Export classification results and analytical reports.

## 1.3 Project Scope

The project will initially focus on a small geographical region to keep processing requirements manageable.

The Area of Interest (AOI) must be configurable.

The application should support:

- City-level analysis
- District-level analysis
- Custom geographical boundaries
- GeoJSON and Shapefile inputs
- Multiple satellite image dates
- Multiple classification experiments

A default sample geographical region can be configured during initial setup. The application must not be restricted to one location.

---

# 2. Technology Stack

## 2.1 Programming Language

**Python 3.11**

Python will be used for:
- Geospatial data processing
- Satellite imagery analysis
- Machine learning
- Data analysis
- Backend processing
- Visualization

## 2.2 Core Libraries

| Library | Purpose |
|---|---|
| NumPy | Numerical operations and image arrays |
| Pandas | Data analysis and tabular processing |
| Rasterio | Reading, writing, and processing raster imagery |
| GeoPandas | Vector data processing and spatial operations |
| Shapely | Geometric operations |
| PyProj | Coordinate reference system transformations |
| Matplotlib | Scientific visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine-learning model development |
| Joblib | Saving and loading trained models |
| Pillow | Image processing |
| SciPy | Scientific and spatial computations |

## 2.3 Satellite Data

**Primary satellite data source: Sentinel-2**

Use Sentinel-2 Level-2A surface reflectance imagery.

Important spectral bands:

| Band | Description | Native Resolution |
|---|---|---|
| B02 | Blue | 10 m |
| B03 | Green | 10 m |
| B04 | Red | 10 m |
| B08 | Near Infrared | 10 m |
| B11 | Shortwave Infrared 1 | 20 m |
| B12 | Shortwave Infrared 2 | 20 m |

For the first implementation, use B02, B03, B04, and B08.

Support B11 and B12 as optional features after the initial pipeline works. When combining different resolutions, resample consistently and document the selected target resolution.

## 2.4 Reference Land-Cover Dataset

Use ESA WorldCover 2021 as the initial reference dataset.

It provides:
- Global land-cover information.
- Approximately 10-metre spatial resolution.
- Eleven land-cover categories.
- GeoTIFF raster data.

Use its classification map as training-label data, not as an unquestioned ground-truth dataset.

Important: The Sentinel-2 imagery and reference labels should represent compatible periods and geographical coverage. Do not combine a 2021 reference map with unrelated recent imagery and claim that the labels are current.

## 2.5 Machine-Learning Algorithms

Initial algorithm:

**Random Forest Classifier**

Reasons:
- Suitable for multispectral image classification.
- Handles nonlinear relationships.
- Works with numerical spectral features.
- Provides feature importance.
- Does not require GPU acceleration.
- Suitable for a first practical GeoAI project.

Possible future algorithms:
- Extra Trees Classifier
- Support Vector Machine
- XGBoost
- CNN-based classification
- Deep-learning segmentation

Do not implement all algorithms in the MVP. First establish a reliable Random Forest baseline.

## 2.6 Dashboard

Use Streamlit for the initial interactive dashboard.

Use Folium or another suitable geospatial mapping library for interactive map visualization.

The dashboard should be simple, responsive, and suitable for demonstration.

---

# 3. Functional Requirements

## FR-01: Area of Interest Management

The application must allow users to define the geographical area to analyze.

Supported input formats:
- GeoJSON
- Shapefile
- GeoPackage (optional)

Requirements:

1. Upload a geographical boundary.
2. Read and validate the geometry.
3. Detect its coordinate reference system.
4. Reproject the geometry when necessary.
5. Display the selected boundary on a map.
6. Calculate its approximate geographical area.
7. Clip satellite imagery to the selected boundary.

The system must handle invalid geometries and missing coordinate reference systems gracefully.

## FR-02: Satellite Imagery Input

The application must support loading satellite imagery from local files.

Initial supported format:
- GeoTIFF

Required functionality:

1. Load multispectral raster imagery.
2. Read raster metadata.
3. Identify the number of bands.
4. Identify spatial resolution.
5. Read coordinate reference system.
6. Read image dimensions.
7. Identify NoData values.
8. Display a satellite image preview.
9. Validate compatibility between the imagery and AOI.

The initial version should not depend on automatic satellite downloads.

Future functionality can integrate Copernicus Data Space APIs for downloading Sentinel-2 imagery.

## FR-03: Image Preprocessing

The system must implement a preprocessing pipeline.

Processing steps:

1. Load individual spectral bands.
2. Check band dimensions.
3. Check coordinate reference systems.
4. Reproject when necessary.
5. Resample bands to a consistent grid.
6. Apply cloud and invalid-pixel masking where suitable quality information is available.
7. Remove NoData pixels.
8. Clip imagery using the AOI.
9. Convert pixel values into appropriate numerical reflectance features.
10. Generate a clean multispectral image stack.

The system should preserve georeferencing information throughout processing.

## FR-04: Spectral Index Calculation

Implement the following spectral indices.

### NDVI – Normalized Difference Vegetation Index

Formula:

NDVI = (NIR - Red) / (NIR + Red)

Uses:
- Vegetation monitoring.
- Identifying healthy vegetation.
- Differentiating vegetation from non-vegetated areas.

### NDWI – Normalized Difference Water Index

Formula:

NDWI = (Green - NIR) / (Green + NIR)

Uses:
- Identifying possible water bodies.
- Monitoring surface-water patterns.

### NDBI – Normalized Difference Built-up Index

Formula:

NDBI = (SWIR - NIR) / (SWIR + NIR)

Uses:
- Identifying possible built-up surfaces.
- Supporting urban-area analysis.

Important: NDWI, NDBI, and other spectral indices are indicators, not definitive land-cover labels. Their usefulness depends on the sensor bands and local conditions.

The application must safely handle zero denominators and invalid pixels.

NDBI requires SWIR imagery, so it should be enabled only when the required band is available.

## FR-05: Reference Label Preparation

Use ESA WorldCover 2021 as the initial training-label source.

Required tasks:

1. Load the reference land-cover raster.
2. Read the class values.
3. Map class codes to meaningful class names.
4. Align the reference raster with the Sentinel-2 feature raster.
5. Match spatial resolution and coordinate systems.
6. Mask invalid reference pixels.
7. Create training samples.

Initial ESA WorldCover class mapping:

| Code | Class |
|---|---|
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

Unrecognized or invalid codes must be excluded from training.

## FR-06: Machine-Learning Model Training

Implement a Random Forest classification pipeline.

Input features:

- Blue reflectance
- Green reflectance
- Red reflectance
- Near Infrared reflectance
- Optional SWIR bands
- NDVI
- NDWI
- Optional NDBI

Target:
- Reference land-cover class.

Training requirements:

1. Generate valid pixel-level feature samples.
2. Remove invalid samples.
3. Handle class imbalance where appropriate.
4. Split training and validation data.
5. Train a Random Forest model.
6. Save the trained model.
7. Save model metadata.
8. Calculate evaluation metrics.

Suggested initial configuration:

- n_estimators: 150
- max_depth: configurable
- random_state: 42
- n_jobs: -1
- class_weight: balanced_subsample

These are starting parameters, not guaranteed optimal values.

Important evaluation requirement:

Do not rely only on random pixel-level splitting. Nearby satellite pixels are spatially correlated, which can cause data leakage and misleadingly high accuracy.

Prefer spatially separated training and validation regions or spatial blocks.

## FR-07: Model Evaluation

The system must calculate:

- Overall Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Per-class performance
- Class support

Generate:

1. Classification report.
2. Confusion matrix visualization.
3. Model evaluation summary.
4. Feature importance chart.

The dashboard must display the evaluation results.

Do not claim that the model is scientifically validated merely because it performs well on its internal validation split.

## FR-08: Land-Cover Prediction

After training, the application must classify the selected satellite imagery.

Required output:

- Classified land-cover raster.
- Georeferenced GeoTIFF.
- Class legend.
- Class names.
- Class distribution statistics.

Each output pixel must contain a valid land-cover class code or a defined NoData value.

The classification must preserve:
- Coordinate reference system.
- Transform.
- Spatial dimensions.
- Geographical alignment.

For large imagery, process data in windows or chunks instead of loading the entire raster into memory.

## FR-09: Land-Cover Area Statistics

Calculate the area occupied by each class.

Output example:

| Land-Cover Type | Pixel Count | Area (km²) | Percentage |
|---|---:|---:|---:|
| Tree Cover | Calculated | Calculated | Calculated |
| Cropland | Calculated | Calculated | Calculated |
| Built-up | Calculated | Calculated | Calculated |
| Water | Calculated | Calculated | Calculated |
| Grassland | Calculated | Calculated | Calculated |

Area calculations must use an appropriate projected coordinate reference system or geodesic area calculations.

Do not calculate square kilometres by treating longitude and latitude degrees as metres.

## FR-10: Land-Cover Change Detection

The application should compare two classified maps representing different dates.

Example:

- Classification A: Earlier observation.
- Classification B: Later observation.

Required functionality:

1. Load two classified rasters.
2. Validate their geographical alignment.
3. Reproject or align if necessary.
4. Compare corresponding pixels.
5. Identify changed pixels.
6. Generate a change-detection raster.
7. Calculate class-to-class transitions.
8. Calculate the area of each transition.
9. Generate a change summary.

Example transitions:

- Vegetation → Built-up
- Cropland → Built-up
- Vegetation → Bare Land
- Land → Water
- Water → Land

Generate a transition matrix showing the area of each transition.

Important scientific limitation:

A difference between two classifications does not automatically prove actual environmental change. Differences can result from seasonal variation, cloud contamination, classification errors, image acquisition differences, or model changes.

For reliable comparisons, use consistent preprocessing, compatible observation periods, the same class definitions, and preferably the same trained model.

## FR-11: Interactive Dashboard

The Streamlit dashboard should provide the following sections.

### Dashboard Home

Display:
- Project title.
- Short project description.
- Selected study area.
- Latest processing status.
- Total analyzed area.
- Available datasets.

### Study Area

Display:
- Uploaded geographical boundary.
- Area information.
- Coordinate reference system.
- Bounding box.

### Satellite Imagery

Display:
- Satellite image preview.
- Available spectral bands.
- Image acquisition date, if available.
- Spatial resolution.
- Raster metadata.

### Spectral Analysis

Display:
- NDVI map.
- NDWI map.
- NDBI map when available.
- Spectral distribution charts.

### Land-Cover Classification

Display:
- Classified land-cover map.
- Class legend.
- Class distribution chart.
- Area statistics.
- Model information.

### Model Evaluation

Display:
- Overall accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion matrix.
- Feature importance.

### Change Detection

Display:
- Earlier classified map.
- Later classified map.
- Change-detection map.
- Transition matrix.
- Area changes.
- Percentage changes.

### Export Results

Allow users to export:
- Classified GeoTIFF.
- Change-detection GeoTIFF.
- CSV statistics.
- Confusion matrix.
- Classification report.
- Analysis report.

---

# 4. Non-Functional Requirements

## NFR-01: Performance

- Support small and medium geographical regions on a standard development laptop.
- Avoid loading unnecessarily large raster datasets into memory.
- Use raster window processing when appropriate.
- Avoid unnecessary duplicate processing.
- Cache expensive operations where practical.

## NFR-02: Reliability

- Validate input files.
- Handle corrupted or unsupported files.
- Handle missing bands.
- Handle invalid geometries.
- Handle incompatible CRS information.
- Provide meaningful error messages.
- Maintain processing logs.

## NFR-03: Code Quality

- Follow Python coding conventions.
- Use modular architecture.
- Add type hints where practical.
- Write reusable functions.
- Document important algorithms.
- Avoid putting all processing logic in a single Python file.

## NFR-04: Reproducibility

Every classification experiment should record:

- Input imagery identifiers or filenames.
- Reference dataset information.
- Acquisition dates where available.
- AOI details.
- Feature configuration.
- Model parameters.
- Random seed.
- Evaluation metrics.
- Software versions.

## NFR-05: Security

- Validate uploaded files.
- Do not expose API credentials.
- Keep secrets in environment variables.
- Do not commit sensitive credentials to Git.
- Limit upload sizes where practical.

---

# 5. System Architecture

Use a modular architecture.

### Layer 1: Data Input

Responsibilities:
- AOI upload.
- Satellite imagery loading.
- Reference dataset loading.
- Input validation.

### Layer 2: Geospatial Processing

Responsibilities:
- Raster processing.
- Vector processing.
- CRS transformations.
- Resampling.
- Clipping.
- Cloud masking.
- Feature extraction.

### Layer 3: Machine Learning

Responsibilities:
- Training sample generation.
- Dataset splitting.
- Random Forest training.
- Model evaluation.
- Prediction.
- Model persistence.

### Layer 4: Analytical Processing

Responsibilities:
- Area calculation.
- Land-cover statistics.
- Change detection.
- Transition matrix generation.

### Layer 5: Visualization

Responsibilities:
- Interactive maps.
- Charts.
- Classification legends.
- Statistical reports.
- Downloadable outputs.

### Layer 6: User Interface

Responsibilities:
- Streamlit dashboard.
- File uploads.
- Processing controls.
- Analysis results.
- Export functionality.

---

# 6. Recommended Project Folder Structure

Create a maintainable project structure.

```text
geoai-land-cover/
│
├── app.py
├── requirements.txt
├── pyproject.toml
├── README.md
├── .gitignore
├── .env.example
│
├── config/
│   ├── settings.py
│   └── land_cover_classes.py
│
├── data/
│   ├── raw/
│   │   ├── sentinel2/
│   │   ├── worldcover/
│   │   └── aoi/
│   │
│   ├── processed/
│   │   ├── imagery/
│   │   ├── features/
│   │   └── labels/
│   │
│   ├── models/
│   │
│   └── outputs/
│       ├── classifications/
│       ├── change_detection/
│       ├── reports/
│       └── statistics/
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── raster_loader.py
│   │   ├── vector_loader.py
│   │   ├── dataset_validator.py
│   │   └── data_metadata.py
│   │
│   ├── preprocessing/
│   │   ├── raster_alignment.py
│   │   ├── cloud_masking.py
│   │   ├── raster_clipping.py
│   │   └── image_stacking.py
│   │
│   ├── features/
│   │   ├── spectral_indices.py
│   │   └── feature_extraction.py
│   │
│   ├── training/
│   │   ├── sample_generator.py
│   │   ├── dataset_split.py
│   │   ├── random_forest.py
│   │   ├── evaluation.py
│   │   └── model_manager.py
│   │
│   ├── classification/
│   │   └── predictor.py
│   │
│   ├── analysis/
│   │   ├── area_statistics.py
│   │   ├── change_detection.py
│   │   └── transition_matrix.py
│   │
│   ├── visualization/
│   │   ├── raster_visualizer.py
│   │   ├── charts.py
│   │   └── map_visualizer.py
│   │
│   ├── reports/
│   │   └── report_generator.py
│   │
│   └── utils/
│       ├── logging_config.py
│       ├── file_utils.py
│       └── validation.py
│
├── dashboard/
│   ├── home.py
│   ├── study_area.py
│   ├── satellite_imagery.py
│   ├── spectral_analysis.py
│   ├── classification.py
│   ├── model_evaluation.py
│   ├── change_detection.py
│   └── exports.py
│
├── tests/
│   ├── test_raster_loader.py
│   ├── test_vector_loader.py
│   ├── test_spectral_indices.py
│   ├── test_preprocessing.py
│   ├── test_training.py
│   ├── test_classification.py
│   └── test_change_detection.py
│
└── docs/
    ├── PROJECT_REQUIREMENTS.md
    ├── ARCHITECTURE.md
    ├── DATA_SOURCES.md
    └── USER_GUIDE.md
```

Do not commit large satellite datasets, trained model binaries, or generated raster outputs to GitHub.

Use `.gitignore` and provide instructions for obtaining the datasets.

---

# 7. Dataset Requirements and Acquisition

## 7.1 Sentinel-2

Source: Copernicus Data Space Ecosystem.

Website:
https://dataspace.copernicus.eu/

Select:
- Sentinel-2
- Level-2A
- Appropriate geographical coverage
- Compatible acquisition dates
- Low cloud coverage

Initially, allow users to download imagery manually and upload GeoTIFF files.

Later, implement automatic data acquisition using the Copernicus Data Space APIs.

## 7.2 ESA WorldCover

Source:
https://esa-worldcover.org/en

Use the 2021 v200 classification map.

Download only the tiles covering the selected AOI.

Avoid downloading the entire global dataset.

## 7.3 Sample Data

Provide a small sample dataset for testing the complete workflow.

Include:
- One small Sentinel-2 sample.
- One corresponding WorldCover reference raster.
- One AOI GeoJSON file.

Use openly licensed data and document the source, date, licence, and attribution.

The sample data should be small enough for a normal laptop.

---

# 8. Processing Workflow

The complete processing pipeline should follow this order.

```text
START
  |
  v
Upload AOI
  |
  v
Validate AOI Geometry
  |
  v
Load Sentinel-2 Imagery
  |
  v
Validate Raster Metadata
  |
  v
Align and Clip Raster Bands
  |
  v
Apply Cloud / NoData Mask
  |
  v
Generate Multispectral Stack
  |
  v
Calculate Spectral Indices
  |
  v
Load ESA WorldCover Labels
  |
  v
Align Reference Labels
  |
  v
Generate Training Samples
  |
  v
Spatial Training / Validation Split
  |
  v
Train Random Forest Model
  |
  v
Evaluate Model Performance
  |
  v
Classify Satellite Imagery
  |
  v
Generate Land-Cover Statistics
  |
  v
Visualize Classified Map
  |
  v
Compare Multiple Dates
  |
  v
Generate Change Detection Results
  |
  v
Export Results and Reports
  |
  v
END
```

---

# 9. Development Phases

Development must happen incrementally.

Do not attempt to build all functionality in one step.

## Phase 1: Project Setup

Tasks:
- Create Python project.
- Configure virtual environment.
- Install required packages.
- Create folder structure.
- Configure logging.
- Create configuration files.
- Create README.

Deliverable:
A working Python project that runs without dependency errors.

## Phase 2: GIS Fundamentals and Data Processing

Tasks:
- Load GeoTIFF.
- Load GeoJSON and Shapefile.
- Read raster metadata.
- Read CRS information.
- Visualize raster bands.
- Reproject vector geometry.
- Clip raster using AOI.
- Validate NoData handling.

Deliverable:
A working geospatial data-processing pipeline.

## Phase 3: Satellite Image Feature Extraction

Tasks:
- Load Sentinel-2 bands.
- Align different raster resolutions.
- Generate multispectral stack.
- Calculate NDVI.
- Calculate NDWI.
- Calculate NDBI when SWIR data is available.
- Visualize spectral indices.

Deliverable:
Processed satellite features and spectral index maps.

## Phase 4: Training Dataset Preparation

Tasks:
- Load ESA WorldCover.
- Map class codes.
- Align labels and features.
- Extract valid pixel samples.
- Handle class imbalance.
- Create spatially separated training and validation sets.

Deliverable:
A prepared training dataset.

## Phase 5: Machine-Learning Development

Tasks:
- Implement Random Forest.
- Train the classifier.
- Evaluate the model.
- Generate confusion matrix.
- Calculate per-class metrics.
- Generate feature importance.
- Save model and metadata.

Deliverable:
A trained and evaluated land-cover classification model.

## Phase 6: Land-Cover Classification

Tasks:
- Load trained model.
- Predict land-cover classes.
- Process imagery in manageable windows.
- Generate classified GeoTIFF.
- Generate class distribution statistics.
- Calculate geographical area.

Deliverable:
A georeferenced land-cover classification map.

## Phase 7: Change Detection

Tasks:
- Load two classification rasters.
- Align both datasets.
- Compare class values.
- Generate transition matrix.
- Calculate changed area.
- Generate change map.

Deliverable:
A land-cover change analysis module.

## Phase 8: Interactive Dashboard

Tasks:
- Create Streamlit application.
- Create dashboard navigation.
- Integrate geospatial maps.
- Integrate charts.
- Display model evaluation.
- Add file upload.
- Add export functionality.

Deliverable:
A complete interactive GeoAI dashboard.

## Phase 9: Testing and Documentation

Tasks:
- Write unit tests.
- Test invalid inputs.
- Test CRS alignment.
- Test classification outputs.
- Test change detection.
- Write installation documentation.
- Write data acquisition documentation.
- Add screenshots and example outputs.

Deliverable:
A documented, tested, GitHub-ready project.

---

# 10. Testing Requirements

The following tests are mandatory.

| Test | Expected Result |
|---|---|
| Load valid GeoTIFF | Raster loads successfully |
| Load invalid raster | Meaningful error displayed |
| Load valid GeoJSON | Geometry loads successfully |
| Invalid geometry | Validation or repair handled |
| Different CRS | Correct transformation |
| Missing spectral band | Clear validation error |
| NoData pixels | Excluded from training |
| NDVI calculation | Values within valid range |
| Model training | Model saved successfully |
| Model prediction | Valid class values generated |
| GeoTIFF export | CRS and transform preserved |
| Area calculation | Correct area units |
| Change detection | Correct pixel transitions |
| Invalid date comparison | Clear error message |

Use synthetic raster fixtures for unit testing so that tests do not require downloading large satellite datasets.

---

# 11. Expected Deliverables

At project completion, the repository should contain:

1. Complete Python source code.
2. Geospatial preprocessing pipeline.
3. Sentinel-2 image-processing workflow.
4. Spectral index calculations.
5. ESA WorldCover label-processing module.
6. Random Forest training pipeline.
7. Model evaluation reports.
8. Land-cover classification maps.
9. Land-cover area statistics.
10. Change detection module.
11. Interactive Streamlit dashboard.
12. Exportable GeoTIFF and CSV files.
13. Unit tests.
14. Complete README.
15. Architecture documentation.
16. Dataset acquisition instructions.
17. Screenshots demonstrating the application.

---

# 12. Definition of Done

The MVP will be considered complete when:

- [ ] A user can upload an AOI.
- [ ] A user can load Sentinel-2 GeoTIFF imagery.
- [ ] A user can load reference land-cover data.
- [ ] The system can align and preprocess the data.
- [ ] NDVI and NDWI can be calculated.
- [ ] The system can prepare training samples.
- [ ] Random Forest can be trained.
- [ ] Model evaluation metrics are generated.
- [ ] A land-cover classification raster is produced.
- [ ] Land-cover area statistics are calculated.
- [ ] The dashboard displays the results.
- [ ] Two classification maps can be compared.
- [ ] A change-detection raster is generated.
- [ ] Results can be exported.
- [ ] Unit tests pass.
- [ ] Documentation is complete.

---

# 13. Important Development Guidelines for OpenCode

Follow these instructions throughout development.

1. Work phase by phase.
2. Do not generate the entire project in one response.
3. First inspect the existing repository before creating or modifying files.
4. Maintain a clean modular architecture.
5. Use actual geospatial libraries rather than approximating GIS functionality.
6. Preserve raster metadata, coordinate systems, and geospatial transformations.
7. Do not use random or fabricated satellite data for final model results.
8. Do not report fabricated model accuracy.
9. Never claim that classification results represent verified real-world land-cover changes without appropriate validation.
10. Avoid loading large rasters entirely into RAM.
11. Add meaningful logging and error handling.
12. Write unit tests alongside each module.
13. Keep dependencies compatible with the selected Python version.
14. Use configuration files for class mappings and model parameters.
15. Keep large datasets and generated outputs out of Git.
16. Maintain project documentation as features are added.
17. Run tests after each development phase.
18. Do not proceed to the next phase if the current phase has unresolved critical errors.
19. Do not add unnecessary microservices, databases, authentication, or Kubernetes infrastructure to the initial version.
20. Prioritize scientific correctness, reproducibility, and understandable code.

---

# 14. Future Enhancements

After the MVP is stable, consider:

- Automatic Sentinel-2 image acquisition.
- Cloud-free image compositing.
- Sentinel-1 SAR integration.
- More advanced classification algorithms.
- Deep-learning-based semantic segmentation.
- Time-series land-cover monitoring.
- Urban expansion analysis.
- Vegetation change analysis.
- Drought and water-body monitoring.
- PDF report generation.
- Model comparison dashboard.
- Docker deployment.
- REST API for classification services.

These features are optional and must not delay the first working version.

---

# 15. Final Project Goal

Build a complete, reproducible, and interactive GeoAI application that demonstrates practical knowledge of:

- GIS Fundamentals
- Raster and Vector Data
- QGIS-compatible geospatial datasets
- Python Geospatial Programming
- Rasterio
- GeoPandas
- NumPy and Pandas
- Remote Sensing
- Satellite Image Processing
- Machine Learning
- Scikit-learn
- Random Forest Classification
- Environmental Data Analysis
- Land-Cover Change Detection
- Data Visualization

The final application should be suitable for demonstration in a technical interview, presentation on GitHub, and inclusion in a Software Developer / GeoAI / GIS-oriented resume.

**Primary development priority:** Deliver a scientifically sound, working land-cover classification pipeline first. Build the dashboard and advanced change-detection features on top of that reliable foundation.
