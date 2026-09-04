# OceanTrace — Agent 1: Sentinel-1 SAR Oil Spill Detection Pipeline

Agent 1 is a computational deep-learning and remote sensing pipeline designed to detect potential marine oil spills in Sentinel-1 Synthetic Aperture Radar (SAR) imagery, characterize their geometric and physical properties, and produce validated data contracts (`SpillEvent` JSON + GeoJSON).

The structured output of Agent 1 is explicitly designed to serve as the upstream input for **Agent 2 (Lagrangian Particle Drift Agent)**.

---

## 1. System Architecture

```
                       ┌───────────────────────────────┐
                       │   Sentinel-1 SAR Raster       │
                       │  (GeoTIFF / TIFF / PNG / JPG) │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │       SAR Preprocessing       │
                       │ (No-data, dB, Lee Speckle,    │
                       │     Min-Max Normalization)    │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ PyTorch U-Net Model Inference │
                       │    (Pixel Probability Map)    │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │        Post-Processing        │
                       │ (Thresholding, Morphology,    │
                       │ Connected Component Analysis) │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │  Geospatial Characterization  │
                       │(Polygons, Area km², Centroid, │
                       │    Perimeter, GeoJSON Export) │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │   Automated QC Validation     │
                       │  (VALID / WARNING / INVALID)  │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │    Outputs & API Service      │
                       │ (SpillEvent JSON, GeoJSON,    │
                       │  5-Stage Visual Overlay, API) │
                       └───────────────────────────────┘
```

---

## 2. Technical Stack

* **Language**: Python 3.10+
* **Deep Learning Framework**: PyTorch 2.0+, torchvision
* **Geospatial & Vector**: Rasterio, GeoPandas, PyPROJ, Shapely
* **Image Processing & Morphology**: OpenCV, scikit-image, NumPy, SciPy
* **API Service**: FastAPI, Uvicorn, Pydantic v2
* **Visualization**: Matplotlib
* **Testing**: PyTest

---

## 3. Data Requirements & SAR Input Representation

### SAR Data Characteristics
Synthetic Aperture Radar (SAR) measures surface roughness. Ocean oil slicks dampen capillary and small gravity waves, reducing radar backscatter. Consequently, potential oil slicks manifest as **dark patches** (low radar intensity) against the brighter surrounding ocean surface.

### Supported Formats
* **GeoTIFF (`.tif`, `.tiff`)**: Full support for CRS, affine transformation matrices, and geographic measurements ($km^2$, $km$, Lat/Lon).
* **PNG / JPEG (`.png`, `.jpg`)**: Supported for fast prototyping/testing. Measurements are automatically fallback-marked as **pixel-space** without inventing fictitious geographic coordinates.

---

## 4. Project Directory Structure

```
OceanTrace/
├── app/
│   ├── api/
│   │   ├── main.py          # FastAPI application entrypoint
│   │   └── routes.py        # API endpoints (/detect, /health, /model-info)
│   ├── geospatial/
│   │   └── geometry.py      # Polygonization, Area (km²), Centroid, GeoJSON
│   ├── models/
│   │   └── unet.py          # PyTorch U-Net architecture & predict interface
│   ├── postprocessing/
│   │   └── mask_processor.py# Thresholding, Morphology, Connected Components
│   ├── preprocessing/
│   │   └── sar_preprocessor.py # dB conversion, Lee speckle filter, Normalization
│   ├── schemas/
│   │   └── spill_event.py   # Pydantic SpillEvent data contract
│   ├── validation/
│   │   └── validator.py     # Automated Quality Control validation layer
│   └── visualization/
│       └── visualize.py     # 5-Stage comparative visualization generator
├── data/
│   ├── raw/                 # Raw training/val/test images & masks
│   └── sample/              # Sample images for testing
├── inference/
│   └── predict.py           # Main CLI script for oil spill inference
├── models/
│   └── unet_oil_spill.pth   # Trained PyTorch U-Net weights checkpoint
├── outputs/                 # Output artifacts (maps, overlays, JSON, GeoJSON)
├── training/
│   ├── dataset.py           # SARDataset PyTorch dataset loader
│   ├── evaluate.py          # Test set evaluation script (IoU, Dice, Precision, Recall)
│   ├── generate_synthetic_data.py # Benchmark SAR dataset generator
│   ├── losses.py            # Combined BCE + Dice loss function
│   └── train.py             # Model training script
├── tests/                   # PyTest test suite
├── config.yaml              # Central pipeline configuration
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 5. Installation & Virtual Environment Setup

### 1. Create and Activate Virtual Environment
```bash
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 6. Dataset Setup & Model Training

### 1. Generate Synthetic Benchmark SAR Dataset (Optional / First Run)
If you wish to test or train immediately before adding custom Sentinel-1 data:
```bash
python training/generate_synthetic_data.py
```

### 2. Train U-Net Model
Train the PyTorch U-Net model (automatically uses CUDA GPU if available, else CPU):
```bash
python training/train.py --config config.yaml
```
Model checkpoints will be saved to `models/unet_oil_spill.pth`.

### 3. Evaluate Model Performance
Compute IoU, Dice coefficient, Precision, Recall, and F1 score:
```bash
python training/evaluate.py --config config.yaml --output outputs/metrics.json
```

---

## 7. Running Inference (CLI)

Run full oil-spill detection on any Sentinel-1 SAR raster image:

```bash
python inference/predict.py --input data/sample/sample_sentinel1.tif --threshold 0.5
```

### Output Artifacts Generated in `outputs/`:
1. `outputs/probability_map.png`: Pixel-level UNet probability heatmap.
2. `outputs/segmentation_mask.png`: Binary mask after thresholding & morphological cleaning.
3. `outputs/spill_overlay.png`: High-resolution 5-panel pipeline comparison plot.
4. `outputs/spill_event.json`: Structured `SpillEvent` JSON data contract.
5. `outputs/spill.geojson`: Standard GeoJSON FeatureCollection containing slick polygons and metadata.

---

## 8. FastAPI Web Service

### Start the API Server:
```bash
python app/api/main.py
```
Or with Uvicorn directly:
```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Interactive API Documentation:
Open your browser and navigate to: `http://localhost:8000/docs`

### Key Endpoints:
* `POST /api/v1/detect`: Upload a SAR image (`.tif`, `.png`) to receive the structured `SpillEvent` JSON response.
* `GET /api/v1/health`: API status and capabilities check.
* `GET /api/v1/model-info`: Inspect loaded U-Net model parameters and checkpoint status.

---

## 9. Scientific Limitations & Disclaimers

1. **Look-Alikes**: Low wind speeds (< 2-3 m/s), biogenic slicks, grease ice, and rain cells produce dark SAR backscatter regions that mimic oil spills ("look-alikes"). Detections are designated as **"Potential Oil Spill Candidates"** rather than definitive pollution events.
2. **Confidence Score**: The reported confidence score reflects the deep learning model's spatial pixel probability agreement over the slick, not absolute physical certainty of oil composition.
3. **Geographic Measurement Validity**: Area in $km^2$ and geographic centroids are computed strictly when valid CRS and affine transformation matrices are present in the input raster metadata.

---

## 10. Future Integration with Agent 2 (Lagrangian Drift Model)

The `SpillEvent` output schema provides all required parameters for Agent 2:
* **`centroid.latitude` & `centroid.longitude`**: Initial particle release seed coordinate.
* **`polygon_geojson`**: Initial spatial distribution boundary of the slick.
* **`acquisition_time`**: Temporal origin for drift integration.
* **`area_km2`**: Estimated initial slick extent.
* **`overall_confidence`**: Weighting factor for drift scenario simulation.
#   O c e a n T r a c e  
 