# 🌊 OceanTrace — Agent 1: Sentinel-1 SAR Oil Spill Detection

> **An AI-powered Sentinel-1 SAR pipeline for detecting, characterizing, validating, and geospatially representing potential marine oil spills.**

OceanTrace **Agent 1** is a computational deep-learning and remote-sensing pipeline designed to detect **potential marine oil spills** from **Sentinel-1 Synthetic Aperture Radar (SAR)** imagery.

The pipeline performs the complete workflow from SAR preprocessing and U-Net segmentation to geospatial characterization, quality validation, and structured output generation.

The resulting `SpillEvent` data contract is explicitly designed to serve as the upstream input for:

> 🛰️ **Agent 2 — Lagrangian Particle Drift Agent**

---

## ✨ Key Capabilities

* 🛰️ Sentinel-1 SAR image preprocessing
* 🧹 No-data handling and normalization
* 📡 SAR backscatter preprocessing
* 🔇 Lee speckle filtering
* 🧠 PyTorch U-Net semantic segmentation
* 🎯 Pixel-level oil-spill probability estimation
* 🧩 Morphological post-processing
* 🔗 Connected-component analysis
* 🌍 Geospatial polygon generation
* 📐 Slick area and perimeter calculation
* 📍 Centroid extraction
* 🗺️ GeoJSON generation
* ✅ Automated quality-control validation
* 📊 Five-stage visual pipeline visualization
* 📦 Structured `SpillEvent` JSON data contract
* 🚀 FastAPI inference service
* 🔌 Agent 2 integration readiness

---

# 🏗️ System Architecture

```text
                         ┌───────────────────────────────┐
                         │      Sentinel-1 SAR Raster    │
                         │   GeoTIFF / TIFF / PNG / JPG  │
                         └───────────────┬───────────────┘
                                         │
                                         ▼
                         ┌───────────────────────────────┐
                         │       SAR Preprocessing       │
                         │                               │
                         │  • No-data handling           │
                         │  • dB conversion              │
                         │  • Lee speckle filtering      │
                         │  • Min-Max normalization      │
                         └───────────────┬───────────────┘
                                         │
                                         ▼
                         ┌───────────────────────────────┐
                         │     PyTorch U-Net Inference   │
                         │                               │
                         │   Pixel-level probability     │
                         │          prediction           │
                         └───────────────┬───────────────┘
                                         │
                                         ▼
                         ┌───────────────────────────────┐
                         │        Post-Processing        │
                         │                               │
                         │  • Probability thresholding   │
                         │  • Morphological operations   │
                         │  • Connected components       │
                         └───────────────┬───────────────┘
                                         │
                                         ▼
                         ┌───────────────────────────────┐
                         │    Geospatial Characterization│
                         │                               │
                         │  • Polygonization             │
                         │  • Area (km²)                 │
                         │  • Centroid                   │
                         │  • Perimeter                  │
                         │  • GeoJSON export             │
                         └───────────────┬───────────────┘
                                         │
                                         ▼
                         ┌───────────────────────────────┐
                         │       Automated QC Layer      │
                         │                               │
                         │   VALID / WARNING / INVALID   │
                         └───────────────┬───────────────┘
                                         │
                                         ▼
                         ┌───────────────────────────────┐
                         │      Outputs & API Service    │
                         │                               │
                         │  • SpillEvent JSON            │
                         │  • GeoJSON                     │
                         │  • Probability maps            │
                         │  • Segmentation masks          │
                         │  • Visual overlays             │
                         │  • FastAPI service             │
                         └───────────────────────────────┘
```

---

# 🧰 Technical Stack

| Category            | Technology                |
| ------------------- | ------------------------- |
| Language            | Python 3.10+              |
| Deep Learning       | PyTorch 2.0+, torchvision |
| Geospatial Raster   | Rasterio                  |
| Geospatial Vector   | GeoPandas, Shapely        |
| Coordinate Systems  | PyProj                    |
| Image Processing    | OpenCV, scikit-image      |
| Numerical Computing | NumPy, SciPy              |
| API                 | FastAPI, Uvicorn          |
| Data Validation     | Pydantic v2               |
| Visualization       | Matplotlib                |
| Testing             | PyTest                    |

---

# 🛰️ SAR Data & Input Representation

## What is SAR?

**Synthetic Aperture Radar (SAR)** measures the radar backscatter from the Earth's surface.

Marine oil slicks can dampen capillary and small gravity waves on the ocean surface. This can reduce radar backscatter and cause oil slicks to appear as:

```text
Normal Ocean
████████████████████████
████████████████████████
████████████████████████

        ↓ Oil Slick

████████████████████████
██████░░░░░░░░░█████████
█████░░░░░░░░░░░████████
██████░░░░░░░░░█████████
████████████████████████

░ = Low-backscatter / dark region
```

Therefore, potential oil slicks may appear as **dark regions** surrounded by brighter ocean backscatter.

> ⚠️ Dark SAR regions are not necessarily oil spills. Several natural phenomena can produce similar signatures.

---

# 📁 Supported Input Formats

### GeoTIFF

**`.tif` / `.tiff`**

Full geospatial support, including:

* CRS
* Affine transformation
* Geographic coordinates
* Area measurements in km²
* Geographic centroids
* Perimeter measurements
* GeoJSON export

### PNG / JPEG

**`.png` / `.jpg`**

Supported primarily for:

* Prototyping
* Development
* Testing
* Visualization

For non-georeferenced images, geographic measurements are automatically treated as **pixel-space measurements** rather than inventing geographic coordinates.

---

# 📂 Project Structure

```text
OceanTrace/
│
├── app/
│   ├── api/
│   │   ├── main.py
│   │   └── routes.py
│   │
│   ├── geospatial/
│   │   └── geometry.py
│   │
│   ├── models/
│   │   └── unet.py
│   │
│   ├── postprocessing/
│   │   └── mask_processor.py
│   │
│   ├── preprocessing/
│   │   └── sar_preprocessor.py
│   │
│   ├── schemas/
│   │   └── spill_event.py
│   │
│   ├── validation/
│   │   └── validator.py
│   │
│   └── visualization/
│       └── visualize.py
│
├── data/
│   ├── raw/
│   └── sample/
│
├── inference/
│   └── predict.py
│
├── models/
│   └── unet_oil_spill.pth
│
├── outputs/
│
├── training/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── generate_synthetic_data.py
│   ├── losses.py
│   └── train.py
│
├── tests/
│
├── config.yaml
├── requirements.txt
└── README.md
```

### Directory Responsibilities

| Directory             | Purpose                                           |
| --------------------- | ------------------------------------------------- |
| `app/api/`            | FastAPI application and REST endpoints            |
| `app/geospatial/`     | Polygonization and spatial measurements           |
| `app/models/`         | U-Net architecture and inference interface        |
| `app/postprocessing/` | Segmentation mask cleanup                         |
| `app/preprocessing/`  | SAR preprocessing pipeline                        |
| `app/schemas/`        | Pydantic data contracts                           |
| `app/validation/`     | Automated quality-control validation              |
| `app/visualization/`  | Pipeline visualization generation                 |
| `data/raw/`           | Training, validation, and test datasets           |
| `data/sample/`        | Sample inference data                             |
| `inference/`          | CLI inference pipeline                            |
| `models/`             | Trained model checkpoints                         |
| `outputs/`            | Generated prediction artifacts                    |
| `training/`           | Dataset, training, evaluation, and loss functions |
| `tests/`              | Automated test suite                              |

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd OceanTrace
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows — PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# 🧠 Dataset & Model Training

## Generate Synthetic Benchmark Dataset

For initial testing and development, a synthetic SAR benchmark dataset can be generated:

```bash
python training/generate_synthetic_data.py
```

This step is optional when using an existing Sentinel-1 dataset.

---

## Train the U-Net Model

Train the PyTorch U-Net model using the central configuration:

```bash
python training/train.py --config config.yaml
```

The training pipeline automatically uses:

```text
CUDA GPU → if available
   │
   └── otherwise
        ↓
       CPU
```

The trained checkpoint is saved to:

```text
models/unet_oil_spill.pth
```

---

# 📊 Model Evaluation

Evaluate the trained model on the test dataset:

```bash
python training/evaluate.py \
    --config config.yaml \
    --output outputs/metrics.json
```

The evaluation pipeline reports:

* IoU
* Dice coefficient
* Precision
* Recall
* F1 Score

Example output structure:

```json
{
  "iou": 0.00,
  "dice": 0.00,
  "precision": 0.00,
  "recall": 0.00,
  "f1": 0.00
}
```

> The actual values depend on the dataset and trained model.

---

# 🔍 Running Inference

Run the complete oil-spill detection pipeline:

```bash
python inference/predict.py \
    --input data/sample/sample_sentinel1.tif \
    --threshold 0.5
```

### Pipeline

```text
SAR Image
   │
   ▼
Preprocessing
   │
   ▼
U-Net
   │
   ▼
Probability Map
   │
   ▼
Thresholding
   │
   ▼
Morphological Cleaning
   │
   ▼
Connected Components
   │
   ▼
Geospatial Characterization
   │
   ▼
Quality Validation
   │
   ▼
SpillEvent + GeoJSON
```

---

# 📦 Output Artifacts

The inference pipeline generates artifacts inside `outputs/`.

| Output                  | Description                                       |
| ----------------------- | ------------------------------------------------- |
| `probability_map.png`   | Pixel-level U-Net probability heatmap             |
| `segmentation_mask.png` | Binary segmentation mask                          |
| `spill_overlay.png`     | Five-stage comparative visualization              |
| `spill_event.json`      | Structured `SpillEvent` data contract             |
| `spill.geojson`         | GeoJSON representation of detected slick polygons |

---

# 🚀 FastAPI Service

OceanTrace can also be exposed as a REST API.

## Start the API

```bash
python app/api/main.py
```

Or using Uvicorn:

```bash
uvicorn app.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload
```

The API will be available at:

```text
http://localhost:8000
```

### Interactive API Documentation

FastAPI automatically provides Swagger documentation at:

```text
http://localhost:8000/docs
```

---

# 🔌 API Endpoints

| Method | Endpoint             | Description                                  |
| ------ | -------------------- | -------------------------------------------- |
| `POST` | `/api/v1/detect`     | Upload SAR image and run oil-spill detection |
| `GET`  | `/api/v1/health`     | Check API health and capabilities            |
| `GET`  | `/api/v1/model-info` | Inspect model and checkpoint information     |

### Detection Request

```text
POST /api/v1/detect
        │
        │ SAR Image
        ▼
┌───────────────────┐
│ OceanTrace Agent 1│
└─────────┬─────────┘
          │
          ▼
   SpillEvent JSON
```

---

# 📄 SpillEvent Data Contract

The output of Agent 1 is structured around a `SpillEvent` schema.

Conceptually:

```json
{
  "centroid": {
    "latitude": 0.0,
    "longitude": 0.0
  },
  "polygon_geojson": {},
  "acquisition_time": "YYYY-MM-DDTHH:MM:SS",
  "area_km2": 0.0,
  "overall_confidence": 0.0
}
```

This structured contract provides a clean interface between the computer-vision pipeline and downstream ocean-drift modeling.

---

# 🧪 Automated Quality Control

OceanTrace includes an automated validation layer that classifies detections into:

```text
                 ┌───────────────┐
                 │  SpillEvent   │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │  QC Validator │
                 └───────┬───────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
           VALID      WARNING    INVALID
```

The QC layer helps identify suspicious or incomplete detections before they are passed to downstream agents.

---

# ⚠️ Scientific Limitations

## 1. SAR Look-Alikes

Several non-oil phenomena can produce dark SAR signatures similar to oil slicks, including:

* Low wind speeds
* Biogenic slicks
* Grease ice
* Rain cells
* Other ocean-surface phenomena

Low wind speeds, particularly around **2–3 m/s or lower**, can significantly complicate interpretation.

Therefore:

> **OceanTrace detections should be interpreted as "Potential Oil Spill Candidates", not definitive pollution events.**

---

## 2. Confidence Score

The reported confidence score represents the agreement of the deep-learning model's spatial pixel probabilities over the detected slick.

It should **not** be interpreted as absolute physical certainty that the detected region contains oil.

---

## 3. Geographic Measurement Validity

Geographic measurements are generated only when valid geospatial metadata is available.

For GeoTIFF inputs:

```text
Valid CRS
   +
Valid Affine Transform
   ↓
Geographic measurements
   ↓
Area / Perimeter / Centroid
```

For non-georeferenced PNG/JPEG inputs:

```text
No CRS
   +
No Affine Transform
   ↓
Pixel-space measurements
```

The system does **not** invent geographic coordinates when the source image does not contain sufficient geospatial metadata.

---

# 🤖 Multi-Agent System Integration

OceanTrace Agent 1 is the first stage of a larger multi-agent marine oil-spill intelligence system.

```text
┌─────────────────────────────┐
│        AGENT 1              │
│  SAR Oil Spill Detection    │
│                             │
│  Sentinel-1 → SpillEvent    │
└──────────────┬──────────────┘
               │
               │ SpillEvent
               ▼
┌─────────────────────────────┐
│        AGENT 2              │
│ Lagrangian Particle Drift   │
│                             │
│  SpillEvent → Drift Paths   │
└──────────────┬──────────────┘
               │
               │ Particle trajectories
               ▼
┌─────────────────────────────┐
│        AGENT 3              │
│      Vessel Attribution     │
│                             │
│  Drift + Vessel Data        │
│          ↓                  │
│   Possible Source Vessel    │
└─────────────────────────────┘
```

---

# 🌊 Agent 2 Integration

The `SpillEvent` output provides the required parameters for the **Lagrangian Particle Drift Agent**.

| Agent 1 Field        | Agent 2 Usage                                 |
| -------------------- | --------------------------------------------- |
| `centroid.latitude`  | Initial particle latitude                     |
| `centroid.longitude` | Initial particle longitude                    |
| `polygon_geojson`    | Initial spatial slick boundary                |
| `acquisition_time`   | Temporal starting point for drift integration |
| `area_km2`           | Estimated initial slick extent                |
| `overall_confidence` | Confidence weighting for drift scenarios      |

This allows Agent 2 to consume Agent 1's output without requiring direct knowledge of the internal segmentation pipeline.

---

# 🧩 Design Philosophy

OceanTrace follows a modular pipeline architecture:

```text
┌────────────┐
│ Input Data │
└─────┬──────┘
      ▼
┌────────────┐
│ Processing │
└─────┬──────┘
      ▼
┌────────────┐
│ ML Model   │
└─────┬──────┘
      ▼
┌────────────┐
│ Postprocess│
└─────┬──────┘
      ▼
┌────────────┐
│ Geospatial │
└─────┬──────┘
      ▼
┌────────────┐
│ Validation │
└─────┬──────┘
      ▼
┌────────────┐
│ Data       │
│ Contract   │
└─────┬──────┘
      ▼
┌────────────┐
│ Agent 2    │
└────────────┘
```

This separation makes individual components easier to:

* Test
* Replace
* Optimize
* Retrain
* Deploy
* Integrate with downstream agents

---

# 🧪 Testing

Run the complete PyTest suite:

```bash
pytest
```

For more detailed output:

```bash
pytest -v
```

---

# 📌 Current Status

| Component                   | Status        |
| --------------------------- | ------------- |
| SAR preprocessing           | ✅ Implemented |
| U-Net architecture          | ✅ Implemented |
| Model training              | ✅ Implemented |
| Model evaluation            | ✅ Implemented |
| Post-processing             | ✅ Implemented |
| Geospatial characterization | ✅ Implemented |
| GeoJSON export              | ✅ Implemented |
| QC validation               | ✅ Implemented |
| Visualization               | ✅ Implemented |
| CLI inference               | ✅ Implemented |
| FastAPI service             | ✅ Implemented |
| Agent 2 interface           | ✅ Defined     |
| Agent 3 integration         | 🔄 Planned    |

---

# 🚧 Future Improvements

Potential future development areas include:

* Multi-source SAR data integration
* Sentinel-1 metadata-aware preprocessing
* Advanced look-alike discrimination
* Improved model architectures
* Temporal SAR analysis
* Multi-temporal change detection
* Uncertainty estimation
* Real-world Sentinel-1 benchmark evaluation
* Model explainability
* Production model monitoring
* Automated inference pipelines
* Agent 2 Lagrangian drift integration
* Agent 3 vessel attribution integration
* End-to-end multi-agent orchestration

---

# 📜 Disclaimer

OceanTrace is a research and engineering system for **potential oil-spill detection from SAR imagery**.

A model prediction should not be treated as definitive evidence of an oil spill without appropriate scientific validation, additional remote-sensing analysis, and/or independent observations.

---

# 🌊 OceanTrace

```text
Sentinel-1 SAR
      │
      ▼
┌─────────────────┐
│   Agent 1       │
│ Spill Detection │
└────────┬────────┘
         │
         ▼
   SpillEvent JSON
         │
         ▼
┌─────────────────┐
│    Agent 2      │
│ Drift Modeling  │
└────────┬────────┘
         │
         ▼
   Particle Paths
         │
         ▼
┌─────────────────┐
│    Agent 3      │
│ Vessel Attribution│
└─────────────────┘
```

> **Detect → Characterize → Validate → Track → Attribute**

**OceanTrace — Building an intelligent pipeline for marine oil-spill monitoring.**
