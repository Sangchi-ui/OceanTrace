# OceanTrace — Agent 1: Sentinel-1 SAR Oil Spill Detection Pipeline

Agent 1 is the remote-sensing earth observation component of OceanTrace. It processes satellite Synthetic Aperture Radar (SAR) imagery to detect marine oil spills, segment slick contours, extract geographic and geometric characteristics, and generate structured validation records for downstream analysis by **Agent 2 (Hindcasting & Forecasting)**.

---

## Architecture & Structure

```text
agent1/
├── __init__.py              # Package entry point
├── predict.py               # Main CLI and inference pipeline
├── config.yaml              # Detection and model thresholds
├── api/                     # Dedicated Agent 1 REST API
│   ├── main.py              # FastAPI application
│   └── routes.py            # /detect and /results routes
├── models/                  # UNet segmentation models
│   ├── unet.py
│   └── smp_unet.py
├── preprocessing/           # SAR radiometric normalization
│   ├── sar_preprocessor.py
│   └── eo_preprocessor.py
├── postprocessing/          # Connected components & morphology
│   └── mask_processor.py
├── geospatial/              # Vector polygon extraction
│   └── geometry.py
├── schemas/                 # Pydantic data schemas
│   └── spill_event.py
├── validation/              # QC validation
│   └── validator.py
└── visualization/           # Slick visualizer
    └── visualize.py
```

---

## Running Agent 1

### CLI Inference
```bash
python -m agent1.predict --input data/sample/sample_sentinel1.tif --output_dir outputs/
```

### Python API
```python
from agent1.predict import run_inference

spill_event = run_inference(
    input_image_path="data/sample/sample_sentinel1.tif",
    threshold=0.5,
    output_dir="outputs"
)
print(f"Event ID: {spill_event.event_id}, Slicks: {spill_event.num_spill_regions}")
```

### Dedicated API Service
```bash
uvicorn agent1.api.main:app --port 8000 --reload
```
