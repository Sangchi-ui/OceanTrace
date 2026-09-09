import json
import os
import shutil
import tempfile

import yaml
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.schemas.spill_event import SpillEvent
from inference.predict import run_inference

router = APIRouter()


@router.get("/health", summary="Health Check")
def health_check():
    return {
        "status": "healthy",
        "agent": "Agent 1 - Deep Learning SAR Oil Spill Detector",
        "service": "OceanTrace API",
        "version": "1.0.0"
    }


@router.get("/model-info", summary="Model Metadata & Architecture")
def model_info(config_path: str = "config.yaml"):
    if not os.path.exists(config_path):
        raise HTTPException(status_code=500, detail="Config file missing")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    checkpoint_exists = os.path.exists(cfg["training"].get(
        "best_model_path", "models/unet_oil_spill.pth"))

    return {
        "architecture": cfg["model"].get("architecture", "unet"),
        "input_channels": cfg["model"].get("input_channels", 1),
        "num_classes": cfg["model"].get("num_classes", 1),
        "checkpoint_path": cfg["training"].get("best_model_path"),
        "checkpoint_loaded": checkpoint_exists,
        "default_threshold": cfg["postprocessing"].get("probability_threshold", 0.5),
        "preprocessing": cfg.get("preprocessing")
    }


@router.post("/detect", response_model=SpillEvent, summary="Perform Oil Spill Detection on SAR Image")
async def detect_oil_spill(
    file: UploadFile = File(...),
    threshold: float | None = Query(
        0.5, ge=0.0, le=1.0, description="Segmentation probability threshold"),
    config_path: str = Query(
        "config.yaml", description="Path to configuration file")
):
    """
    Takes a Sentinel-1 SAR raster image file (GeoTIFF, TIFF, PNG, JPEG),
    runs the full Agent 1 deep learning detection pipeline,
    and returns a structured SpillEvent JSON payload.
    """
    # Create temporary file to store uploaded raster
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run full Agent 1 inference pipeline
        spill_event = run_inference(
            input_image_path=temp_file_path,
            threshold=threshold,
            config_path=config_path,
            output_dir="outputs"
        )
        return spill_event
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Inference failed: {e!s}")
    finally:
        # Cleanup temp upload directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("/results/{event_id}", summary="Get Stored Spill Event Output")
def get_spill_event_result(event_id: str, output_dir: str = "outputs"):
    event_json_path = os.path.join(output_dir, "spill_event.json")
    if not os.path.exists(event_json_path):
        raise HTTPException(
            status_code=404, detail="No spill event results found.")

    with open(event_json_path, "r") as f:
        data = json.load(f)

    if data.get("event_id") == event_id or event_id == "latest":
        return data

    raise HTTPException(
        status_code=404, detail=f"Event ID '{event_id}' not found.")
