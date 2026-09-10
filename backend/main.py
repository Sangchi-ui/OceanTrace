import base64
import io
import os
import sys
import time
import re
from datetime import datetime

import numpy as np
from scipy.ndimage import binary_dilation
import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Add root directory to python path so we can import from existing OceanTrace modules
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from app.models.unet import get_segmentation_model
from app.postprocessing.mask_processor import MaskProcessor
from app.preprocessing.sar_preprocessor import SARPreprocessor

app = FastAPI(title="OceanTrace SAR Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for loaded model and config
model_state = {
    "model": None,
    "device": None,
    "config": None,
    "preprocessor": None,
    "mask_processor": None
}

@app.on_event("startup")
async def load_model():
    """Load configuration and PyTorch model into memory on startup."""
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    if not os.path.exists(config_path):
        raise RuntimeError(f"Config not found at {config_path}")
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    model_state["config"] = cfg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state["device"] = device
    
    # Preprocessor initialization (matching training)
    target_size = tuple(cfg["preprocessing"].get("target_size", [256, 256]))
    model_state["preprocessor"] = SARPreprocessor(
        target_size=target_size,
        normalize=cfg["preprocessing"].get("normalize", True)
    )
    
    # Mask Processor
    th = cfg["postprocessing"].get("probability_threshold", 0.63)
    model_state["mask_processor"] = MaskProcessor(
        probability_threshold=th,
        morphology_kernel_size=cfg["postprocessing"].get("morphology_kernel_size", 3),
        min_region_pixels=cfg["postprocessing"].get("min_region_pixels", 50)
    )

    # Load Model
    model_path = os.path.join(ROOT_DIR, cfg["training"].get("best_model_path", "models/unet_oil_spill.pth"))
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model checkpoint missing at {model_path}")

    model = get_segmentation_model(
        architecture=cfg["model"].get("architecture", "smp_unet"),
        in_channels=cfg["model"].get("input_channels", 2),
        num_classes=cfg["model"].get("num_classes", 1),
        features=cfg["model"].get("features", [32, 64, 128, 256]),
        classification_head=cfg["model"].get("classification_head", False)
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    model_state["model"] = model
    print(f"Model loaded successfully on {device}. Threshold: {th}")

def extract_acquisition_time(filename: str, filepath: str) -> str:
    """Extract SAR acquisition start time from filename or metadata."""
    filename = filename or ""
    # 1. Filename extraction (Primary)
    # Match pattern: YYYYMMDDTHHMMSS
    match = re.search(r"(\d{8}T\d{6})", filename)
    if match:
        try:
            dt = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except ValueError:
            pass

    # 2. Metadata fallback
    if filepath and filepath.lower().endswith(('.tif', '.tiff', '.gtiff')):
        try:
            import rasterio
            with rasterio.open(filepath) as src:
                # Common tag format in TIFF: "YYYY:MM:DD HH:MM:SS" or "YYYY-MM-DD HH:MM:SS"
                tags = src.tags()
                time_str = tags.get("TIFFTAG_DATETIME") or tags.get("DATETIME") or tags.get("ACQUISITION_DATETIME")
                if time_str:
                    time_clean = time_str.strip()[:19]
                    for fmt in (
                        "%Y:%m:%d %H:%M:%S",
                        "%Y-%m-%d %H:%M:%S",
                        "%Y-%m-%dT%H:%M:%S",
                        "%Y%m%dT%H%M%S",
                    ):
                        try:
                            dt = datetime.strptime(time_clean, fmt)
                            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                        except ValueError:
                            continue
        except Exception:
            pass
            
    # 3. Graceful fallback
    return "Acquisition time unavailable"

def encode_image_base64(img_array: np.ndarray) -> str:
    """Convert numpy array (H, W, C) or (H, W) to base64 PNG."""
    if img_array.dtype != np.uint8:
        img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    
    img = Image.fromarray(img_array)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_overlay(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create a semi-transparent red overlay with a solid 1px border."""
    if original.ndim == 2:
        overlay = np.stack((original,) * 3, axis=-1)
    elif original.ndim == 3 and original.shape[0] in [1, 2]:
        overlay = np.stack((original[0],) * 3, axis=-1)
    else:
        overlay = original.copy()
        
    overlay = np.clip(overlay, 0, 1).astype(np.float32)
    
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return overlay
        
    # Dilate by 1 pixel to get the thickened border
    struct_elem = np.ones((3, 3), dtype=bool)
    dilated_mask = binary_dilation(mask_bool, structure=struct_elem)
    
    # Border is dilated mask XOR original mask
    border_mask = dilated_mask ^ mask_bool
    
    # 1. Semi-transparent interior (20% red, 80% original)
    overlay[mask_bool, 0] = overlay[mask_bool, 0] * 0.8 + 0.2
    overlay[mask_bool, 1] = overlay[mask_bool, 1] * 0.8
    overlay[mask_bool, 2] = overlay[mask_bool, 2] * 0.8
    
    # 2. Solid 1px border (100% red)
    overlay[border_mask] = [1.0, 0.0, 0.0]
    
    return overlay

def generate_heatmap(prob_map: np.ndarray) -> np.ndarray:
    """Create a simple inferno-like heatmap for probabilities."""
    heatmap = np.zeros((*prob_map.shape, 3), dtype=np.float32)
    # Simple colormap: low=black, mid=purple/red, high=yellow
    heatmap[:, :, 0] = prob_map  # Red
    heatmap[:, :, 1] = np.clip(prob_map - 0.5, 0, 0.5) * 2  # Green
    heatmap[:, :, 2] = np.clip(0.5 - prob_map, 0, 0.5) * 2  # Blue
    return heatmap

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model_state["model"] is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Use TIF, PNG, or JPG.")
    
    start_time = time.time()
    
    # Save uploaded file temporarily
    import uuid
    temp_path = f"temp_{uuid.uuid4().hex}_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
        
    try:
        # Preprocessing (matches training exactly)
        preprocessor = model_state["preprocessor"]
        original_img, input_tensor, meta = preprocessor.preprocess_sar(temp_path)
        
        # Enforce channels (handling 1 channel synthetic vs 2 channel real)
        in_channels = model_state["config"]["model"].get("input_channels", 2)
        if input_tensor.shape[1] == 1 and in_channels > 1:
            input_tensor = input_tensor.repeat(1, in_channels, 1, 1)
        elif input_tensor.shape[1] > in_channels:
            input_tensor = input_tensor[:, :in_channels, :, :]

        # Inference
        device = model_state["device"]
        with torch.no_grad():
            prob_tensor = model_state["model"].predict(input_tensor.to(device))
        prob_map = prob_tensor.squeeze().cpu().numpy()
        
        # Postprocessing
        threshold = model_state["config"]["postprocessing"].get("probability_threshold", 0.63)
        mask_processor = model_state["mask_processor"]
        _, cleaned_mask, _ = mask_processor.process(prob_map, threshold=threshold)
        
        # Original Image to display (take first channel)
        if original_img.ndim == 3:
            display_img = original_img[0]
        else:
            display_img = original_img

        # Resize masks back to original if needed
        h, w = display_img.shape
        if prob_map.shape != (h, w):
            pil_prob = Image.fromarray((prob_map * 255).astype(np.uint8)).resize((w, h), resample=Image.BILINEAR)
            prob_map = np.array(pil_prob, dtype=np.float32) / 255.0
            
            pil_cleaned = Image.fromarray((cleaned_mask * 255).astype(np.uint8)).resize((w, h), resample=Image.NEAREST)
            cleaned_mask = (np.array(pil_cleaned, dtype=np.uint8) > 127).astype(np.uint8)

        # Generate Visuals
        b64_mask = encode_image_base64(cleaned_mask)
        b64_heatmap = encode_image_base64(generate_heatmap(prob_map))
        b64_overlay = encode_image_base64(generate_overlay(display_img, cleaned_mask))
        
        # Calculate Metrics
        total_pixels = cleaned_mask.size
        oil_pixels = int(cleaned_mask.sum())
        coverage_pct = (oil_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        positive_probs = prob_map[cleaned_mask > 0]
        mean_conf = float(np.mean(positive_probs)) if len(positive_probs) > 0 else 0.0

        inference_time_ms = int((time.time() - start_time) * 1000)
        
        acquisition_time = extract_acquisition_time(file.filename, temp_path)

        return {
            "success": True,
            "images": {
                "mask": f"data:image/png;base64,{b64_mask}",
                "heatmap": f"data:image/png;base64,{b64_heatmap}",
                "overlay": f"data:image/png;base64,{b64_overlay}"
            },
            "metrics": {
                "oil_coverage_percent": round(coverage_pct, 4),
                "oil_pixels": oil_pixels,
                "background_pixels": total_pixels - oil_pixels,
                "mean_confidence": round(mean_conf, 4),
                "inference_time_ms": inference_time_ms
            },
            "metadata": {
                "model_name": model_state["config"]["model"].get("architecture", "unet"),
                "threshold": threshold,
                "checkpoint": os.path.basename(model_state["config"]["training"].get("best_model_path", "")),
                "acquisition_time": acquisition_time
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Import and include the v2 router
from backend.api_v2 import router as v2_router
app.include_router(v2_router, prefix="/api/v2")
