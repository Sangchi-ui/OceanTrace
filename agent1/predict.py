import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch
import yaml
from PIL import Image

try:
    from agent1.geospatial.geometry import GeospatialProcessor
    from agent1.models.unet import get_segmentation_model
    from agent1.postprocessing.mask_processor import MaskProcessor
    from agent1.preprocessing.sar_preprocessor import SARPreprocessor
    from agent1.schemas.spill_event import (
        Centroid,
        DetectionMetadata,
        GeometryMetadata,
        MeasurementsMetadata,
        SourceMetadata,
        SpillEvent,
        SpillRegionRecord,
        ValidationMetadata,
    )
    from agent1.validation.validator import SpillValidator
    from agent1.visualization.visualize import visualize_pipeline_results
except ImportError:
    from app.geospatial.geometry import GeospatialProcessor
    from app.models.unet import get_segmentation_model
    from app.postprocessing.mask_processor import MaskProcessor
    from app.preprocessing.sar_preprocessor import SARPreprocessor
    from app.schemas.spill_event import (
        Centroid,
        DetectionMetadata,
        GeometryMetadata,
        MeasurementsMetadata,
        SourceMetadata,
        SpillEvent,
        SpillRegionRecord,
        ValidationMetadata,
    )
    from app.validation.validator import SpillValidator
    from app.visualization.visualize import visualize_pipeline_results

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))


def run_inference(
    input_image_path: str,
    threshold: float = 0.5,
    config_path: str = "config.yaml",
    output_dir: str = "outputs"
) -> SpillEvent:
    """
    Complete Agent 1 Inference Pipeline:
    EO Optical Image -> Preprocessing -> Model Inference -> Probability Map -> Threshold & Morphology ->
    Connected Components -> Geometry Extraction -> QC Validation -> SpillEvent JSON + GeoJSON + Visualizations.
    """
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    # Load configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    th = threshold if threshold is not None else cfg["postprocessing"].get(
        "probability_threshold", 0.5)
    model_path = cfg["training"].get(
        "best_model_path", "models/unet_oil_spill.pth")
    target_size = tuple(cfg["preprocessing"].get("target_size", [256, 256]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure trained model checkpoint exists
    if not os.path.exists(model_path):
        print(
            f"Model checkpoint '{model_path}' not found. Training UNet model on synthetic benchmark dataset...")
        from training.train import train_model
        train_model(config_path)

    # 1. Load UNet Model
    model = get_segmentation_model(
        architecture=cfg["model"].get("architecture", "unet"),
        in_channels=cfg["model"].get("input_channels", 1),
        num_classes=cfg["model"].get("num_classes", 1),
        features=cfg["model"].get("features", [64, 128, 256, 512]),
        classification_head=cfg["model"].get("classification_head", False)
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. EO Preprocessing
    preprocessor = SARPreprocessor(
        target_size=target_size,
        normalize=cfg["preprocessing"].get("normalize", True)
    )
    preprocessed_2d, input_tensor, meta = preprocessor.preprocess_sar(
        input_image_path)

    # 3. Model Inference (Probability Map)
    in_channels = cfg["model"].get("input_channels", 1)
    if input_tensor.shape[1] == 1 and in_channels > 1:
        input_tensor = input_tensor.repeat(1, in_channels, 1, 1)
    elif input_tensor.shape[1] > in_channels:
        input_tensor = input_tensor[:, :in_channels, :, :]
        
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        prob_tensor = model.predict(input_tensor)
    prob_map = prob_tensor.squeeze().cpu().numpy()

    # 4. Post-processing (Thresholding, Morphology, Connected Component Analysis)
    mask_processor = MaskProcessor(
        probability_threshold=th,
        morphology_kernel_size=cfg["postprocessing"].get(
            "morphology_kernel_size", 3),
        min_region_pixels=cfg["postprocessing"].get("min_region_pixels", 50)
    )
    _raw_mask, cleaned_mask, regions_raw = mask_processor.process(
        prob_map, threshold=th)

    # Resize masks and probability map back to original image dimensions for accurate geometry mapping
    orig_h, orig_w = meta["height"], meta["width"]
    if prob_map.shape != (orig_h, orig_w):
        pil_prob = Image.fromarray(
            (prob_map * 255).astype(np.uint8)).resize((orig_w, orig_h), resample=Image.BILINEAR)
        prob_map_orig = np.array(pil_prob, dtype=np.float32) / 255.0

        pil_cleaned = Image.fromarray(
            (cleaned_mask * 255).astype(np.uint8)).resize((orig_w, orig_h), resample=Image.NEAREST)
        cleaned_mask_orig = (
            np.array(pil_cleaned, dtype=np.uint8) > 127).astype(np.uint8)

        cleaned_mask, regions_raw = mask_processor.extract_connected_components(
            cleaned_mask_orig, prob_map_orig)
        prob_map = prob_map_orig
    else:
        cleaned_mask_orig = cleaned_mask

    # 5. Geospatial & Quantitative Characterization
    geo_processor = GeospatialProcessor(
        default_crs=cfg["geospatial"].get("default_crs", "EPSG:4326"))
    validator = SpillValidator(
        min_area_km2=cfg["validation"].get("min_area_km2", 0.001),
        max_area_km2=cfg["validation"].get("max_area_km2", 5000.0),
        min_confidence=cfg["validation"].get("min_confidence", 0.3)
    )

    spill_region_records = []
    geojson_region_data = []

    for reg in regions_raw:
        geom_info = geo_processor.process_region_geometry(
            cleaned_mask, reg, meta)
        reg_status, reg_warnings = validator.validate_region(
            {"detection": {"confidence": reg["confidence"]},
                "measurements": geom_info, "geometry": geom_info},
            meta
        )

        region_record = SpillRegionRecord(
            spill_id=reg["spill_id"],
            detection=DetectionMetadata(
                confidence=reg["confidence"],
                model=cfg["model"].get("architecture", "UNet"),
                threshold=th
            ),
            geometry=GeometryMetadata(
                is_geographic=geom_info["is_geographic"],
                crs=geom_info["crs"],
                centroid=Centroid(**geom_info["centroid"]),
                bbox=geom_info["bbox"],
                polygon_geojson=geom_info["polygon_geojson"]
            ),
            measurements=MeasurementsMetadata(
                area_km2=geom_info["area_km2"],
                perimeter_km=geom_info["perimeter_km"],
                area_pixels=geom_info["area_pixels"],
                width_pixels=geom_info["width_pixels"],
                height_pixels=geom_info["height_pixels"],
                aspect_ratio=geom_info["aspect_ratio"]
            ),
            validation=ValidationMetadata(
                status=reg_status, warnings=reg_warnings)
        )
        spill_region_records.append(region_record)
        geojson_region_data.append(region_record.model_dump())

    # 6. Overall Image Validation
    overall_status, overall_warnings = validator.validate_event(
        len(spill_region_records),
        [r.model_dump() for r in spill_region_records],
        meta
    )

    overall_conf = float(np.mean(
        [r.detection.confidence for r in spill_region_records])) if spill_region_records else 0.0

    # 7. Construct SpillEvent Data Contract
    event_id = f"SPILL_EVENT_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    spill_event = SpillEvent(
        event_id=event_id,
        source=SourceMetadata(
            satellite="Sentinel-1",
            sensor="SAR",
            image_path=os.path.abspath(input_image_path)
        ),
        overall_confidence=round(overall_conf, 4),
        num_spill_regions=len(spill_region_records),
        spill_regions=spill_region_records,
        validation=ValidationMetadata(
            status=overall_status, warnings=overall_warnings)
    )

    # 8. Save Outputs
    os.makedirs(output_dir, exist_ok=True)
    masks_dir = os.path.join(output_dir, "masks")
    polygons_dir = os.path.join(output_dir, "polygons")
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(polygons_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Save probability map & binary mask images
    Image.fromarray((prob_map * 255).astype(np.uint8)
                    ).save(os.path.join(output_dir, "probability_map.png"))
    Image.fromarray((cleaned_mask * 255).astype(np.uint8)
                    ).save(os.path.join(output_dir, "segmentation_mask.png"))

    # Save SpillEvent JSON
    with open(os.path.join(output_dir, "spill_event.json"), "w") as f:
        f.write(spill_event.model_dump_json(indent=2))

    # Save GeoJSON
    geojson_data = geo_processor.export_geojson(
        geojson_region_data, crs_str=meta.get("crs"))
    with open(os.path.join(output_dir, "spill.geojson"), "w") as f:
        json.dump(geojson_data, f, indent=2)

    # Save Visualization Overlay
    original_img, _ = preprocessor.read_raster(input_image_path)
    visualize_pipeline_results(
        original_img=original_img,
        preprocessed_img=preprocessed_2d,
        prob_map=prob_map,
        binary_mask=cleaned_mask,
        region_records=geojson_region_data,
        output_path=os.path.join(output_dir, "spill_overlay.png")
    )

    print("\n" + "="*60)
    print(f"Agent 1 Detection Complete for: {input_image_path}")
    print(f"Event ID:             {spill_event.event_id}")
    print(f"Overall Validation:   {spill_event.validation.status}")
    print(f"Detected Slicks:      {spill_event.num_spill_regions}")
    print(f"Overall Confidence:   {spill_event.overall_confidence}")
    print(f"Outputs written to:   '{output_dir}/'")
    print("="*60 + "\n")

    return spill_event


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Agent 1 - EO Optical Oil Spill Inference")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input EO image (.tif, .png, etc.)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Segmentation probability threshold (default: 0.5)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration YAML file")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save output predictions and artifacts")
    args = parser.parse_args()

    run_inference(args.input, threshold=args.threshold,
                  config_path=args.config, output_dir=args.output_dir)
