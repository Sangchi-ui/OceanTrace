import json
import os
from datetime import datetime, timezone
import pytest

from agent2.pipeline import Agent2Pipeline


def test_agent2_full_pipeline_end_to_end(tmp_path):
    # 1. Create a synthetic Agent 1 spill_event.json in tmp_path
    sample_agent1_event = {
        "event_id": "SPILL_EVENT_20260909_080000",
        "timestamp": "2026-09-09T08:00:00Z",
        "source": {
            "satellite": "Sentinel-1",
            "sensor": "SAR",
            "image_path": "data/sample/sample_sentinel1.tif",
            "acquisition_time": "2026-09-09T07:55:00Z"
        },
        "overall_confidence": 0.89,
        "num_spill_regions": 1,
        "spill_regions": [
            {
                "spill_id": "SPILL_REGION_001",
                "detection": {
                    "confidence": 0.89,
                    "model": "smp_unet",
                    "threshold": 0.63
                },
                "geometry": {
                    "is_geographic": True,
                    "crs": "EPSG:4326",
                    "centroid": {
                        "latitude": 17.75,
                        "longitude": 83.35,
                        "pixel_x": 200.0,
                        "pixel_y": 200.0
                    },
                    "bbox": [83.30, 17.70, 83.40, 17.80],
                    "polygon_geojson": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [83.30, 17.70],
                                [83.40, 17.70],
                                [83.40, 17.80],
                                [83.30, 17.80],
                                [83.30, 17.70]
                            ]
                        ]
                    }
                },
                "measurements": {
                    "area_km2": 8.5,
                    "perimeter_km": 12.0,
                    "area_pixels": 8500,
                    "width_pixels": 100.0,
                    "height_pixels": 100.0,
                    "aspect_ratio": 1.0
                },
                "validation": {
                    "status": "VALID",
                    "warnings": []
                }
            }
        ],
        "validation": {
            "status": "VALID",
            "warnings": []
        }
    }

    input_json_path = os.path.join(tmp_path, "spill_event.json")
    with open(input_json_path, "w", encoding="utf-8") as f:
        json.dump(sample_agent1_event, f, indent=2)

    output_dir = os.path.join(tmp_path, "agent2_outputs")

    # 2. Run Pipeline
    pipeline = Agent2Pipeline()
    result = pipeline.run_from_file(
        input_path=input_json_path,
        output_dir=output_dir,
        mode="both"
    )

    # 3. Verify Result Contract
    assert result.analysis_id.startswith("A2_")
    assert result.hindcast.enabled is True
    assert result.hindcast.probable_origin is not None
    assert result.forecast.enabled is True
    assert len(result.forecast.horizons) == 3

    # 4. Verify Exported Artifacts
    res_json_file = os.path.join(output_dir, "agent2_result.json")
    unified_geojson_file = os.path.join(output_dir, "agent2_analysis.geojson")
    geojson_dir = os.path.join(output_dir, "geojson")
    origin_geojson_file = os.path.join(geojson_dir, "origin_region.geojson")
    paths_geojson_file = os.path.join(geojson_dir, "hindcast_paths.geojson")
    f6_file = os.path.join(geojson_dir, "forecast_6h.geojson")
    f12_file = os.path.join(geojson_dir, "forecast_12h.geojson")
    f24_file = os.path.join(geojson_dir, "forecast_24h.geojson")

    assert os.path.exists(res_json_file)
    assert os.path.exists(unified_geojson_file)
    assert os.path.exists(origin_geojson_file)
    assert os.path.exists(paths_geojson_file)
    assert os.path.exists(f6_file)
    assert os.path.exists(f12_file)
    assert os.path.exists(f24_file)

    # Validate GeoJSON structure
    with open(unified_geojson_file, "r") as f:
        geo = json.load(f)
    assert geo["type"] == "FeatureCollection"
    assert len(geo["features"]) >= 5
