import numpy as np

from app.geospatial.geometry import GeospatialProcessor


def test_fallback_pixel_geometry():
    processor = GeospatialProcessor()
    cleaned_mask = np.zeros((100, 100), dtype=np.uint8)
    cleaned_mask[10:30, 20:50] = 1

    region_info = {
        "spill_id": "SPILL_001",
        "area_pixels": 900,
        "confidence": 0.85,
        "centroid_pixel": {"x": 35.0, "y": 20.0},
        "bbox_pixel": [20, 10, 50, 30],
        "width_pixels": 30.0,
        "height_pixels": 20.0,
        "aspect_ratio": 1.5,
        "region_mask_coords": np.argwhere(cleaned_mask == 1)
    }

    meta = {"is_georeferenced": False}
    geom_info = processor.process_region_geometry(
        cleaned_mask, region_info, meta)

    assert geom_info["is_geographic"] is False
    assert geom_info["area_km2"] is None
    assert geom_info["perimeter_km"] is None
    assert geom_info["area_pixels"] == 900
    assert geom_info["centroid"]["pixel_x"] == 35.0
    assert geom_info["centroid"]["pixel_y"] == 20.0
