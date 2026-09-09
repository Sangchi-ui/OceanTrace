import pytest
from datetime import datetime, timezone

from agent2.adapters.agent1_adapter import Agent1Adapter
from agent2.errors import GeometryValidationError, Agent1ContractError


def test_agent1_adapter_from_spill_event_dict():
    sample_agent1_dict = {
        "event_id": "SPILL_EVENT_20260909_120000",
        "timestamp": "2026-09-09T12:00:00+00:00",
        "source": {
            "satellite": "Sentinel-1",
            "sensor": "SAR",
            "image_path": "data/sample/sample_sentinel1.tif",
            "acquisition_time": "2026-09-09T11:45:00Z"
        },
        "overall_confidence": 0.91,
        "num_spill_regions": 1,
        "spill_regions": [
            {
                "spill_id": "SPILL_001",
                "detection": {
                    "confidence": 0.91,
                    "model": "UNet",
                    "threshold": 0.5
                },
                "geometry": {
                    "is_geographic": True,
                    "crs": "EPSG:4326",
                    "centroid": {
                        "latitude": 17.5,
                        "longitude": 83.2,
                        "pixel_x": 128.0,
                        "pixel_y": 128.0
                    },
                    "bbox": [83.15, 17.45, 83.25, 17.55],
                    "polygon_geojson": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [83.15, 17.45],
                                [83.25, 17.45],
                                [83.25, 17.55],
                                [83.15, 17.55],
                                [83.15, 17.45]
                            ]
                        ]
                    }
                },
                "measurements": {
                    "area_km2": 4.52,
                    "perimeter_km": 9.15,
                    "area_pixels": 4500,
                    "width_pixels": 80.0,
                    "height_pixels": 60.0,
                    "aspect_ratio": 1.33
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

    obs = Agent1Adapter.parse(sample_agent1_dict)

    assert obs.spill_id == "SPILL_001"
    assert obs.lon == 83.2
    assert obs.lat == 17.5
    assert obs.area_km2 == 4.52
    assert obs.confidence == 0.91
    assert obs.source_satellite == "Sentinel-1"
    assert obs.observation_timestamp == datetime(2026, 9, 9, 11, 45, 0, tzinfo=timezone.utc)


def test_agent1_adapter_rejects_non_georeferenced():
    ungeo_dict = {
        "event_id": "SPILL_EVENT_UNGEO",
        "timestamp": "2026-09-09T12:00:00Z",
        "spill_regions": [
            {
                "spill_id": "SPILL_PIXEL",
                "geometry": {
                    "is_geographic": False,
                    "crs": None,
                    "centroid": {"latitude": None, "longitude": None, "pixel_x": 50.0, "pixel_y": 50.0},
                    "bbox": [0, 0, 100, 100],
                    "polygon_geojson": {"type": "Polygon", "coordinates": [[[0,0], [10,0], [10,10], [0,10], [0,0]]]}
                },
                "measurements": {"area_km2": None, "area_pixels": 100, "aspect_ratio": 1.0},
                "detection": {"confidence": 0.8}
            }
        ]
    }

    with pytest.raises(GeometryValidationError) as excinfo:
        Agent1Adapter.parse(ungeo_dict)

    assert "not georeferenced" in str(excinfo.value)


def test_agent1_adapter_from_geojson_collection():
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "spill_id": "SLICK_GEO_01",
                    "confidence": 0.85,
                    "area_km2": 2.75,
                    "centroid_lon": 82.5,
                    "centroid_lat": 16.8,
                    "is_geographic": True,
                    "timestamp": "2026-09-09T10:00:00Z"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[82.48, 16.78], [82.52, 16.78], [82.52, 16.82], [82.48, 16.82], [82.48, 16.78]]
                    ]
                }
            }
        ]
    }

    obs = Agent1Adapter.parse(geojson_data)
    assert obs.spill_id == "SLICK_GEO_01"
    assert obs.lon == 82.5
    assert obs.lat == 16.8
    assert obs.area_km2 == 2.75
