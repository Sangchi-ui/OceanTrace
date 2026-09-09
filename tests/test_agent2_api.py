from fastapi.testclient import TestClient
from app.api.main import app as agent1_app
from agent2.api.main import app as agent2_app

client_agent1 = TestClient(agent1_app)
client_agent2 = TestClient(agent2_app)


def test_agent1_routes_remain_functional():
    """Verify Agent 1 endpoints are completely isolated and functional."""
    res = client_agent1.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["agent"] == "Agent 1 - Deep Learning SAR Oil Spill Detector"


def test_agent2_health():
    """Verify dedicated Agent 2 API health check."""
    res = client_agent2.get("/api/v1/agent2/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "Agent 2" in data["agent"]


def test_agent2_analyze_endpoint():
    """Verify dedicated Agent 2 /analyze endpoint."""
    sample_payload = {
        "event_id": "SPILL_API_TEST",
        "timestamp": "2026-09-09T12:00:00Z",
        "spill_regions": [
            {
                "spill_id": "SPILL_API_01",
                "geometry": {
                    "is_geographic": True,
                    "crs": "EPSG:4326",
                    "centroid": {"longitude": 80.0, "latitude": 15.0},
                    "polygon_geojson": {
                        "type": "Polygon",
                        "coordinates": [[[80.0, 15.0], [80.02, 15.0], [80.02, 15.02], [80.0, 15.02], [80.0, 15.0]]]
                    }
                },
                "measurements": {"area_km2": 4.5},
                "detection": {"confidence": 0.88}
            }
        ]
    }

    res = client_agent2.post("/api/v1/agent2/analyze?mode=both", json=sample_payload)
    assert res.status_code == 200
    data = res.json()
    assert "analysis_id" in data
    assert data["hindcast"]["enabled"] is True
    assert data["forecast"]["enabled"] is True
    assert len(data["forecast"]["horizons"]) > 0
    assert "geojson_collection" in data
