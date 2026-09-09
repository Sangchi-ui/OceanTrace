from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from agent2.contracts import (
    SpillObservation,
    TrajectoryPoint,
    CandidateHypothesis,
    OriginRegion,
    HindcastResult,
    ForecastHorizonResult,
    ForecastResult,
    Agent2Result,
)


def test_spill_observation_contract():
    now_utc = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    obs = SpillObservation(
        spill_id="SPILL_001",
        observation_timestamp=now_utc,
        geometry={
            "type": "Polygon",
            "coordinates": [[[80.0, 15.0], [80.1, 15.0], [80.1, 15.1], [80.0, 15.1], [80.0, 15.0]]]
        },
        centroid=(80.05, 15.05),
        area_km2=12.5,
        perimeter_km=14.2,
        confidence=0.88,
        crs="EPSG:4326",
        source_satellite="Sentinel-1",
        source_sensor="SAR"
    )

    assert obs.spill_id == "SPILL_001"
    assert obs.lon == 80.05
    assert obs.lat == 15.05
    assert obs.observation_timestamp.tzinfo == timezone.utc

    # Test JSON serialization and roundtrip
    dumped = obs.model_dump_json()
    loaded = SpillObservation.model_validate_json(dumped)
    assert loaded.spill_id == obs.spill_id
    assert loaded.area_km2 == 12.5


def test_naive_timestamp_auto_converted_to_utc():
    naive_time = datetime(2026, 9, 9, 12, 0, 0)
    obs = SpillObservation(
        spill_id="SPILL_002",
        observation_timestamp=naive_time,
        geometry={"type": "Point", "coordinates": [0.0, 0.0]},
        centroid=(0.0, 0.0),
        area_km2=1.0,
        confidence=0.5
    )
    assert obs.observation_timestamp.tzinfo == timezone.utc


def test_invalid_area_raises_validation_error():
    with pytest.raises(ValidationError):
        SpillObservation(
            spill_id="SPILL_INV",
            observation_timestamp=datetime.now(timezone.utc),
            geometry={},
            centroid=(0.0, 0.0),
            area_km2=-5.0,  # Negative area should fail
            confidence=0.5
        )
