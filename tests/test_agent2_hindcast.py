from datetime import datetime, timezone
from shapely.geometry import Polygon, mapping

from agent2.adapters.mock_forcing import MockEnvironmentalForcingProvider
from agent2.contracts.spill_observation import SpillObservation
from agent2.hindcast.hindcast_engine import HindcastingEngine


def test_hindcast_two_stage_execution():
    forcing = MockEnvironmentalForcingProvider(
        mode="uniform",
        current_u=0.25,
        current_v=0.1,
        wind_u=3.0,
        wind_v=1.0
    )

    t0 = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    poly = Polygon([[80.1, 15.1], [80.13, 15.1], [80.13, 15.13], [80.1, 15.13], [80.1, 15.1]])

    spill = SpillObservation(
        spill_id="TEST_HINDCAST_001",
        observation_timestamp=t0,
        geometry=mapping(poly),
        centroid=(80.115, 15.115),
        area_km2=6.5,
        confidence=0.92
    )

    engine = HindcastingEngine(
        forcing_provider=forcing,
        max_lookback_hours=12,
        time_step_hours=6,
        particles_per_candidate=50,
        timestep_minutes=30.0,
        seed=42
    )

    res = engine.run(spill)

    assert res.status == "SUCCESS"
    assert res.probable_origin is not None
    assert len(res.candidate_hypotheses) > 0
    assert res.best_candidate is not None

    # Verify probable origin is situated upstream / backward from observation (lower longitude/latitude)
    po = res.probable_origin
    assert po.centroid[0] < spill.lon
    assert po.centroid[1] < spill.lat

    # Verify origin time window is earlier than observation
    assert po.time_window_start < spill.observation_timestamp
    assert po.time_window_end <= spill.observation_timestamp

    # Stage A historical envelope exists
    assert res.historical_envelope_geojson is not None
    assert res.historical_envelope_geojson["type"] in ["Polygon", "MultiPolygon"]
