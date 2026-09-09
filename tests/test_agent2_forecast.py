from datetime import datetime, timezone
from shapely.geometry import Polygon, mapping

from agent2.adapters.mock_forcing import MockEnvironmentalForcingProvider
from agent2.contracts.spill_observation import SpillObservation
from agent2.forecast.forecast_engine import ForecastingEngine


def test_forecasting_engine_multi_horizons():
    forcing = MockEnvironmentalForcingProvider(
        mode="uniform",
        current_u=0.3,   # 0.3 m/s eastward
        current_v=0.1,   # 0.1 m/s northward
        wind_u=4.0,
        wind_v=1.0
    )

    t0 = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    poly = Polygon([[80.0, 15.0], [80.03, 15.0], [80.03, 15.03], [80.0, 15.03], [80.0, 15.0]])

    spill = SpillObservation(
        spill_id="TEST_FORECAST_001",
        observation_timestamp=t0,
        geometry=mapping(poly),
        centroid=(80.015, 15.015),
        area_km2=10.0,
        confidence=0.9
    )

    engine = ForecastingEngine(
        forcing_provider=forcing,
        horizons_hours=[6, 12, 24],
        particle_count=200,
        ensemble_size=5,
        timestep_minutes=30.0,
        seed=42
    )

    res = engine.run(spill)

    assert res.status == "SUCCESS"
    assert len(res.horizons) == 3
    assert len(res.ensemble_trajectories) == 5

    h6, h12, h24 = res.horizons
    assert h6.horizon_hours == 6
    assert h12.horizon_hours == 12
    assert h24.horizon_hours == 24

    # Monotonic drift distance progression
    assert h6.drift_distance_km < h12.drift_distance_km < h24.drift_distance_km
    # Eastward movement (longitude increases)
    assert h6.centroid[0] > spill.lon
    assert h12.centroid[0] > h6.centroid[0]
    assert h24.centroid[0] > h12.centroid[0]

    # Envelopes exist and have non-empty geometry
    assert h24.core_envelope_50_geojson["type"] in ["Polygon", "MultiPolygon"]
    assert h24.spread_envelope_90_geojson["type"] in ["Polygon", "MultiPolygon"]
    assert h24.spread_area_km2 > 0.0
