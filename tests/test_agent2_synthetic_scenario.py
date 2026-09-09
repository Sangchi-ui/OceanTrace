from datetime import datetime, timezone, timedelta
import math
import numpy as np
from shapely.geometry import Point, Polygon, shape, mapping

from agent2.adapters.mock_forcing import MockEnvironmentalForcingProvider
from agent2.contracts.spill_observation import SpillObservation
from agent2.geospatial.geodesy import (
    haversine_distance_km,
    meters_to_degrees_lon,
    meters_to_degrees_lat,
)
from agent2.pipeline import Agent2Pipeline
from agent2.config import Agent2Config, HindcastConfig, ForecastConfig, ForcingConfig, MockForcingConfig


def test_synthetic_controlled_recovery_scenario():
    """
    Controlled benchmark experiment:
    1. Known true source origin: (80.0, 15.0)
    2. Known true release time: T0 (12 hours prior to observation)
    3. Known forcing: uniform eastward current (0.3 m/s) + northward current (0.15 m/s),
       eastward wind (4.0 m/s) + northward wind (2.0 m/s), 3% windage.
    4. Advect to T_obs = T0 + 12h to form the detected synthetic slick.
    5. Agent 2 must recover the true source region within uncertainty tolerance.
    """
    # 1. True Source & Release Time
    true_origin_lon = 80.00
    true_origin_lat = 15.00
    t_release = datetime(2026, 9, 9, 0, 0, 0, tzinfo=timezone.utc)
    lookback_hours = 12
    t_obs = t_release + timedelta(hours=lookback_hours)

    u_curr = 0.30
    v_curr = 0.15
    u_wind = 4.00
    v_wind = 2.00
    windage = 0.030

    u_eff = u_curr + (windage * u_wind)  # 0.42 m/s
    v_eff = v_curr + (windage * v_wind)  # 0.21 m/s

    dt_sec = lookback_hours * 3600.0  # 43,200 s
    dx_m = u_eff * dt_sec             # 18,144 m
    dy_m = v_eff * dt_sec             #  9,072 m

    # Synthetic observed slick location
    obs_lat = true_origin_lat + meters_to_degrees_lat(dy_m)
    obs_lon = true_origin_lon + meters_to_degrees_lon(dx_m, (true_origin_lat + obs_lat) / 2.0)

    # Build observed polygon (~1 km radius)
    r_deg = 0.01
    poly_obs = Polygon([
        [obs_lon - r_deg, obs_lat - r_deg],
        [obs_lon + r_deg, obs_lat - r_deg],
        [obs_lon + r_deg, obs_lat + r_deg],
        [obs_lon - r_deg, obs_lat + r_deg],
        [obs_lon - r_deg, obs_lat - r_deg]
    ])

    spill = SpillObservation(
        spill_id="CONTROLLED_SPILL_EXP1",
        observation_timestamp=t_obs,
        geometry=mapping(poly_obs),
        centroid=(obs_lon, obs_lat),
        area_km2=3.14,
        confidence=0.95
    )

    # 2. Configure Agent 2 with matching forcing
    cfg = Agent2Config(
        hindcast=HindcastConfig(
            max_lookback_hours=18,
            time_step_hours=6,
            particles_per_candidate=100,
            timestep_minutes=20.0
        ),
        forecast=ForecastConfig(
            horizons_hours=[6, 12],
            particle_count=200,
            ensemble_size=5,
            timestep_minutes=20.0
        ),
        forcing=ForcingConfig(
            provider="mock",
            mock=MockForcingConfig(
                mode="uniform",
                current_u=u_curr,
                current_v=v_curr,
                wind_u=u_wind,
                wind_v=v_wind
            )
        ),
        seed=42
    )

    pipeline = Agent2Pipeline(config=cfg)
    result = pipeline.run(spill, mode="both")

    # 3. Verify Hindcast Source Recovery
    assert result.hindcast.enabled is True
    assert result.hindcast.status == "SUCCESS"
    assert result.hindcast.probable_origin is not None

    po = result.hindcast.probable_origin
    origin_poly = shape(po.geometry_geojson)

    # Distance between estimated origin centroid and true source
    est_dist_to_true_km = haversine_distance_km(
        po.centroid[0], po.centroid[1], true_origin_lon, true_origin_lat
    )
    # The estimated origin centroid should be close to the true source
    assert est_dist_to_true_km <= po.uncertainty_radius_km + 2.0

    # True release time should fall within estimated origin time window
    assert po.time_window_start <= t_release <= po.time_window_end

    # The best candidate should have high composite score (> 0.70)
    best_cand = result.hindcast.best_candidate
    assert best_cand is not None
    assert best_cand.composite_score >= 0.65
    assert best_cand.centroid_distance_km < 3.0

    # 4. Verify Forecast Progression
    assert result.forecast.enabled is True
    h6, h12 = result.forecast.horizons
    assert h6.drift_distance_km < h12.drift_distance_km
    # Forecast moves further east and north
    assert h6.centroid[0] > obs_lon
    assert h6.centroid[1] > obs_lat
    assert h12.centroid[0] > h6.centroid[0]
