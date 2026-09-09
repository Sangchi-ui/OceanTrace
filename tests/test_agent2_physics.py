from datetime import datetime, timezone, timedelta
import numpy as np
from shapely.geometry import Polygon

from agent2.adapters.mock_forcing import MockEnvironmentalForcingProvider
from agent2.physics.particle_cloud import ParticleCloud
from agent2.physics.transport_engine import LagrangianTransportEngine
from agent2.geospatial.geodesy import haversine_distance_km


def test_particle_cloud_polygon_sampling():
    # Square polygon around (80.0, 15.0) of ~10 km width
    poly = Polygon([
        [79.95, 14.95],
        [80.05, 14.95],
        [80.05, 15.05],
        [79.95, 15.05],
        [79.95, 14.95]
    ])
    cloud = ParticleCloud.from_polygon(poly, num_particles=300, seed=42)

    assert cloud.count == 300
    c_lon, c_lat = cloud.get_centroid()
    assert abs(c_lon - 80.0) < 0.02
    assert abs(c_lat - 15.0) < 0.02

    # Envelope test
    env = cloud.to_envelope_polygon(percentile=90.0)
    assert not env.is_empty
    assert env.area > 0


def test_eastward_advection_displacement():
    # Pure eastward current of 0.5 m/s, zero wind
    forcing = MockEnvironmentalForcingProvider(
        mode="uniform", current_u=0.5, current_v=0.0, wind_u=0.0, wind_v=0.0
    )
    engine = LagrangianTransportEngine(
        forcing_provider=forcing,
        timestep_minutes=15.0,
        horizontal_diffusivity=0.0  # zero diffusion for deterministic distance check
    )

    t0 = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=2)  # 2 hours = 7200 seconds -> 3600 meters = 3.6 km

    poly = Polygon([[80.0, 15.0], [80.02, 15.0], [80.02, 15.02], [80.0, 15.02], [80.0, 15.0]])
    cloud = ParticleCloud.from_polygon(poly, num_particles=100, seed=42)

    init_lon, init_lat = cloud.get_centroid()
    engine.simulate(cloud, start_time=t0, end_time=t1, apply_diffusion=False)
    final_lon, final_lat = cloud.get_centroid()

    # Final longitude should have moved eastward (greater longitude)
    assert final_lon > init_lon
    # Final latitude should have stayed nearly constant
    assert abs(final_lat - init_lat) < 0.001

    dist_km = haversine_distance_km(init_lon, init_lat, final_lon, final_lat)
    # Expected distance: 3.6 km (within ~5% spherical projection tolerance)
    assert abs(dist_km - 3.6) < 0.2


def test_windage_effect():
    # Zero current, 10 m/s northward wind
    forcing = MockEnvironmentalForcingProvider(
        mode="uniform", current_u=0.0, current_v=0.0, wind_u=0.0, wind_v=10.0
    )
    # 3% windage -> 0.3 m/s northward
    engine_3pct = LagrangianTransportEngine(
        forcing_provider=forcing, timestep_minutes=15.0, windage=0.030, horizontal_diffusivity=0.0
    )
    # 4% windage -> 0.4 m/s northward
    engine_4pct = LagrangianTransportEngine(
        forcing_provider=forcing, timestep_minutes=15.0, windage=0.040, horizontal_diffusivity=0.0
    )

    t0 = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=1)

    cloud1 = ParticleCloud.from_point(80.0, 15.0, radius_km=0.1, num_particles=50, seed=42)
    cloud2 = ParticleCloud.from_point(80.0, 15.0, radius_km=0.1, num_particles=50, seed=42)

    engine_3pct.simulate(cloud1, t0, t1, apply_diffusion=False)
    engine_4pct.simulate(cloud2, t0, t1, apply_diffusion=False)

    _, lat1 = cloud1.get_centroid()
    _, lat2 = cloud2.get_centroid()

    # 4% windage should have traveled further north than 3% windage
    assert lat2 > lat1
