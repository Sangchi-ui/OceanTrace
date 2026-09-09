from datetime import datetime, timezone
import numpy as np
import pytest

from agent2.adapters.mock_forcing import MockEnvironmentalForcingProvider
from agent2.adapters.cmems_adapter import CMEMSCurrentProvider
from agent2.adapters.composite_forcing import CompositeForcingProvider
from agent2.errors import EnvironmentalDataError


def test_mock_forcing_uniform():
    provider = MockEnvironmentalForcingProvider(
        mode="uniform", current_u=0.3, current_v=-0.1, wind_u=5.0, wind_v=2.5
    )
    t = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    lons = np.array([80.0, 81.0, 82.0])
    lats = np.array([15.0, 15.5, 16.0])

    u_curr, v_curr = provider.get_current_vectors(lons, lats, t)
    assert np.allclose(u_curr, 0.3)
    assert np.allclose(v_curr, -0.1)

    u_w, v_w = provider.get_wind_vectors(lons, lats, t)
    assert len(u_w) == 3
    assert np.all(u_w > 0.0)


def test_mock_forcing_gyre_rotational():
    provider = MockEnvironmentalForcingProvider(
        mode="gyre", gyre_center=(80.0, 15.0), gyre_radius_km=30.0, gyre_max_speed=0.6
    )
    t = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    # North of gyre center should have negative eastward velocity (counter-clockwise)
    u_n, v_n = provider.get_current(80.0, 15.2, t)
    # South of gyre center should have positive eastward velocity
    u_s, v_s = provider.get_current(80.0, 14.8, t)
    assert u_n < u_s


def test_cmems_raises_when_unconfigured():
    provider = CMEMSCurrentProvider(dataset_path=None, username=None, password=None)
    t = datetime.now(timezone.utc)
    with pytest.raises(EnvironmentalDataError) as excinfo:
        provider.get_current_vectors(np.array([80.0]), np.array([15.0]), t)
    assert "CMEMS current forcing data is unavailable" in str(excinfo.value)
