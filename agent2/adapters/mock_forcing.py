"""
Deterministic Synthetic Environmental Forcing Provider for Agent 2.
Supports offline development, reproducible benchmarking, unit testing, and CI/CD pipelines
without requiring live CMEMS or ERA5 credentials or network downloads.
"""

from datetime import datetime, timezone
import math
from typing import Any, Dict, Optional, Tuple
import numpy as np

from agent2.adapters.forcing_base import EnvironmentalForcingProvider


class MockEnvironmentalForcingProvider(EnvironmentalForcingProvider):
    """
    Synthesizes physically realistic, deterministic ocean current and 10m wind fields.
    Modes:
      - 'uniform': Constant eastward/northward current and wind vectors.
      - 'gyre': Rotating oceanic vortex (mesoscale eddy) with background advection.
      - 'shear': Spatially sheared current velocity field.
      - 'tidal': Harmonic oscillatory tidal current superimposed on mean drift.
    """

    def __init__(
        self,
        mode: str = "uniform",
        current_u: float = 0.25,   # m/s eastward
        current_v: float = 0.15,   # m/s northward
        wind_u: float = 4.0,       # m/s 10m eastward wind
        wind_v: float = 2.0,       # m/s 10m northward wind
        gyre_center: Tuple[float, float] = (80.0, 15.0),
        gyre_radius_km: float = 40.0,
        gyre_max_speed: float = 0.5,
        gyre_clockwise: bool = False,
        tidal_period_hours: float = 12.42,
        tidal_amplitude: float = 0.2,
        seed: Optional[int] = 42
    ):
        self.mode = mode.lower()
        self.current_u = float(current_u)
        self.current_v = float(current_v)
        self.wind_u = float(wind_u)
        self.wind_v = float(wind_v)
        self.gyre_center = gyre_center
        self.gyre_radius_km = float(gyre_radius_km)
        self.gyre_max_speed = float(gyre_max_speed)
        self.gyre_clockwise = gyre_clockwise
        self.tidal_period_hours = float(tidal_period_hours)
        self.tidal_amplitude = float(tidal_amplitude)
        self.seed = seed

    def get_current_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates current velocity vectors at array of coordinates and timestamp."""
        lons = np.asarray(lons, dtype=np.float64)
        lats = np.asarray(lats, dtype=np.float64)

        if self.mode == "uniform":
            u = np.full_like(lons, self.current_u)
            v = np.full_like(lats, self.current_v)
            return u, v

        elif self.mode == "gyre":
            # Mesoscale vortex centered at (c_lon, c_lat)
            c_lon, c_lat = self.gyre_center
            dx_deg = lons - c_lon
            dy_deg = lats - c_lat
            # Convert degrees to km approximately
            mean_lat = np.mean(lats) if len(lats) > 0 else c_lat
            dx_km = dx_deg * 111.32 * math.cos(math.radians(mean_lat))
            dy_km = dy_deg * 110.57
            r_km = np.sqrt(dx_km**2 + dy_km**2) + 1e-6

            # Rankine/Gaussian vortex profile: V_theta = V_max * (r / R) * exp(-0.5 * (r/R)^2)
            r_norm = r_km / self.gyre_radius_km
            v_theta = self.gyre_max_speed * r_norm * np.exp(-0.5 * (r_norm**2)) * math.e

            # Tangential velocity components (-sin(theta), cos(theta))
            sign = 1.0 if not self.gyre_clockwise else -1.0
            u_rot = -sign * v_theta * (dy_km / r_km)
            v_rot = sign * v_theta * (dx_km / r_km)

            # Superimpose background drift
            u = self.current_u + u_rot
            v = self.current_v + v_rot
            return u, v

        elif self.mode == "shear":
            # Current speed increases linearly with latitude
            c_lat = self.gyre_center[1]
            shear_factor = 1.0 + 0.05 * (lats - c_lat)
            u = self.current_u * shear_factor
            v = np.full_like(lats, self.current_v)
            return u, v

        elif self.mode == "tidal":
            # Semi-diurnal M2 tidal oscillation
            epoch_seconds = timestamp.replace(tzinfo=timezone.utc).timestamp()
            omega = 2.0 * math.pi / (self.tidal_period_hours * 3600.0)
            phase = omega * epoch_seconds
            u = self.current_u + self.tidal_amplitude * np.cos(phase)
            v = self.current_v + self.tidal_amplitude * np.sin(phase)
            return np.full_like(lons, u), np.full_like(lats, v)

        else:
            # Fallback uniform
            return np.full_like(lons, self.current_u), np.full_like(lats, self.current_v)

    def get_wind_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates 10m wind velocity vectors at array of coordinates and timestamp."""
        lons = np.asarray(lons, dtype=np.float64)
        lats = np.asarray(lats, dtype=np.float64)
        # 10m wind with slight synoptic spatial gradient
        c_lat = self.gyre_center[1]
        grad = 1.0 + 0.01 * (lats - c_lat)
        u_wind = self.wind_u * grad
        v_wind = np.full_like(lats, self.wind_v)
        return u_wind, v_wind

    def metadata(self) -> Dict[str, Any]:
        return {
            "provider": "MockEnvironmentalForcingProvider",
            "type": "synthetic_deterministic",
            "mode": self.mode,
            "mean_current_u_m_s": self.current_u,
            "mean_current_v_m_s": self.current_v,
            "mean_wind_u_m_s": self.wind_u,
            "mean_wind_v_m_s": self.wind_v,
            "gyre_center": list(self.gyre_center) if self.mode == "gyre" else None,
            "seed": self.seed
        }
