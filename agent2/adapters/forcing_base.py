"""
Abstract base classes and data structures for Environmental Forcing Providers.
Provides clean separation between transport simulation and external data acquisition (CMEMS, ERA5, Mock).
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Tuple
import numpy as np


class EnvironmentalForcingProvider(ABC):
    """
    Abstract interface for retrieving oceanographic and atmospheric forcing fields.
    All velocities are in meters per second (m/s) in eastward (u) and northward (v) directions.
    """

    @abstractmethod
    def get_current_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves ocean surface current velocity components (u_current, v_current) in m/s
        at the specified positions and timestamp.
        """
        pass

    @abstractmethod
    def get_wind_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves 10-meter wind velocity components (u10, v10) in m/s
        at the specified positions and timestamp.
        """
        pass

    def get_current(self, lon: float, lat: float, timestamp: datetime) -> Tuple[float, float]:
        """Convenience method for a single point query."""
        u_arr, v_arr = self.get_current_vectors(
            np.array([lon], dtype=np.float64),
            np.array([lat], dtype=np.float64),
            timestamp
        )
        return float(u_arr[0]), float(v_arr[0])

    def get_ocean_current(self, lon: float, lat: float, timestamp: datetime) -> Tuple[float, float]:
        """Alias for get_current."""
        return self.get_current(lon, lat, timestamp)

    def get_wind(self, lon: float, lat: float, timestamp: datetime) -> Tuple[float, float]:
        """Convenience method for a single point query."""
        u_arr, v_arr = self.get_wind_vectors(
            np.array([lon], dtype=np.float64),
            np.array([lat], dtype=np.float64),
            timestamp
        )
        return float(u_arr[0]), float(v_arr[0])

    def get_surface_wind(self, lon: float, lat: float, timestamp: datetime) -> Tuple[float, float]:
        """Alias for get_wind."""
        return self.get_wind(lon, lat, timestamp)

    def get_sst(self, lon: float, lat: float, timestamp: datetime) -> float:
        """Returns Sea Surface Temperature in degrees Celsius (default 26.0 C)."""
        return 26.0

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Returns provenance and diagnostic metadata about the forcing source."""
        pass
