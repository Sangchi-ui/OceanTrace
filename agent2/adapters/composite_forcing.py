"""
Composite Environmental Forcing Provider.
Pairs independent ocean current and atmospheric wind providers into a unified forcing engine.
"""

from datetime import datetime
from typing import Any, Dict, Tuple
import numpy as np

from agent2.adapters.forcing_base import EnvironmentalForcingProvider


class CompositeForcingProvider(EnvironmentalForcingProvider):
    """
    Blends an ocean current provider with an atmospheric wind provider.
    """

    def __init__(
        self,
        current_provider: EnvironmentalForcingProvider,
        wind_provider: EnvironmentalForcingProvider
    ):
        self.current_provider = current_provider
        self.wind_provider = wind_provider

    def get_current_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.current_provider.get_current_vectors(lons, lats, timestamp)

    def get_wind_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.wind_provider.get_wind_vectors(lons, lats, timestamp)

    def metadata(self) -> Dict[str, Any]:
        return {
            "provider": "CompositeForcingProvider",
            "current_source": self.current_provider.metadata(),
            "wind_source": self.wind_provider.metadata()
        }
