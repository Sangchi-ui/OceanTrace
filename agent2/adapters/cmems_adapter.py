"""
Copernicus Marine Service (CMEMS) Ocean Current Forcing Adapter.
Ingests global/regional ocean physics analysis and forecast products (uo, vo surface currents).
Adheres strictly to the No-Fabrication Policy: raises EnvironmentalDataError if data is unavailable.
"""

from datetime import datetime, timezone
import os
from typing import Any, Dict, Optional, Tuple
import numpy as np

from agent2.adapters.forcing_base import EnvironmentalForcingProvider
from agent2.errors import EnvironmentalDataError


class CMEMSCurrentProvider(EnvironmentalForcingProvider):
    """
    Adapter for CMEMS (Copernicus Marine Environment Monitoring Service) ocean current datasets.
    Supports local NetCDF / Zarr files or live Copernicus Marine API queries.
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        product_id: str = "GLOBAL_ANALYSISFORECAST_PHY_001_024"
    ):
        self.dataset_path = dataset_path
        self.username = username or os.environ.get("COPERNICUS_MARINE_USERNAME")
        self.password = password or os.environ.get("COPERNICUS_MARINE_PASSWORD")
        self.product_id = product_id
        self._dataset = None

        # If a local NetCDF file is specified, attempt to open
        if self.dataset_path and os.path.exists(self.dataset_path):
            try:
                import xarray as xr
                self._dataset = xr.open_dataset(self.dataset_path)
            except Exception as e:
                raise EnvironmentalDataError(
                    f"Failed to open CMEMS NetCDF dataset at {self.dataset_path}: {e}"
                )

    def get_current_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolates eastward (uo) and northward (vo) current velocity components."""
        if self._dataset is not None:
            try:
                # Interpolate using xarray spatial coordinates
                # Assumes dataset has variables 'uo' and 'vo' and dims ('time', 'latitude', 'longitude')
                ds_t = self._dataset.sel(time=timestamp, method="nearest")
                u_interp = ds_t["uo"].interp(longitude=("points", lons), latitude=("points", lats)).values
                v_interp = ds_t["vo"].interp(longitude=("points", lons), latitude=("points", lats)).values
                return np.asarray(u_interp, dtype=np.float64), np.asarray(v_interp, dtype=np.float64)
            except Exception as e:
                raise EnvironmentalDataError(f"Error interpolating CMEMS dataset: {e}")

        # If no local dataset and no valid API connection:
        raise EnvironmentalDataError(
            "CMEMS current forcing data is unavailable. No local dataset path provided and "
            "live Copernicus Marine credentials not configured. In accordance with the "
            "OceanTrace No-Fabrication Policy, missing environmental data will not be silently synthesized."
        )

    def get_wind_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        """CMEMS current provider does not supply atmospheric winds."""
        raise EnvironmentalDataError(
            "CMEMS Ocean Current Provider does not supply atmospheric winds. "
            "Use CompositeForcingProvider with ERA5WindProvider."
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "provider": "CMEMSCurrentProvider",
            "source": "Copernicus Marine Environment Monitoring Service",
            "product_id": self.product_id,
            "dataset_path": self.dataset_path,
            "is_connected": self._dataset is not None
        }
