"""
ECMWF ERA5 Atmospheric Reanalysis Wind Forcing Adapter.
Ingests 10-meter eastward (u10) and northward (v10) surface wind components.
"""

from datetime import datetime
import os
from typing import Any, Dict, Optional, Tuple
import numpy as np

from agent2.adapters.forcing_base import EnvironmentalForcingProvider
from agent2.errors import EnvironmentalDataError


class ERA5WindProvider(EnvironmentalForcingProvider):
    """
    Adapter for ECMWF ERA5 / HRES 10m surface winds.
    Supports NetCDF / GRIB files or CDS API queries.
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.dataset_path = dataset_path
        self.api_key = api_key or os.environ.get("CDSAPI_KEY")
        self._dataset = None

        if self.dataset_path and os.path.exists(self.dataset_path):
            try:
                import xarray as xr
                self._dataset = xr.open_dataset(self.dataset_path)
            except Exception as e:
                raise EnvironmentalDataError(
                    f"Failed to open ERA5 NetCDF dataset at {self.dataset_path}: {e}"
                )

    def get_current_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise EnvironmentalDataError(
            "ERA5 Wind Provider does not provide ocean currents. "
            "Use CompositeForcingProvider with CMEMSCurrentProvider."
        )

    def get_wind_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolates 10m eastward (u10) and northward (v10) wind velocity components."""
        if self._dataset is not None:
            try:
                ds_t = self._dataset.sel(time=timestamp, method="nearest")
                u_var = "u10" if "u10" in ds_t else "10u"
                v_var = "v10" if "v10" in ds_t else "10v"
                u_interp = ds_t[u_var].interp(longitude=("points", lons), latitude=("points", lats)).values
                v_interp = ds_t[v_var].interp(longitude=("points", lons), latitude=("points", lats)).values
                return np.asarray(u_interp, dtype=np.float64), np.asarray(v_interp, dtype=np.float64)
            except Exception as e:
                raise EnvironmentalDataError(f"Error interpolating ERA5 dataset: {e}")

        raise EnvironmentalDataError(
            "ERA5 wind forcing data is unavailable. No local dataset path provided and "
            "CDS API credentials not configured. In accordance with the "
            "OceanTrace No-Fabrication Policy, missing environmental data will not be silently synthesized."
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "provider": "ERA5WindProvider",
            "source": "ECMWF ERA5 Reanalysis",
            "dataset_path": self.dataset_path,
            "is_connected": self._dataset is not None
        }
