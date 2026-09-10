"""
NOAA Global Forecast System (GFS) & ERA5 Atmospheric Wind Forcing Provider.
Ingests multi-dimensional gridded wind fields using Xarray and NetCDF4.
Supports standard GFS variables (ugrd10m, vgrd10m) and ECMWF variables (u10, v10).
"""

from datetime import datetime
import os
from typing import Any, Dict, Optional, Tuple
import numpy as np

try:
    import xarray as xr
    import netCDF4
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

from agent2.adapters.forcing_base import EnvironmentalForcingProvider
from agent2.errors import EnvironmentalDataError


class NOAAGFSWindProvider(EnvironmentalForcingProvider):
    """
    Ingests NOAA GFS / ECMWF ERA5 10m atmospheric wind datasets using Xarray / NetCDF4.
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        source_name: str = "NOAA_GFS_0p25"
    ):
        self.dataset_path = dataset_path
        self.source_name = source_name
        self._dataset: Optional[Any] = None

        if self.dataset_path and os.path.exists(self.dataset_path):
            if not XARRAY_AVAILABLE:
                raise EnvironmentalDataError("xarray and netCDF4 are required to read GFS datasets.")
            try:
                self._dataset = xr.open_dataset(self.dataset_path)
            except Exception as e:
                raise EnvironmentalDataError(f"Failed to open GFS NetCDF file at {self.dataset_path}: {e}")

    def get_current_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise EnvironmentalDataError("Atmospheric Wind Provider does not provide ocean current vectors.")

    def get_wind_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolates 10m u/v wind vectors at query coordinates using Xarray."""
        if self._dataset is not None:
            try:
                # Support standard GFS and ERA5 variable naming conventions
                u_var = next((v for v in ["ugrd10m", "u10", "10u", "u_wind"] if v in self._dataset), None)
                v_var = next((v for v in ["vgrd10m", "v10", "10v", "v_wind"] if v in self._dataset), None)

                if not u_var or not v_var:
                    raise EnvironmentalDataError(
                        f"Wind variables not found in GFS dataset. Available: {list(self._dataset.data_vars)}"
                    )

                ds_t = self._dataset.sel(time=timestamp, method="nearest")
                lon_dim = "lon" if "lon" in ds_t.coords else "longitude"
                lat_dim = "lat" if "lat" in ds_t.coords else "latitude"

                u_interp = ds_t[u_var].interp({lon_dim: ("points", lons), lat_dim: ("points", lats)}).values
                v_interp = ds_t[v_var].interp({lon_dim: ("points", lons), lat_dim: ("points", lats)}).values

                return np.asarray(u_interp, dtype=np.float64), np.asarray(v_interp, dtype=np.float64)
            except Exception as e:
                raise EnvironmentalDataError(f"Error interpolating GFS wind vectors: {e}")

        raise EnvironmentalDataError(
            f"NOAA GFS wind forcing is not connected. No dataset path provided. "
            f"In accordance with the OceanTrace No-Fabrication Policy, missing wind forcing will not be synthesized."
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "provider": "NOAAGFSWindProvider",
            "source": self.source_name,
            "dataset_path": self.dataset_path,
            "is_connected": self._dataset is not None,
            "engine": "Xarray/NetCDF4"
        }
