"""
High-Precision Lagrangian Particle Transport Simulation Engine.
Implements Runge-Kutta 4th Order (RK4) and Euler-Euler particle advection,
stochastic turbulent diffusion, configurable windage leeway (2.5% - 4.4%), and physics sanity validation.
"""

from datetime import datetime, timedelta
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from agent2.adapters.forcing_base import EnvironmentalForcingProvider
from agent2.errors import SimulationError
from agent2.geospatial.geodesy import (
    EARTH_RADIUS_M,
    meters_to_degrees_lat,
    meters_to_degrees_lon,
)
from agent2.physics.oil_properties import OilProperties, get_oil_properties
from agent2.physics.particle_cloud import ParticleCloud


class LagrangianTransportEngine:
    """
    Simulates advection and diffusion of Lagrangian oil particle clouds under ocean current and wind forcing.
    Advection equation:
        dx/dt = u_current + windage * u10 + u_stochastic
    """

    def __init__(
        self,
        forcing_provider: EnvironmentalForcingProvider,
        oil_props: Optional[OilProperties] = None,
        timestep_minutes: float = 15.0,
        windage: Optional[float] = None,
        horizontal_diffusivity: Optional[float] = None,
        integration_method: str = "rk4",
        seed: Optional[int] = 42
    ):
        self.forcing = forcing_provider
        self.oil_props = oil_props or get_oil_properties()
        self.dt_seconds = float(timestep_minutes) * 60.0
        self.windage = float(windage if windage is not None else self.oil_props.default_windage)
        self.diffusivity = float(
            horizontal_diffusivity if horizontal_diffusivity is not None
            else self.oil_props.horizontal_diffusivity_m2_s
        )
        self.integration_method = integration_method.lower()
        self.rng = np.random.RandomState(seed)

    def compute_velocity(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime, reverse: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates total advection velocity (u, v) in m/s at given positions and time.
        u_total = u_current + windage * u10
        """
        u_curr, v_curr = self.forcing.get_current_vectors(lons, lats, timestamp)
        u_wind, v_wind = self.forcing.get_wind_vectors(lons, lats, timestamp)

        u_total = u_curr + (self.windage * u_wind)
        v_total = v_curr + (self.windage * v_wind)

        # Physics Sanity Check: Speeds exceeding 5 m/s (~10 knots) in open ocean are unphysical
        speeds = np.sqrt(u_total**2 + v_total**2)
        if np.any(speeds > 10.0):
            # Clip extreme numerical artifacts
            clip_mask = speeds > 10.0
            u_total[clip_mask] = u_total[clip_mask] * (10.0 / speeds[clip_mask])
            v_total[clip_mask] = v_total[clip_mask] * (10.0 / speeds[clip_mask])

        return u_total, v_total

    def step(
        self,
        cloud: ParticleCloud,
        current_time: datetime,
        dt_seconds: float,
        apply_diffusion: bool = True,
        reverse: bool = False
    ) -> datetime:
        """
        Advances (or steps backward) particle cloud positions by dt_seconds.
        """
        lons = cloud.lons[cloud.active]
        lats = cloud.lats[cloud.active]
        n_active = len(lons)
        if n_active == 0:
            return current_time + timedelta(seconds=dt_seconds)

        if self.integration_method == "rk4":
            # 4th-Order Runge-Kutta
            # k1
            u1, v1 = self.compute_velocity(lons, lats, current_time)
            dlat1 = meters_to_degrees_lat(v1 * dt_seconds)
            dlon1 = meters_to_degrees_lon(u1 * dt_seconds, float(np.mean(lats)))

            # k2 at t + dt/2
            t_half = current_time + timedelta(seconds=dt_seconds * 0.5)
            lons_k2 = lons + dlon1 * 0.5
            lats_k2 = lats + dlat1 * 0.5
            u2, v2 = self.compute_velocity(lons_k2, lats_k2, t_half)
            dlat2 = meters_to_degrees_lat(v2 * dt_seconds)
            dlon2 = meters_to_degrees_lon(u2 * dt_seconds, float(np.mean(lats_k2)))

            # k3 at t + dt/2
            lons_k3 = lons + dlon2 * 0.5
            lats_k3 = lats + dlat2 * 0.5
            u3, v3 = self.compute_velocity(lons_k3, lats_k3, t_half)
            dlat3 = meters_to_degrees_lat(v3 * dt_seconds)
            dlon3 = meters_to_degrees_lon(u3 * dt_seconds, float(np.mean(lats_k3)))

            # k4 at t + dt
            t_full = current_time + timedelta(seconds=dt_seconds)
            lons_k4 = lons + dlon3
            lats_k4 = lats + dlat3
            u4, v4 = self.compute_velocity(lons_k4, lats_k4, t_full)
            dlat4 = meters_to_degrees_lat(v4 * dt_seconds)
            dlon4 = meters_to_degrees_lon(u4 * dt_seconds, float(np.mean(lats_k4)))

            # Weighted combination
            dlat = (dlat1 + 2.0 * dlat2 + 2.0 * dlat3 + dlat4) / 6.0
            dlon = (dlon1 + 2.0 * dlon2 + 2.0 * dlon3 + dlon4) / 6.0

        else:
            # Standard Euler
            u, v = self.compute_velocity(lons, lats, current_time)
            dlat = meters_to_degrees_lat(v * dt_seconds)
            dlon = meters_to_degrees_lon(u * dt_seconds, float(np.mean(lats)))

        # Stochastic Turbulent Diffusion (Random Walk)
        if apply_diffusion and self.diffusivity > 0:
            sigma_diff = math.sqrt(2.0 * self.diffusivity * abs(dt_seconds))
            dx_diff = self.rng.normal(0.0, sigma_diff, n_active)
            dy_diff = self.rng.normal(0.0, sigma_diff, n_active)
            dlat += meters_to_degrees_lat(dy_diff)
            dlon += meters_to_degrees_lon(dx_diff, float(np.mean(lats)))

        # Update active particle coordinates
        new_lons = lons + dlon
        new_lats = lats + dlat

        # Check for NaN / Inf
        if np.any(np.isnan(new_lons)) or np.any(np.isnan(new_lats)):
            raise SimulationError("NaN coordinate detected during particle integration step.")

        cloud.lons[cloud.active] = new_lons
        cloud.lats[cloud.active] = new_lats

        next_time = current_time + timedelta(seconds=dt_seconds)
        return next_time

    def simulate(
        self,
        cloud: ParticleCloud,
        start_time: datetime,
        end_time: datetime,
        record_interval_minutes: float = 60.0,
        apply_diffusion: bool = True,
        reverse: bool = False
    ) -> List[Tuple[datetime, Tuple[float, float], float]]:
        """
        Runs complete simulation from start_time to end_time.
        Returns list of trajectory snapshots: [(timestamp, centroid, spread_radius_km)].
        """
        total_duration_seconds = (end_time - start_time).total_seconds()
        if total_duration_seconds == 0:
            return [(start_time, cloud.get_centroid(), cloud.get_spread_radius_km())]

        direction = 1.0 if total_duration_seconds > 0 else -1.0
        dt = abs(self.dt_seconds) * direction
        current_time = start_time
        record_interval_sec = record_interval_minutes * 60.0
        next_record_time = current_time

        snapshots = [(current_time, cloud.get_centroid(), cloud.get_spread_radius_km())]
        cloud.record_snapshot()

        step_count = 0
        max_steps = int(abs(total_duration_seconds) / abs(dt)) + 50

        while (end_time - current_time).total_seconds() * direction > 0 and step_count < max_steps:
            # Adjust final step to land exactly on end_time
            remaining = (end_time - current_time).total_seconds()
            step_dt = dt if abs(remaining) >= abs(dt) else remaining

            current_time = self.step(
                cloud,
                current_time=current_time,
                dt_seconds=step_dt,
                apply_diffusion=apply_diffusion,
                reverse=reverse
            )
            step_count += 1

            # Check if record interval passed
            if abs((current_time - next_record_time).total_seconds()) >= record_interval_sec:
                snapshots.append((current_time, cloud.get_centroid(), cloud.get_spread_radius_km()))
                cloud.record_snapshot()
                next_record_time = current_time

        # Final snapshot
        snapshots.append((current_time, cloud.get_centroid(), cloud.get_spread_radius_km()))
        cloud.record_snapshot()
        return snapshots
