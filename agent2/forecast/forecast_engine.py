"""
Forecasting Engine for Agent 2.
Executes multi-horizon forward dispersion simulations with ensemble perturbations.
"""

from datetime import datetime, timedelta
import math
from typing import Any, Dict, List, Optional
import numpy as np

from agent2.adapters.forcing_base import EnvironmentalForcingProvider
from agent2.contracts.forecast_result import (
    EnsembleTrajectory,
    ForecastHorizonResult,
    ForecastResult,
)
from agent2.contracts.hindcast_result import TrajectoryPoint
from agent2.contracts.spill_observation import SpillObservation
from agent2.forecast.ensemble import EnsembleGenerator, EnsembleMemberConfig
from agent2.forecast.spread_envelope import compute_spread_envelopes
from agent2.physics.oil_properties import OilProperties, get_oil_properties
from agent2.physics.particle_cloud import ParticleCloud
from agent2.physics.transport_engine import LagrangianTransportEngine


class PerturbedForcingWrapper(EnvironmentalForcingProvider):
    """Wraps a forcing provider with scale factor and rotational directional bias."""
    def __init__(
        self,
        base_provider: EnvironmentalForcingProvider,
        config: EnsembleMemberConfig
    ):
        self.base = base_provider
        self.cfg = config
        self._curr_cos = math.cos(math.radians(config.current_angle_deg))
        self._curr_sin = math.sin(math.radians(config.current_angle_deg))
        self._wind_cos = math.cos(math.radians(config.wind_angle_deg))
        self._wind_sin = math.sin(math.radians(config.wind_angle_deg))

    def get_current_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ):
        u, v = self.base.get_current_vectors(lons, lats, timestamp)
        # Apply scaling and 2D rotation
        u_rot = (u * self._curr_cos - v * self._curr_sin) * self.cfg.current_scale
        v_rot = (u * self._curr_sin + v * self._curr_cos) * self.cfg.current_scale
        return u_rot, v_rot

    def get_wind_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ):
        u, v = self.base.get_wind_vectors(lons, lats, timestamp)
        u_rot = (u * self._wind_cos - v * self._wind_sin) * self.cfg.wind_scale
        v_rot = (u * self._wind_sin + v * self._wind_cos) * self.cfg.wind_scale
        return u_rot, v_rot

    def metadata(self) -> Dict[str, Any]:
        return {
            "base": self.base.metadata(),
            "perturbation": self.cfg.description
        }


class ForecastingEngine:
    """
    Simulates forward dispersion trajectories and spread envelopes for specified horizons.
    """

    def __init__(
        self,
        forcing_provider: EnvironmentalForcingProvider,
        oil_props: Optional[OilProperties] = None,
        horizons_hours: Optional[List[int]] = None,
        particle_count: int = 500,
        ensemble_size: int = 10,
        timestep_minutes: float = 15.0,
        seed: Optional[int] = 42
    ):
        self.forcing = forcing_provider
        self.oil_props = oil_props or get_oil_properties()
        self.horizons_hours = sorted(horizons_hours or [6, 12, 24])
        self.particle_count = particle_count
        self.ensemble_size = ensemble_size
        self.timestep_minutes = timestep_minutes
        self.seed = seed

    def run(self, spill: SpillObservation) -> ForecastResult:
        """Executes ensemble forward simulation from the detected spill observation."""
        obs_time = spill.observation_timestamp
        init_centroid = spill.centroid

        members = EnsembleGenerator.generate_members(
            ensemble_size=self.ensemble_size,
            base_windage=self.oil_props.default_windage,
            min_windage=self.oil_props.min_windage,
            max_windage=self.oil_props.max_windage,
            seed=self.seed
        )

        max_horizon_hours = max(self.horizons_hours)
        horizon_timestamps = {h: obs_time + timedelta(hours=h) for h in self.horizons_hours}

        # Store points per horizon: horizon -> list of (lons, lats)
        horizon_particles: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {
            h: [] for h in self.horizons_hours
        }

        ensemble_trajectories: List[EnsembleTrajectory] = []

        # Run each ensemble member
        for m_idx, member in enumerate(members):
            p_provider = PerturbedForcingWrapper(self.forcing, member)
            engine = LagrangianTransportEngine(
                forcing_provider=p_provider,
                oil_props=self.oil_props,
                timestep_minutes=self.timestep_minutes,
                windage=member.windage,
                horizontal_diffusivity=self.oil_props.horizontal_diffusivity_m2_s * member.diffusivity_factor,
                seed=(self.seed + m_idx if self.seed else None)
            )

            # Particles per ensemble member
            n_parts = max(self.particle_count // max(self.ensemble_size, 1), 30)
            cloud = ParticleCloud.from_polygon(
                polygon_geom=spill.geometry,
                num_particles=n_parts,
                seed=(self.seed + m_idx * 100 if self.seed else None)
            )

            traj_points: List[TrajectoryPoint] = [
                TrajectoryPoint(
                    timestamp=obs_time,
                    longitude=init_centroid[0],
                    latitude=init_centroid[1]
                )
            ]

            curr_t = obs_time
            for h in self.horizons_hours:
                target_t = horizon_timestamps[h]
                engine.simulate(
                    cloud=cloud,
                    start_time=curr_t,
                    end_time=target_t,
                    apply_diffusion=True
                )
                curr_t = target_t

                # Record points at this horizon
                horizon_particles[h].append((cloud.lons.copy(), cloud.lats.copy()))

                c_lon, c_lat = cloud.get_centroid()
                traj_points.append(TrajectoryPoint(
                    timestamp=target_t,
                    longitude=c_lon,
                    latitude=c_lat
                ))

            ensemble_trajectories.append(EnsembleTrajectory(
                member_id=member.member_id,
                perturbation_description=member.description,
                trajectory=traj_points
            ))

        # Build ForecastHorizonResult for each horizon
        horizon_results: List[ForecastHorizonResult] = []
        for h in self.horizons_hours:
            member_clouds = horizon_particles[h]
            all_lons = np.concatenate([pair[0] for pair in member_clouds])
            all_lats = np.concatenate([pair[1] for pair in member_clouds])

            env_metrics = compute_spread_envelopes(
                all_lons=all_lons,
                all_lats=all_lats,
                initial_centroid=init_centroid,
                horizon_hours=h
            )

            horizon_results.append(ForecastHorizonResult(
                horizon_hours=h,
                forecast_timestamp=horizon_timestamps[h],
                centroid=env_metrics["centroid"],
                drift_distance_km=env_metrics["drift_distance_km"],
                mean_drift_speed_m_s=env_metrics["mean_drift_speed_m_s"],
                spread_area_km2=env_metrics["spread_area_km2"],
                core_envelope_50_geojson=env_metrics["core_50"],
                spread_envelope_90_geojson=env_metrics["spread_90"],
                particle_count=len(all_lons),
                ensemble_spread_std_km=env_metrics["ensemble_std_km"],
                beached_particle_ratio=0.0
            ))

        # Run OpenDrift Forward simulation for animation frames and weathering
        from agent2.physics.opendrift_wrapper import OpenDriftEngineWrapper
        wrapper = OpenDriftEngineWrapper(
            forcing_provider=self.forcing,
            oil_props=self.oil_props,
            timestep_minutes=self.timestep_minutes,
            seed=self.seed
        )
        # Use full cloud for the open drift run
        od_cloud = ParticleCloud.from_polygon(
            polygon_geom=spill.geometry,
            num_particles=self.particle_count,
            seed=self.seed
        )
        sim_out = wrapper.run_transport(
            cloud=od_cloud,
            start_time=obs_time,
            end_time=horizon_timestamps[max_horizon_hours],
            apply_diffusion=True,
            reverse=False
        )

        frames_dict = []
        for frame in sim_out.frames:
            frames_dict.append({
                "t": frame.timestamp.isoformat() + "Z",
                "phase": "forecast",
                "particles": [[lon, lat] for lon, lat, active in zip(frame.lons, frame.lats, frame.active) if active]
            })
            
        weathering_list = None
        if sim_out.weathering_timeseries:
            weathering_list = []
            for w in sim_out.weathering_timeseries:
                weathering_list.append({
                    "t": w.timestamp.isoformat() + "Z",
                    "evaporated_fraction": w.evaporated_fraction,
                    "water_content": w.water_content,
                    "surface_oil_fraction": w.surface_oil_fraction
                })

        return ForecastResult(
            enabled=True,
            status="SUCCESS",
            horizons=horizon_results,
            ensemble_trajectories=ensemble_trajectories,
            animation_frames=frames_dict,
            weathering_timeseries=weathering_list,
            engine_used=sim_out.engine_used,
            opendrift_available=wrapper.opendrift_available,
            diagnostics={
                "particle_count_total": sum(len(pair[0]) for pair in horizon_particles[self.horizons_hours[0]]),
                "ensemble_size": len(members),
                "horizons": self.horizons_hours,
                "timestep_minutes": self.timestep_minutes,
                "oil_category": self.oil_props.category.value
            }
        )

