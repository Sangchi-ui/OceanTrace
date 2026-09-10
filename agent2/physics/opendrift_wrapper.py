"""
OpenDrift / OpenOil integration wrapper.
Detects installed OpenDrift environment and coordinates with internal transport engine.
Ensures graceful fallback and adherence to the OceanTrace architecture.
"""

from datetime import datetime
import logging
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

from agent2.adapters.forcing_base import EnvironmentalForcingProvider
from agent2.physics.oil_properties import OilProperties
from agent2.physics.particle_cloud import ParticleCloud
from agent2.physics.transport_engine import LagrangianTransportEngine
from agent2.physics.simulation_output import SimulationOutput, ParticleFrame, WeatheringState

logger = logging.getLogger("OceanTrace.Agent2.OpenDriftWrapper")

try:
    import opendrift
    from opendrift.models.openoil import OpenOil
    from opendrift.readers.basereader import BaseReader, ContinuousReader
    OPENDRIFT_AVAILABLE = True
except ImportError:
    OPENDRIFT_AVAILABLE = False


class OpenDriftEngineWrapper:
    """
    Coordinates transport simulation between OpenDrift/OpenOil and OceanTrace's
    built-in high-precision Lagrangian transport engine.
    """

    def __init__(
        self,
        forcing_provider: EnvironmentalForcingProvider,
        oil_props: Optional[OilProperties] = None,
        use_opendrift_if_available: bool = True,
        timestep_minutes: float = 15.0,
        windage: Optional[float] = None,
        seed: Optional[int] = 42
    ):
        self.forcing = forcing_provider
        self.oil_props = oil_props
        self.opendrift_available = OPENDRIFT_AVAILABLE and use_opendrift_if_available
        self.timestep_minutes = timestep_minutes
        self.windage = windage
        self.seed = seed

        # Always initialize internal transport engine for guaranteed execution and candidate replay
        self.internal_engine = LagrangianTransportEngine(
            forcing_provider=forcing_provider,
            oil_props=oil_props,
            timestep_minutes=timestep_minutes,
            windage=windage,
            integration_method="rk4",
            seed=seed
        )

    def is_opendrift_active(self) -> bool:
        return self.opendrift_available

    def run_transport(
        self,
        cloud: ParticleCloud,
        start_time: datetime,
        end_time: datetime,
        apply_diffusion: bool = True,
        reverse: bool = False
    ) -> SimulationOutput:
        """
        Executes transport simulation.
        If OpenDrift is available, runs OpenOil and extracts weathering + trajectory.
        Otherwise, falls back to the native RK4 engine.
        Returns a unified SimulationOutput object.
        """
        if not self.opendrift_available:
            # Fallback to internal native engine
            snapshots = self.internal_engine.simulate(
                cloud=cloud,
                start_time=start_time,
                end_time=end_time,
                record_interval_minutes=60.0,
                apply_diffusion=apply_diffusion,
                reverse=reverse
            )
            frames = []
            for state in cloud.history:
                frames.append(ParticleFrame(
                    timestamp=state.timestamp,
                    lons=state.lons.tolist(),
                    lats=state.lats.tolist(),
                    active=state.active.tolist()
                ))
            return SimulationOutput(
                frames=frames,
                engine_used="rk4_native",
                weathering_timeseries=None
            )

        # OpenDrift Implementation
        o = OpenOil(loglevel=30)
        
        # We need a custom reader to bridge EnvironmentalForcingProvider -> OpenDrift
        class OceanTraceForcingReader(ContinuousReader):
            name = "OceanTraceForcing"
            variables = ['x_sea_water_velocity', 'y_sea_water_velocity', 'x_wind', 'y_wind']
            def __init__(self, forcing: EnvironmentalForcingProvider):
                super().__init__()
                self.forcing = forcing
            def get_variables(self, requested_variables, time=None, x=None, y=None, z=None):
                if time is None or x is None or y is None: return None
                res = {}
                u, v = self.forcing.get_current_vectors(x, y, time)
                wu, wv = self.forcing.get_wind_vectors(x, y, time)
                if 'x_sea_water_velocity' in requested_variables: res['x_sea_water_velocity'] = u
                if 'y_sea_water_velocity' in requested_variables: res['y_sea_water_velocity'] = v
                if 'x_wind' in requested_variables: res['x_wind'] = wu
                if 'y_wind' in requested_variables: res['y_wind'] = wv
                return res
        
        # NOTE: For simplicity, we are passing the environment reader. In a real deployment, 
        # OpenDrift may require more setup.
        try:
            reader = OceanTraceForcingReader(self.forcing)
            o.add_reader(reader)
        except Exception as e:
            logger.warning(f"Failed to setup OpenDrift reader, falling back: {e}")
            return self._run_native_fallback(cloud, start_time, end_time, apply_diffusion, reverse)

        # Seed elements
        o.seed_elements(
            lon=cloud.lons[cloud.active],
            lat=cloud.lats[cloud.active],
            time=start_time,
            oil_type=self.oil_props.name if self.oil_props else "GENERIC",
            wind_drift_factor=self.windage if self.windage else 0.03
        )
        
        o.run(
            end_time=end_time,
            time_step=timedelta(minutes=self.timestep_minutes) if not reverse else timedelta(minutes=-self.timestep_minutes),
            time_step_output=timedelta(minutes=60.0),
            outfile=None
        )
        
        # Extract results into SimulationOutput
        history = o.history
        time_steps = o.get_time_array()
        
        frames = []
        for i, t in enumerate(time_steps):
            lons = history['lon'][:, i]
            lats = history['lat'][:, i]
            status = history['status'][:, i]
            
            # 0 is active in OpenDrift
            active = (status == 0)
            
            frames.append(ParticleFrame(
                timestamp=t,
                lons=lons.tolist(),
                lats=lats.tolist(),
                active=active.tolist()
            ))
            
        weathering = []
        for i, t in enumerate(time_steps):
            weathering.append(WeatheringState(
                timestamp=t,
                evaporated_fraction=float(np.mean(history.get('mass_evaporated', np.zeros_like(history['lon']))[:, i])) if 'mass_evaporated' in history else 0.0,
                water_content=float(np.mean(history.get('water_fraction', np.zeros_like(history['lon']))[:, i])) if 'water_fraction' in history else 0.0,
                surface_oil_fraction=float(np.mean(history.get('mass_surface', np.ones_like(history['lon']))[:, i])) if 'mass_surface' in history else 1.0
            ))

        return SimulationOutput(
            frames=frames,
            engine_used="opendrift_openoil",
            weathering_timeseries=weathering
        )

    def _run_native_fallback(self, cloud, start_time, end_time, apply_diffusion, reverse) -> SimulationOutput:
        self.internal_engine.simulate(
            cloud=cloud,
            start_time=start_time,
            end_time=end_time,
            record_interval_minutes=60.0,
            apply_diffusion=apply_diffusion,
            reverse=reverse
        )
        frames = [ParticleFrame(
            timestamp=state.timestamp,
            lons=state.lons.tolist(),
            lats=state.lats.tolist(),
            active=state.active.tolist()
        ) for state in cloud.history]
        return SimulationOutput(
            frames=frames,
            engine_used="rk4_native",
            weathering_timeseries=None
        )
