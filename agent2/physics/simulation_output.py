from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class ParticleFrame:
    timestamp: datetime
    lons: List[float]
    lats: List[float]
    active: List[bool]

@dataclass
class WeatheringState:
    timestamp: datetime
    evaporated_fraction: float
    water_content: float
    surface_oil_fraction: float

@dataclass
class SimulationOutput:
    frames: List[ParticleFrame]
    engine_used: str
    weathering_timeseries: Optional[List[WeatheringState]] = None
    envelope_50_geojson: Optional[dict] = None
    envelope_90_geojson: Optional[dict] = None
    particle_positions_geojson: Optional[dict] = None
