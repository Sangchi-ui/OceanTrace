"""
Data contracts representing the outputs of Agent 2 Forecasting:
Multi-horizon forecasts (+6h, +12h, +24h), ensemble spread envelopes, centroids, and uncertainty bounds.
"""

from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field, field_validator
from agent2.contracts.hindcast_result import TrajectoryPoint


class EnsembleTrajectory(BaseModel):
    """Trajectory for a single member of the environmental perturbation ensemble."""
    member_id: int = Field(..., description="Ensemble member index")
    perturbation_description: str = Field(..., description="Description of forcing perturbation applied")
    trajectory: list[TrajectoryPoint] = Field(..., description="Simulated points over forecast timeline")


class ForecastHorizonResult(BaseModel):
    """
    Spill dispersion state at a specific forecast horizon (e.g. +6h, +12h, +24h).
    """
    horizon_hours: int = Field(..., description="Forecast horizon offset from observation in hours")
    forecast_timestamp: datetime = Field(..., description="Valid timestamp of forecast in UTC")
    centroid: tuple[float, float] = Field(
        ..., description="Forecasted center of mass (longitude, latitude) in EPSG:4326"
    )
    drift_distance_km: float = Field(..., description="Net displacement distance from initial centroid in km")
    mean_drift_speed_m_s: float = Field(..., description="Average drift speed in m/s")
    spread_area_km2: float = Field(..., description="Total estimated slick spread area in km^2")
    core_envelope_50_geojson: dict[str, Any] = Field(
        ..., description="GeoJSON polygon of core 50% particle density envelope"
    )
    spread_envelope_90_geojson: dict[str, Any] = Field(
        ..., description="GeoJSON polygon of outer 90% particle dispersion envelope"
    )
    particle_count: int = Field(..., description="Number of active particles tracked at this horizon")
    ensemble_spread_std_km: float = Field(
        ..., description="Standard deviation of ensemble centroids indicating spatial uncertainty in km"
    )
    beached_particle_ratio: float = Field(
        default=0.0, description="Fraction of particles reaching coastline / land boundaries"
    )

    @field_validator("forecast_timestamp")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @property
    def uncertainty_radius_km(self) -> float:
        return self.ensemble_spread_std_km

    @property
    def target_timestamp(self) -> datetime:
        return self.forecast_timestamp

    @property
    def area_km2_core_50(self) -> float:
        return round(self.spread_area_km2 * 0.5, 2)

    @property
    def area_km2_spread_90(self) -> float:
        return self.spread_area_km2


class ForecastResult(BaseModel):
    """
    Complete structured forecast analysis result.
    """
    enabled: bool = Field(default=True, description="Whether forecasting was executed")
    status: str = Field(default="SUCCESS", description="Execution status: SUCCESS, PARTIAL, or FAILED")
    horizons: list[ForecastHorizonResult] = Field(
        default_factory=list, description="Dispersion states at each requested forecast horizon"
    )
    ensemble_trajectories: list[EnsembleTrajectory] = Field(
        default_factory=list, description="Individual ensemble member trajectory tracks"
    )
    diagnostics: dict[str, Any] = Field(
        default_factory=dict, description="Forecast simulation parameters and execution diagnostics"
    )
