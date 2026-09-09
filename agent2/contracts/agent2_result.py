"""
Unified data contract for Agent 2 output:
Consolidates hindcasting, forecasting, metadata, diagnostics, and GeoJSON outputs.
Ready for consumption by future Agent 3 (Vessel Attribution).
"""

from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field, field_validator

from agent2.contracts.hindcast_result import HindcastResult
from agent2.contracts.forecast_result import ForecastResult


class Agent2Result(BaseModel):
    """
    Root output container for Agent 2: Hindcasting and Forecasting Pipeline.
    """
    analysis_id: str = Field(..., description="Unique analysis execution identifier")
    agent_version: str = Field(default="2.0.0", description="Agent 2 software version")
    execution_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when Agent 2 analysis was completed in UTC"
    )
    input_reference: dict[str, Any] = Field(
        ..., description="Key parameters of the input SpillObservation from Agent 1"
    )
    hindcast: HindcastResult = Field(
        ..., description="Hindcast analysis results including probable origin region and ranked hypotheses"
    )
    forecast: ForecastResult = Field(
        ..., description="Forecast analysis results including multi-horizon spread envelopes"
    )
    forcing_summary: dict[str, Any] = Field(
        default_factory=dict, description="Metadata and provenance of environmental datasets used"
    )
    diagnostics: dict[str, Any] = Field(
        default_factory=dict, description="Execution runtimes, particle counts, seed, and quality warnings"
    )
    geojson_collection: dict[str, Any] = Field(
        default_factory=dict, description="Unified GIS-compliant GeoJSON FeatureCollection"
    )

    @field_validator("execution_timestamp")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)
