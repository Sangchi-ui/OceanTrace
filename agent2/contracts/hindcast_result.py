"""
Data contracts representing the outputs of Agent 2 Hindcasting:
Candidate hypotheses, backward envelope, forward replay comparisons, and estimated origin region.
"""

from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field, field_validator


class TrajectoryPoint(BaseModel):
    """A single spatial-temporal coordinate along a simulated drift trajectory."""
    timestamp: datetime = Field(..., description="Timestamp of point in UTC")
    longitude: float = Field(..., description="Longitude in EPSG:4326")
    latitude: float = Field(..., description="Latitude in EPSG:4326")

    @field_validator("timestamp")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)


class CandidateHypothesis(BaseModel):
    """
    A specific candidate release origin hypothesis tested via forward replay.
    """
    candidate_id: str = Field(..., description="Unique identifier for candidate hypothesis")
    release_timestamp: datetime = Field(..., description="Hypothesized release timestamp in UTC")
    origin_coordinates: tuple[float, float] = Field(
        ..., description="Hypothesized release coordinate (longitude, latitude)"
    )
    simulated_final_centroid: tuple[float, float] = Field(
        ..., description="Centroid (lon, lat) of simulated spill at observed time"
    )
    simulated_final_area_km2: float = Field(
        ..., description="Area of simulated particle cloud at observed time in km^2"
    )
    centroid_distance_km: float = Field(
        ..., description="Geodesic distance between simulated and observed centroids in km"
    )
    iou_score: float = Field(..., ge=0.0, le=1.0, description="Intersection-over-Union spatial overlap score")
    centroid_score: float = Field(..., ge=0.0, le=1.0, description="Gaussian distance agreement score")
    area_score: float = Field(..., ge=0.0, le=1.0, description="Area agreement score")
    shape_score: float = Field(..., ge=0.0, le=1.0, description="Aspect ratio / shape similarity score")
    composite_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall weighted source hypothesis score"
    )
    trajectory: list[TrajectoryPoint] = Field(
        default_factory=list, description="Forward drift trajectory from candidate origin to observation time"
    )


class OriginRegion(BaseModel):
    """
    Probable origin region and time window estimated from ranked candidate hypotheses.
    """
    geometry_geojson: dict[str, Any] = Field(
        ..., description="GeoJSON polygon or MultiPolygon delineating the probable origin zone"
    )
    bounding_box: list[float] = Field(
        ..., description="Bounding box [min_lon, min_lat, max_lon, max_lat]"
    )
    centroid: tuple[float, float] = Field(
        ..., description="Center of mass of origin region (longitude, latitude)"
    )
    area_km2: float = Field(..., description="Estimated area of origin region in km^2")
    time_window_start: datetime = Field(..., description="Earliest plausible release time (UTC)")
    time_window_end: datetime = Field(..., description="Latest plausible release time (UTC)")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Overall confidence in the identified origin region"
    )
    uncertainty_radius_km: float = Field(
        ..., description="Estimated 1-sigma spatial uncertainty radius in km"
    )

    @field_validator("time_window_start", "time_window_end")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)


class HindcastResult(BaseModel):
    """
    Complete structured hindcast analysis result.
    """
    enabled: bool = Field(default=True, description="Whether hindcasting was executed")
    status: str = Field(default="SUCCESS", description="Execution status: SUCCESS, PARTIAL, or FAILED")
    probable_origin: OriginRegion | None = Field(
        default=None, description="Estimated probable origin region and time window"
    )
    candidate_hypotheses: list[CandidateHypothesis] = Field(
        default_factory=list, description="All tested candidate hypotheses sorted by score descending"
    )
    best_candidate: CandidateHypothesis | None = Field(
        default=None, description="Highest-scoring candidate hypothesis"
    )
    historical_envelope_geojson: dict[str, Any] | None = Field(
        default=None, description="Stage A backward search envelope polygon"
    )
    diagnostics: dict[str, Any] = Field(
        default_factory=dict, description="Simulation diagnostics and tuning parameters"
    )
