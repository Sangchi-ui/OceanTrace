"""
Data contract representing an observed oil spill extracted from Agent 1 or external detection.
Normalizes geospatial coordinates, timestamps (UTC), geometries, and quantitative metrics.
"""

from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field, field_validator


class SpillObservation(BaseModel):
    """
    Validated internal representation of an observed oil spill.
    All timestamps must be UTC timezone-aware.
    Coordinates are in EPSG:4326 (lon, lat degrees) unless specified.
    """
    spill_id: str = Field(..., description="Unique spill region identifier")
    observation_timestamp: datetime = Field(
        ..., description="Timestamp of satellite acquisition / detection in UTC"
    )
    geometry: dict[str, Any] = Field(
        ..., description="GeoJSON geometry object representing the spill outline (Polygon or MultiPolygon)"
    )
    centroid: tuple[float, float] = Field(
        ..., description="Geographic centroid coordinates (longitude, latitude) in EPSG:4326"
    )
    area_km2: float = Field(..., gt=0.0, description="Observed slick area in square kilometers")
    perimeter_km: float | None = Field(default=None, description="Observed slick perimeter in kilometers")
    major_axis_km: float | None = Field(default=None, description="Length of the major axis in km")
    minor_axis_km: float | None = Field(default=None, description="Length of the minor axis in km")
    orientation_deg: float | None = Field(default=None, description="Orientation angle of slick in degrees")
    compactness: float | None = Field(default=None, description="Compactness metric (4 * pi * area / perimeter^2)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score [0.0, 1.0]")
    crs: str = Field(default="EPSG:4326", description="Coordinate Reference System")
    source_satellite: str = Field(default="Sentinel-1", description="Source satellite constellation")
    source_sensor: str = Field(default="SAR", description="Sensor mode/type")
    source_image_path: str | None = Field(default=None, description="Path or URI to source image")
    raw_metadata: dict[str, Any] = Field(default_factory=dict, description="Additional source or quality flags")

    @field_validator("observation_timestamp")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @property
    def lon(self) -> float:
        return self.centroid[0]

    @property
    def lat(self) -> float:
        return self.centroid[1]
