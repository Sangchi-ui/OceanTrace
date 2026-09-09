from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class SourceMetadata(BaseModel):
    satellite: str = Field(default="Sentinel-1",
                           description="Satellite constellation name")
    sensor: str = Field(
        default="SAR", description="Sensor type (e.g. SAR C-band)")
    image_path: str = Field(..., description="Path to the source SAR image")
    acquisition_time: str | None = Field(
        default=None, description="ISO-8601 acquisition timestamp")


class DetectionMetadata(BaseModel):
    confidence: float = Field(..., ge=0.0, le=1.0,
                              description="Mean probability score over spill region")
    model: str = Field(default="UNet", description="Model architecture name")
    threshold: float = Field(
        default=0.5, description="Segmentation probability threshold used")


class Centroid(BaseModel):
    latitude: float | None = Field(
        default=None, description="Centroid latitude (Y coordinate in CRS)")
    longitude: float | None = Field(
        default=None, description="Centroid longitude (X coordinate in CRS)")
    pixel_x: float = Field(..., description="Centroid X in pixel coordinates")
    pixel_y: float = Field(..., description="Centroid Y in pixel coordinates")


class GeometryMetadata(BaseModel):
    is_geographic: bool = Field(
        ..., description="True if coordinates are in EPSG/geographic space, False if pixel space")
    crs: str | None = Field(
        default=None, description="Coordinate reference system EPSG string")
    centroid: Centroid
    bbox: list[float] = Field(...,
                              description="Bounding box [min_x, min_y, max_x, max_y]")
    polygon_geojson: dict[str, Any] | None = Field(
        default=None, description="GeoJSON Geometry object of detected spill polygon")


class MeasurementsMetadata(BaseModel):
    area_km2: float | None = Field(
        default=None, description="Geographic area in square kilometers")
    perimeter_km: float | None = Field(
        default=None, description="Geographic perimeter in kilometers")
    area_pixels: int = Field(...,
                             description="Total pixel area of detected slick")
    width_pixels: float = Field(...,
                                description="Bounding box width in pixels")
    height_pixels: float = Field(...,
                                 description="Bounding box height in pixels")
    aspect_ratio: float = Field(...,
                                description="Aspect ratio (width / height)")


class ValidationMetadata(BaseModel):
    status: str = Field(...,
                        description="Validation status: VALID, WARNING, or INVALID")
    warnings: list[str] = Field(
        default_factory=list, description="List of quality warning messages")


class SpillRegionRecord(BaseModel):
    spill_id: str = Field(...,
                          description="Unique region identifier within the image")
    detection: DetectionMetadata
    geometry: GeometryMetadata
    measurements: MeasurementsMetadata
    validation: ValidationMetadata


class SpillEvent(BaseModel):
    event_id: str = Field(..., description="Unique event identifier")
    timestamp: str = Field(default_factory=lambda: datetime.now(
        timezone.utc).isoformat(), description="Processing timestamp")
    source: SourceMetadata
    overall_confidence: float = Field(
        ..., description="Overall confidence aggregated across detected slicks")
    num_spill_regions: int = Field(...,
                                   description="Number of candidate spill regions detected")
    spill_regions: list[SpillRegionRecord] = Field(
        default_factory=list, description="Individual detected spill regions")
    validation: ValidationMetadata = Field(...,
                                           description="Overall image validation status")
