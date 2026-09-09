"""
Adapter for consuming Agent 1 outputs (SpillEvent objects, spill_event.json, or spill.geojson)
and translating them into validated, normalized SpillObservation instances for Agent 2.
Strictly decoupled from Agent 1 internals; operates entirely on public contract schemas.
"""

from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Union
from shapely.geometry import shape, Polygon, MultiPolygon

from agent2.contracts.spill_observation import SpillObservation
from agent2.errors import Agent1ContractError, GeometryValidationError
from agent2.geospatial.geodesy import approximate_polygon_area_km2


class Agent1Adapter:
    """
    Translates Agent 1 detection products into Agent 2 SpillObservation contracts.
    Enforces geographic validity and timezone-aware UTC timestamps.
    """

    @classmethod
    def parse(cls, data: Union[Dict[str, Any], Any], region_index: int = 0) -> SpillObservation:
        """
        Generic parser that accepts a dict or Pydantic SpillEvent and delegates appropriately.
        """
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        elif hasattr(data, "dict"):
            data = data.dict()

        if not isinstance(data, dict):
            raise Agent1ContractError(f"Expected dict or model payload, received: {type(data)}")

        # Check if it's a FeatureCollection (spill.geojson format)
        if data.get("type") == "FeatureCollection":
            return cls._from_geojson_collection(data, feature_index=region_index)

        # Otherwise assume it is a SpillEvent structure
        return cls._from_spill_event_dict(data, region_index=region_index)

    @classmethod
    def from_file(cls, file_path: str, region_index: int = 0) -> SpillObservation:
        """
        Loads an Agent 1 output file (.json or .geojson) and parses into SpillObservation.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Agent 1 output file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.parse(data, region_index=region_index)

    @classmethod
    def from_all_regions(cls, data: Union[Dict[str, Any], Any]) -> List[SpillObservation]:
        """
        Parses all detected spill regions within a multi-slick SpillEvent into a list of SpillObservations.
        """
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        elif hasattr(data, "dict"):
            data = data.dict()

        if data.get("type") == "FeatureCollection":
            features = data.get("features", [])
            observations = []
            for idx in range(len(features)):
                try:
                    obs = cls._from_geojson_collection(data, feature_index=idx)
                    observations.append(obs)
                except GeometryValidationError:
                    continue
            return observations

        regions = data.get("spill_regions", [])
        observations = []
        for idx in range(len(regions)):
            try:
                obs = cls._from_spill_event_dict(data, region_index=idx)
                observations.append(obs)
            except GeometryValidationError:
                continue
        return observations

    @classmethod
    def _from_spill_event_dict(cls, event_dict: Dict[str, Any], region_index: int = 0) -> SpillObservation:
        """Parses a dictionary representing an Agent 1 SpillEvent schema."""
        regions = event_dict.get("spill_regions", [])
        if not regions:
            raise Agent1ContractError("No spill regions found in Agent 1 SpillEvent payload.")

        if region_index >= len(regions):
            raise IndexError(f"Requested region index {region_index} exceeds available regions ({len(regions)})")

        region = regions[region_index]
        geometry_meta = region.get("geometry", {})
        measurements = region.get("measurements", {})
        detection = region.get("detection", {})
        source_meta = event_dict.get("source", {})

        # Strict Georeference Validation (No Fabrication Policy)
        is_geo = geometry_meta.get("is_geographic", False)
        if not is_geo:
            raise GeometryValidationError(
                f"Spill region {region.get('spill_id', 'unknown')} is not georeferenced. "
                f"Agent 2 cannot simulate ocean drift in pixel space without inventing coordinates."
            )

        poly_geojson = geometry_meta.get("polygon_geojson")
        if not poly_geojson:
            raise GeometryValidationError(
                f"Missing polygon_geojson geometry for spill region {region.get('spill_id')}"
            )

        # Extract centroid coordinates
        centroid_dict = geometry_meta.get("centroid", {})
        lon = centroid_dict.get("longitude")
        lat = centroid_dict.get("latitude")

        if lon is None or lat is None:
            # Fallback to computing from polygon geometry
            shapely_geom = shape(poly_geojson)
            lon = float(shapely_geom.centroid.x)
            lat = float(shapely_geom.centroid.y)

        # Extract or compute area
        area_km2 = measurements.get("area_km2")
        if area_km2 is None or area_km2 <= 0:
            shapely_geom = shape(poly_geojson)
            area_km2 = approximate_polygon_area_km2(shapely_geom)
            if area_km2 <= 0:
                raise GeometryValidationError(f"Calculated area for {region.get('spill_id')} is non-positive.")

        # Timestamp Parsing and Normalization to UTC
        time_str = (
            source_meta.get("acquisition_time") or
            event_dict.get("timestamp") or
            datetime.now(timezone.utc).isoformat()
        )
        obs_time = cls._parse_timestamp_to_utc(time_str)

        # Orientation / Axis properties
        aspect_ratio = measurements.get("aspect_ratio")
        w_px = measurements.get("width_pixels", 1.0)
        h_px = measurements.get("height_pixels", 1.0)
        orientation_deg = 0.0
        if w_px > 0 and h_px > 0:
            # Orientation estimate from aspect ratio
            orientation_deg = 90.0 if h_px > w_px else 0.0

        confidence = float(detection.get("confidence", 0.5))

        return SpillObservation(
            spill_id=str(region.get("spill_id", f"SPILL_{region_index:03d}")),
            observation_timestamp=obs_time,
            geometry=poly_geojson,
            centroid=(round(float(lon), 6), round(float(lat), 6)),
            area_km2=round(float(area_km2), 4),
            perimeter_km=measurements.get("perimeter_km"),
            major_axis_km=None,
            minor_axis_km=None,
            orientation_deg=orientation_deg,
            compactness=None,
            confidence=round(confidence, 4),
            crs=geometry_meta.get("crs", "EPSG:4326") or "EPSG:4326",
            source_satellite=str(source_meta.get("satellite", "Sentinel-1")),
            source_sensor=str(source_meta.get("sensor", "SAR")),
            source_image_path=source_meta.get("image_path"),
            raw_metadata={
                "event_id": event_dict.get("event_id"),
                "model": detection.get("model", "UNet"),
                "threshold": detection.get("threshold", 0.5),
                "validation_status": region.get("validation", {}).get("status", "UNKNOWN"),
                "validation_warnings": region.get("validation", {}).get("warnings", [])
            }
        )

    @classmethod
    def _from_geojson_collection(
        cls, geojson_data: Dict[str, Any], feature_index: int = 0
    ) -> SpillObservation:
        """Parses a GeoJSON FeatureCollection (spill.geojson)."""
        features = geojson_data.get("features", [])
        if not features:
            raise Agent1ContractError("Empty GeoJSON FeatureCollection provided.")

        if feature_index >= len(features):
            raise IndexError(f"Feature index {feature_index} exceeds features count ({len(features)})")

        feat = features[feature_index]
        props = feat.get("properties", {})
        geom_dict = feat.get("geometry", {})

        is_geo = props.get("is_geographic", True)
        if not is_geo:
            raise GeometryValidationError("GeoJSON feature is flagged as non-geographic.")

        shapely_geom = shape(geom_dict)
        if shapely_geom.is_empty:
            raise GeometryValidationError("GeoJSON geometry is empty.")

        lon = props.get("centroid_lon")
        lat = props.get("centroid_lat")
        if lon is None or lat is None:
            lon = float(shapely_geom.centroid.x)
            lat = float(shapely_geom.centroid.y)

        area_km2 = props.get("area_km2")
        if area_km2 is None or float(area_km2) <= 0:
            area_km2 = approximate_polygon_area_km2(shapely_geom)

        time_str = props.get("timestamp") or datetime.now(timezone.utc).isoformat()
        obs_time = cls._parse_timestamp_to_utc(time_str)

        return SpillObservation(
            spill_id=str(props.get("spill_id", f"SPILL_{feature_index:03d}")),
            observation_timestamp=obs_time,
            geometry=geom_dict,
            centroid=(round(float(lon), 6), round(float(lat), 6)),
            area_km2=round(float(area_km2), 4),
            perimeter_km=props.get("perimeter_km"),
            confidence=float(props.get("confidence", 0.8)),
            crs=props.get("crs", "EPSG:4326"),
            raw_metadata=props
        )

    @staticmethod
    def _parse_timestamp_to_utc(time_str: str) -> datetime:
        """Parses ISO timestamp string and enforces timezone.utc."""
        try:
            # Handle trailing 'Z'
            if time_str.endswith("Z"):
                time_str = time_str[:-1] + "+00:00"
            dt = datetime.fromisoformat(time_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)
