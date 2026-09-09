"""
Geospatial and cartographic package for Agent 2.
"""

from agent2.geospatial.geodesy import (
    EARTH_RADIUS_KM,
    EARTH_RADIUS_M,
    haversine_distance_km,
    meters_to_degrees_lat,
    meters_to_degrees_lon,
    degrees_to_meters_lat,
    degrees_to_meters_lon,
    compute_bearing_deg,
    destination_point,
    approximate_polygon_area_km2,
    buffer_points_to_envelope,
)
from agent2.geospatial.geojson_export import (
    create_feature,
    create_feature_collection,
    build_unified_geojson,
)

__all__ = [
    "EARTH_RADIUS_KM",
    "EARTH_RADIUS_M",
    "haversine_distance_km",
    "meters_to_degrees_lat",
    "meters_to_degrees_lon",
    "degrees_to_meters_lat",
    "degrees_to_meters_lon",
    "compute_bearing_deg",
    "destination_point",
    "approximate_polygon_area_km2",
    "buffer_points_to_envelope",
    "create_feature",
    "create_feature_collection",
    "build_unified_geojson",
]
