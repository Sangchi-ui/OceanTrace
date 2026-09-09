"""
Geodetic and spatial calculation utilities for Agent 2.
Calculates high-accuracy spherical/geodesic distances, bearings, destination points,
and metric displacements avoiding naive degree-as-meter operations.
"""

import math
from typing import Tuple, List, Sequence
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point, shape, mapping
from shapely.ops import unary_union

EARTH_RADIUS_KM = 6371.0088  # IUGG mean Earth radius in km
EARTH_RADIUS_M = EARTH_RADIUS_KM * 1000.0


def haversine_distance_km(
    lon1: float, lat1: float, lon2: float, lat2: float
) -> float:
    """
    Computes great-circle distance between two points on Earth using Haversine formula.
    Returns distance in kilometers.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def meters_to_degrees_lat(meters: float) -> float:
    """Converts a displacement in meters along meridian to degrees latitude."""
    return (meters / EARTH_RADIUS_M) * (180.0 / math.pi)


def meters_to_degrees_lon(meters: float, lat_deg: float) -> float:
    """Converts a displacement in meters along parallel to degrees longitude at given latitude."""
    cos_lat = math.cos(math.radians(lat_deg))
    if abs(cos_lat) < 1e-7:
        cos_lat = 1e-7
    return (meters / (EARTH_RADIUS_M * cos_lat)) * (180.0 / math.pi)


def degrees_to_meters_lat(deg_lat: float) -> float:
    """Converts degrees latitude difference to meters."""
    return (deg_lat * math.pi / 180.0) * EARTH_RADIUS_M


def degrees_to_meters_lon(deg_lon: float, lat_deg: float) -> float:
    """Converts degrees longitude difference to meters at given latitude."""
    return (deg_lon * math.pi / 180.0) * (EARTH_RADIUS_M * math.cos(math.radians(lat_deg)))


def compute_bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Computes forward azimuth (bearing) from (lon1, lat1) to (lon2, lat2) in degrees [0, 360)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = (math.cos(phi1) * math.sin(phi2) -
         math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda))
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def destination_point(
    lon: float, lat: float, distance_km: float, bearing_deg: float
) -> Tuple[float, float]:
    """
    Computes destination point given start coordinates, distance in km, and bearing in degrees.
    Returns (dest_lon, dest_lat).
    """
    delta = distance_km / EARTH_RADIUS_KM
    theta = math.radians(bearing_deg)
    phi1 = math.radians(lat)
    lambda1 = math.radians(lon)

    phi2 = math.asin(
        math.sin(phi1) * math.cos(delta) +
        math.cos(phi1) * math.sin(delta) * math.cos(theta)
    )
    lambda2 = lambda1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2)
    )
    return (math.degrees(lambda2), math.degrees(phi2))


def approximate_polygon_area_km2(geom: Polygon | MultiPolygon) -> float:
    """
    Computes geodesic area of a Shapely geometry in square kilometers
    using latitude-adjusted metric projection.
    """
    if geom.is_empty:
        return 0.0

    centroid = geom.centroid
    lat_rad = math.radians(centroid.y)
    cos_lat = math.cos(lat_rad)

    # 1 deg lat in km ~ (pi / 180) * R
    km_per_deg_lat = (math.pi / 180.0) * EARTH_RADIUS_KM
    km_per_deg_lon = km_per_deg_lat * cos_lat

    # Degree area scaled by local metric factors
    area_deg2 = geom.area
    return float(area_deg2 * km_per_deg_lat * km_per_deg_lon)


def buffer_points_to_envelope(
    points_lon_lat: np.ndarray,
    buffer_radius_km: float,
    simplify_tolerance_km: float = 0.05
) -> Polygon | MultiPolygon:
    """
    Creates a geographic envelope polygon around a collection of points (Nx2 array of [lon, lat])
    by buffering each point by buffer_radius_km and taking the unary union.
    """
    if len(points_lon_lat) == 0:
        return Polygon()

    mean_lat = float(np.mean(points_lon_lat[:, 1]))
    buf_deg_lat = meters_to_degrees_lat(buffer_radius_km * 1000.0)
    buf_deg_lon = meters_to_degrees_lon(buffer_radius_km * 1000.0, mean_lat)
    buf_deg = (buf_deg_lat + buf_deg_lon) / 2.0

    # Build shapely points and buffer
    circles = [Point(pt[0], pt[1]).buffer(buf_deg, quad_segs=16) for pt in points_lon_lat]
    merged = unary_union(circles)
    if simplify_tolerance_km > 0:
        sim_deg = meters_to_degrees_lat(simplify_tolerance_km * 1000.0)
        merged = merged.simplify(sim_deg, preserve_topology=True)
    return merged
