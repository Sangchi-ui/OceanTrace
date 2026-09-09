"""
Spread Envelope and spatial uncertainty delineation for forecast horizons.
Generates core (50%) and spread (90%) geographic polygons from particle ensemble clusters.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from shapely.geometry import mapping, Polygon, MultiPolygon

from agent2.geospatial.geodesy import (
    approximate_polygon_area_km2,
    buffer_points_to_envelope,
    haversine_distance_km,
)


def compute_spread_envelopes(
    all_lons: np.ndarray,
    all_lats: np.ndarray,
    initial_centroid: Tuple[float, float],
    horizon_hours: int,
    base_buffer_km: float = 0.6
) -> Dict[str, Any]:
    """
    Computes statistical spread metrics, core 50% polygon, and outer 90% polygon
    from an aggregated particle cloud at a specific forecast horizon.
    """
    if len(all_lons) == 0:
        empty_poly = mapping(Polygon())
        return {
            "centroid": initial_centroid,
            "core_50": empty_poly,
            "spread_90": empty_poly,
            "spread_area_km2": 0.0,
            "drift_distance_km": 0.0,
            "mean_drift_speed_m_s": 0.0,
            "ensemble_std_km": 0.0
        }

    c_lon = float(np.mean(all_lons))
    c_lat = float(np.mean(all_lats))

    # Geodesic distance from initial centroid
    drift_distance_km = haversine_distance_km(
        initial_centroid[0], initial_centroid[1], c_lon, c_lat
    )

    # Mean drift speed over horizon duration
    duration_seconds = max(horizon_hours * 3600.0, 1.0)
    mean_speed_m_s = (drift_distance_km * 1000.0) / duration_seconds

    # Distances of individual particles from current centroid
    mean_lat_rad = np.radians(c_lat)
    dx_km = (all_lons - c_lon) * 111.32 * np.cos(mean_lat_rad)
    dy_km = (all_lats - c_lat) * 110.57
    dists_km = np.sqrt(dx_km**2 + dy_km**2)
    ensemble_std_km = float(np.std(dists_km))

    # Determine 50th and 90th percentile radial cutoffs
    r50 = float(np.percentile(dists_km, 50.0))
    r90 = float(np.percentile(dists_km, 90.0))

    # Core 50% points
    mask_50 = dists_km <= r50
    pts_50 = np.column_stack((all_lons[mask_50], all_lats[mask_50]))
    poly_50 = buffer_points_to_envelope(pts_50, buffer_radius_km=base_buffer_km)

    # Outer 90% points
    mask_90 = dists_km <= r90
    pts_90 = np.column_stack((all_lons[mask_90], all_lats[mask_90]))
    poly_90 = buffer_points_to_envelope(pts_90, buffer_radius_km=base_buffer_km * 1.2)

    area_90_km2 = approximate_polygon_area_km2(poly_90)

    return {
        "centroid": (round(c_lon, 6), round(c_lat, 6)),
        "core_50": mapping(poly_50),
        "spread_90": mapping(poly_90),
        "spread_area_km2": round(float(area_90_km2), 3),
        "drift_distance_km": round(float(drift_distance_km), 3),
        "mean_drift_speed_m_s": round(float(mean_speed_m_s), 3),
        "ensemble_std_km": round(float(ensemble_std_km), 3)
    }
