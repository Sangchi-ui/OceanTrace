"""
GeoJSON conversion and export utilities for Agent 2 outputs.
Ensures RFC 7946 compliance, WGS84 CRS, and comprehensive GIS feature properties.
"""

from datetime import datetime
from typing import Any, List, Dict
import json
from shapely.geometry import mapping, Polygon, MultiPolygon, LineString, Point


def create_feature(
    geometry: dict | Polygon | MultiPolygon | LineString | Point,
    properties: Dict[str, Any]
) -> Dict[str, Any]:
    """Wraps a Shapely geometry or GeoJSON geometry dict into a GeoJSON Feature."""
    geom_dict = mapping(geometry) if hasattr(geometry, "__geo_interface__") else geometry
    return {
        "type": "Feature",
        "geometry": geom_dict,
        "properties": properties
    }


def create_feature_collection(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Wraps a list of GeoJSON features into a FeatureCollection."""
    return {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
        },
        "features": features
    }


def build_unified_geojson(
    observed_spill: Any,
    hindcast_result: Any,
    forecast_result: Any
) -> Dict[str, Any]:
    """
    Creates a single unified GeoJSON FeatureCollection representing the complete
    Agent 2 analysis (Observed slick, Probable Origin Region, Best Candidate Path,
    Forecast Spread Envelopes 50% & 90%, and Forecast Centroids).
    """
    features = []

    # 1. Observed Slick Polygon
    if observed_spill is not None:
        features.append(create_feature(
            geometry=observed_spill.geometry,
            properties={
                "agent": "agent_1",
                "layer": "observed_spill",
                "spill_id": observed_spill.spill_id,
                "timestamp": observed_spill.observation_timestamp.isoformat(),
                "area_km2": observed_spill.area_km2,
                "confidence": observed_spill.confidence,
                "stroke": "#e63946",
                "fill": "#e63946",
                "fill_opacity": 0.4
            }
        ))

    # 2. Hindcast: Probable Origin Region
    if hindcast_result is not None and hindcast_result.probable_origin is not None:
        po = hindcast_result.probable_origin
        features.append(create_feature(
            geometry=po.geometry_geojson,
            properties={
                "agent": "agent_2",
                "layer": "probable_origin_region",
                "spill_id": observed_spill.spill_id if observed_spill else "unknown",
                "time_window_start": po.time_window_start.isoformat(),
                "time_window_end": po.time_window_end.isoformat(),
                "area_km2": po.area_km2,
                "confidence": po.confidence,
                "uncertainty_radius_km": po.uncertainty_radius_km,
                "stroke": "#f4a261",
                "fill": "#f4a261",
                "fill_opacity": 0.35
            }
        ))

    # 3. Hindcast: Best Candidate Trajectory Path
    if hindcast_result is not None and hindcast_result.best_candidate is not None:
        best = hindcast_result.best_candidate
        if len(best.trajectory) > 1:
            coords = [[pt.longitude, pt.latitude] for pt in best.trajectory]
            features.append(create_feature(
                geometry={"type": "LineString", "coordinates": coords},
                properties={
                    "agent": "agent_2",
                    "layer": "best_hindcast_trajectory",
                    "candidate_id": best.candidate_id,
                    "release_time": best.release_timestamp.isoformat(),
                    "composite_score": best.composite_score,
                    "centroid_distance_km": best.centroid_distance_km,
                    "iou_score": best.iou_score,
                    "stroke": "#e76f51",
                    "stroke_width": 3
                }
            ))

    # 4. Forecast Horizons
    if forecast_result is not None and forecast_result.horizons:
        for hor in forecast_result.horizons:
            h_val = hor.horizon_hours
            # Core 50%
            features.append(create_feature(
                geometry=hor.core_envelope_50_geojson,
                properties={
                    "agent": "agent_2",
                    "layer": f"forecast_envelope_50_{h_val}h",
                    "horizon_hours": h_val,
                    "forecast_time": hor.forecast_timestamp.isoformat(),
                    "spread_area_km2": hor.spread_area_km2,
                    "drift_distance_km": hor.drift_distance_km,
                    "stroke": "#2a9d8f",
                    "fill": "#2a9d8f",
                    "fill_opacity": 0.5
                }
            ))
            # Outer 90%
            features.append(create_feature(
                geometry=hor.spread_envelope_90_geojson,
                properties={
                    "agent": "agent_2",
                    "layer": f"forecast_envelope_90_{h_val}h",
                    "horizon_hours": h_val,
                    "forecast_time": hor.forecast_timestamp.isoformat(),
                    "spread_area_km2": hor.spread_area_km2,
                    "stroke": "#457b9d",
                    "fill": "#457b9d",
                    "fill_opacity": 0.25
                }
            ))
            # Forecasted Centroid Point
            features.append(create_feature(
                geometry={"type": "Point", "coordinates": [hor.centroid[0], hor.centroid[1]]},
                properties={
                    "agent": "agent_2",
                    "layer": f"forecast_centroid_{h_val}h",
                    "horizon_hours": h_val,
                    "forecast_time": hor.forecast_timestamp.isoformat(),
                    "centroid_lon": hor.centroid[0],
                    "centroid_lat": hor.centroid[1],
                    "drift_distance_km": hor.drift_distance_km
                }
            ))
            
        # 5. Particle Scatter (Final Frame)
        if forecast_result.animation_frames and len(forecast_result.animation_frames) > 0:
            final_frame = forecast_result.animation_frames[-1]
            if "particles" in final_frame and final_frame["particles"]:
                features.append(create_feature(
                    geometry={"type": "MultiPoint", "coordinates": final_frame["particles"]},
                    properties={
                        "agent": "agent_2",
                        "layer": "particle_scatter_geojson",
                        "forecast_time": final_frame["t"],
                        "stroke": "#ff6b2b",
                        "fill": "#ff6b2b"
                    }
                ))

    return create_feature_collection(features)
