"""
Probable Origin Region and Time Window Estimator for Agent 2.
Aggregates top-ranked candidate hypotheses into an uncertain geographic region (Polygon)
and temporal window, adhering strictly to the No-Falsely-Precise-Point scientific policy.
"""

from datetime import datetime, timezone
from typing import List, Optional, Tuple
import numpy as np
from shapely.geometry import mapping, Polygon, MultiPolygon

from agent2.contracts.hindcast_result import CandidateHypothesis, OriginRegion
from agent2.geospatial.geodesy import (
    approximate_polygon_area_km2,
    buffer_points_to_envelope,
    haversine_distance_km,
)


class OriginRegionEstimator:
    """
    Synthesizes ranked candidate hypotheses into a probable origin region and time window.
    """

    @classmethod
    def estimate(
        cls,
        hypotheses: List[CandidateHypothesis],
        top_k_ratio: float = 0.25,
        min_candidates: int = 3,
        base_buffer_km: float = 3.5
    ) -> Optional[OriginRegion]:
        """
        Delineates origin region and time window from the highest-scoring candidate hypotheses.
        """
        if not hypotheses:
            return None

        # Filter top candidates
        best_score = hypotheses[0].composite_score
        # Take either top_k_ratio or candidates within 80% of best score
        cutoff_score = max(best_score * 0.75, 0.2)
        top_candidates = [
            h for h in hypotheses
            if h.composite_score >= cutoff_score
        ]

        if len(top_candidates) < min_candidates:
            # Fallback to top N candidates if threshold was too strict
            top_candidates = hypotheses[:min_candidates]

        # Extract coordinates and release times
        coords = np.array([
            [h.origin_coordinates[0], h.origin_coordinates[1]]
            for h in top_candidates
        ], dtype=np.float64)

        release_times = [h.release_timestamp for h in top_candidates]
        time_start = min(release_times)
        time_end = max(release_times)

        # Center of mass (score-weighted)
        scores = np.array([h.composite_score for h in top_candidates], dtype=np.float64)
        total_score = float(np.sum(scores))
        if total_score > 0:
            weights = scores / total_score
        else:
            weights = np.ones(len(scores)) / len(scores)

        c_lon = float(np.sum(coords[:, 0] * weights))
        c_lat = float(np.sum(coords[:, 1] * weights))

        # Spatial uncertainty radius: weighted RMS distance from center of mass + buffer
        dists = [
            haversine_distance_km(c_lon, c_lat, pt[0], pt[1])
            for pt in coords
        ]
        mean_lat = float(np.mean(coords[:, 1]))
        uncertainty_radius_km = float(np.sqrt(np.sum(weights * (np.array(dists)**2)))) + base_buffer_km

        # Build Polygon envelope around top candidate coordinates
        origin_poly = buffer_points_to_envelope(
            coords,
            buffer_radius_km=base_buffer_km,
            simplify_tolerance_km=0.1
        )
        area_km2 = approximate_polygon_area_km2(origin_poly)
        bounds = list(origin_poly.bounds)  # [minx, miny, maxx, maxy]

        # Overall origin confidence
        mean_confidence = float(np.sum(scores * weights))

        return OriginRegion(
            geometry_geojson=mapping(origin_poly),
            bounding_box=[round(b, 6) for b in bounds],
            centroid=(round(c_lon, 6), round(c_lat, 6)),
            area_km2=round(float(area_km2), 3),
            time_window_start=time_start,
            time_window_end=time_end,
            confidence=round(float(mean_confidence), 4),
            uncertainty_radius_km=round(float(uncertainty_radius_km), 3)
        )
