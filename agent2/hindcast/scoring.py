"""
Modular Source Hypothesis Scoring Engine for Agent 2.
Implements multi-metric comparison between simulated replay spills and the observed Agent 1 spill:
1. Centroid distance agreement (Gaussian decay)
2. Intersection-over-Union (IoU) spatial overlap
3. Area ratio agreement
4. Aspect ratio / Shape similarity
"""

import math
from typing import Any, Dict, Optional, Tuple
from shapely.geometry import Polygon, MultiPolygon, shape
from shapely.ops import unary_union

from agent2.geospatial.geodesy import haversine_distance_km, approximate_polygon_area_km2


class ScoringConfig:
    """Configurable weights and scale parameters for hypothesis scoring."""
    def __init__(
        self,
        weight_centroid: float = 0.35,
        weight_iou: float = 0.35,
        weight_area: float = 0.15,
        weight_shape: float = 0.15,
        centroid_sigma_km: float = 3.5
    ):
        self.w_centroid = float(weight_centroid)
        self.w_iou = float(weight_iou)
        self.w_area = float(weight_area)
        self.w_shape = float(weight_shape)
        self.sigma_km = float(centroid_sigma_km)

        # Normalize weights to sum to 1.0
        total = self.w_centroid + self.w_iou + self.w_area + self.w_shape
        if total > 0:
            self.w_centroid /= total
            self.w_iou /= total
            self.w_area /= total
            self.w_shape /= total


class HypothesisScorer:
    """
    Evaluates agreement between forward-simulated candidate spill and observed Agent 1 spill.
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.cfg = config or ScoringConfig()

    def score_centroid(self, obs_centroid: Tuple[float, float], sim_centroid: Tuple[float, float]) -> Tuple[float, float]:
        """
        Computes Gaussian distance agreement score:
            S_c = exp(-d^2 / (2 * sigma^2))
        Returns (centroid_score, distance_km).
        """
        d_km = haversine_distance_km(
            obs_centroid[0], obs_centroid[1], sim_centroid[0], sim_centroid[1]
        )
        sigma = max(self.cfg.sigma_km, 0.1)
        score = math.exp(-(d_km**2) / (2.0 * sigma**2))
        return float(np_clip(score, 0.0, 1.0)), round(float(d_km), 3)

    def score_iou(
        self,
        obs_geom: Polygon | MultiPolygon | dict,
        sim_geom: Polygon | MultiPolygon | dict
    ) -> float:
        """
        Computes spatial Intersection-over-Union (IoU) between observed and simulated polygons.
        IoU = Area(Obs ∩ Sim) / Area(Obs ∪ Sim)
        """
        g_obs = shape(obs_geom) if isinstance(obs_geom, dict) else obs_geom
        g_sim = shape(sim_geom) if isinstance(sim_geom, dict) else sim_geom

        if g_obs.is_empty or g_sim.is_empty:
            return 0.0

        try:
            # Fix any self-intersections or topology invalidities
            if not g_obs.is_valid:
                g_obs = g_obs.buffer(0)
            if not g_sim.is_valid:
                g_sim = g_sim.buffer(0)

            intersection = g_obs.intersection(g_sim)
            union = g_obs.union(g_sim)

            if union.area <= 0:
                return 0.0

            iou = intersection.area / union.area
            return float(np_clip(iou, 0.0, 1.0))
        except Exception:
            return 0.0

    def score_area(self, obs_area_km2: float, sim_area_km2: float) -> float:
        """
        Computes normalized area agreement:
            S_area = min(A_sim, A_obs) / max(A_sim, A_obs)
        """
        if obs_area_km2 <= 0 or sim_area_km2 <= 0:
            return 0.0
        ratio = min(obs_area_km2, sim_area_km2) / max(obs_area_km2, sim_area_km2)
        return float(np_clip(ratio, 0.0, 1.0))

    def score_shape(
        self,
        obs_geom: Polygon | MultiPolygon | dict,
        sim_geom: Polygon | MultiPolygon | dict
    ) -> float:
        """
        Computes shape agreement based on aspect ratio difference and compactness.
        """
        g_obs = shape(obs_geom) if isinstance(obs_geom, dict) else obs_geom
        g_sim = shape(sim_geom) if isinstance(sim_geom, dict) else sim_geom

        if g_obs.is_empty or g_sim.is_empty:
            return 0.0

        minx1, miny1, maxx1, maxy1 = g_obs.bounds
        minx2, miny2, maxx2, maxy2 = g_sim.bounds

        w1 = max(maxx1 - minx1, 1e-5)
        h1 = max(maxy1 - miny1, 1e-5)
        ar1 = max(w1 / h1, h1 / w1)

        w2 = max(maxx2 - minx2, 1e-5)
        h2 = max(maxy2 - miny2, 1e-5)
        ar2 = max(w2 / h2, h2 / w2)

        ar_score = min(ar1, ar2) / max(ar1, ar2)
        return float(np_clip(ar_score, 0.0, 1.0))

    def evaluate(
        self,
        obs_geom: Polygon | MultiPolygon | dict,
        obs_centroid: Tuple[float, float],
        obs_area_km2: float,
        sim_geom: Polygon | MultiPolygon | dict,
        sim_centroid: Tuple[float, float],
        sim_area_km2: float
    ) -> Dict[str, float]:
        """Calculates all individual metric scores and the weighted composite score."""
        c_score, dist_km = self.score_centroid(obs_centroid, sim_centroid)
        iou_score = self.score_iou(obs_geom, sim_geom)
        area_score = self.score_area(obs_area_km2, sim_area_km2)
        shape_score = self.score_shape(obs_geom, sim_geom)

        total_score = (
            self.cfg.w_centroid * c_score +
            self.cfg.w_iou * iou_score +
            self.cfg.w_area * area_score +
            self.cfg.w_shape * shape_score
        )

        return {
            "composite_score": round(float(np_clip(total_score, 0.0, 1.0)), 4),
            "centroid_score": round(float(c_score), 4),
            "iou_score": round(float(iou_score), 4),
            "area_score": round(float(area_score), 4),
            "shape_score": round(float(shape_score), 4),
            "centroid_distance_km": round(float(dist_km), 3)
        }


def np_clip(val: float, low: float, high: float) -> float:
    return max(min(val, high), low)
