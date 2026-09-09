"""
Particle Cloud representation and spatial initialization for Lagrangian oil dispersion.
Distributes particles according to detected slick geometry without naive point approximations.
"""

from typing import Any, List, Optional, Tuple
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point, shape
from shapely.prepared import prep

from agent2.geospatial.geodesy import (
    haversine_distance_km,
    approximate_polygon_area_km2,
    buffer_points_to_envelope,
)


class ParticleCloud:
    """
    Manages state, spatial distribution, and lifecycle of Lagrangian oil particles.
    """

    def __init__(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        masses: Optional[np.ndarray] = None,
        active: Optional[np.ndarray] = None
    ):
        self.lons = np.asarray(lons, dtype=np.float64)
        self.lats = np.asarray(lats, dtype=np.float64)
        self.count = len(self.lons)

        if masses is not None:
            self.masses = np.asarray(masses, dtype=np.float64)
        else:
            self.masses = np.ones(self.count, dtype=np.float64)

        if active is not None:
            self.active = np.asarray(active, dtype=bool)
        else:
            self.active = np.ones(self.count, dtype=bool)

        # Coordinate history snapshots: list of (lon_array, lat_array)
        self.history: List[Tuple[np.ndarray, np.ndarray]] = []

    @classmethod
    def from_polygon(
        cls,
        polygon_geom: Polygon | MultiPolygon | dict,
        num_particles: int = 500,
        seed: Optional[int] = 42
    ) -> "ParticleCloud":
        """
        Generates num_particles uniformly sampled within the given polygon geometry
        using optimized vectorized rejection sampling.
        """
        if isinstance(polygon_geom, dict):
            geom = shape(polygon_geom)
        else:
            geom = polygon_geom

        if geom.is_empty:
            raise ValueError("Cannot initialize particles inside an empty geometry.")

        rng = np.random.RandomState(seed)
        minx, miny, maxx, maxy = geom.bounds

        sampled_lons: List[float] = []
        sampled_lats: List[float] = []

        prepared_geom = prep(geom)
        batch_size = max(num_particles * 2, 200)

        # Rejection sampling in bounding box
        attempts = 0
        while len(sampled_lons) < num_particles and attempts < 25:
            cand_x = rng.uniform(minx, maxx, batch_size)
            cand_y = rng.uniform(miny, maxy, batch_size)

            for x, y in zip(cand_x, cand_y):
                if prepared_geom.contains(Point(x, y)):
                    sampled_lons.append(float(x))
                    sampled_lats.append(float(y))
                    if len(sampled_lons) >= num_particles:
                        break
            attempts += 1

        # If strict polygon rejection was insufficient (e.g. extremely thin line), fallback to centroid with jitter
        if len(sampled_lons) < num_particles:
            c = geom.centroid
            needed = num_particles - len(sampled_lons)
            scale_x = max(maxx - minx, 1e-4) * 0.1
            scale_y = max(maxy - miny, 1e-4) * 0.1
            sampled_lons.extend(list(rng.normal(c.x, scale_x, needed)))
            sampled_lats.extend(list(rng.normal(c.y, scale_y, needed)))

        return cls(
            lons=np.array(sampled_lons[:num_particles]),
            lats=np.array(sampled_lats[:num_particles])
        )

    @classmethod
    def from_point(
        cls,
        lon: float,
        lat: float,
        radius_km: float = 0.5,
        num_particles: int = 500,
        seed: Optional[int] = 42
    ) -> "ParticleCloud":
        """Initializes a Gaussian circular particle cloud around a point source."""
        rng = np.random.RandomState(seed)
        # Convert radius in km to degrees
        deg_lat = radius_km / 111.0
        deg_lon = radius_km / (111.0 * np.cos(np.radians(lat)) + 1e-6)

        lons = rng.normal(lon, deg_lon * 0.5, num_particles)
        lats = rng.normal(lat, deg_lat * 0.5, num_particles)
        return cls(lons=lons, lats=lats)

    def record_snapshot(self) -> None:
        """Saves current particle positions to trajectory history."""
        self.history.append((self.lons.copy(), self.lats.copy()))

    def get_centroid(self) -> Tuple[float, float]:
        """Calculates mass-weighted center of active particles (lon, lat)."""
        active_mask = self.active
        if not np.any(active_mask):
            return float(np.mean(self.lons)), float(np.mean(self.lats))

        w = self.masses[active_mask]
        total_w = np.sum(w)
        if total_w <= 0:
            total_w = 1.0

        c_lon = float(np.sum(self.lons[active_mask] * w) / total_w)
        c_lat = float(np.sum(self.lats[active_mask] * w) / total_w)
        return round(c_lon, 6), round(c_lat, 6)

    def get_spread_radius_km(self) -> float:
        """Calculates root-mean-square spatial dispersion radius around centroid in km."""
        c_lon, c_lat = self.get_centroid()
        active_lons = self.lons[self.active]
        active_lats = self.lats[self.active]
        if len(active_lons) <= 1:
            return 0.1

        mean_lat = float(np.mean(active_lats))
        dx_km = (active_lons - c_lon) * 111.32 * np.cos(np.radians(mean_lat))
        dy_km = (active_lats - c_lat) * 110.57
        rms_radius = float(np.sqrt(np.mean(dx_km**2 + dy_km**2)))
        return round(rms_radius, 3)

    def to_envelope_polygon(
        self,
        percentile: float = 90.0,
        buffer_km: float = 0.5
    ) -> Polygon | MultiPolygon:
        """
        Constructs an envelope polygon covering a given percentile of active particles.
        Filters out extreme outliers before buffering to obtain core (50%) or spread (90%) envelopes.
        """
        active_lons = self.lons[self.active]
        active_lats = self.lats[self.active]
        if len(active_lons) == 0:
            return Polygon()

        c_lon, c_lat = self.get_centroid()
        mean_lat = float(np.mean(active_lats))
        dx_km = (active_lons - c_lon) * 111.32 * np.cos(np.radians(mean_lat))
        dy_km = (active_lats - c_lat) * 110.57
        dists_km = np.sqrt(dx_km**2 + dy_km**2)

        cutoff = float(np.percentile(dists_km, percentile))
        mask = dists_km <= cutoff
        selected_points = np.column_stack((active_lons[mask], active_lats[mask]))

        return buffer_points_to_envelope(selected_points, buffer_radius_km=buffer_km)
