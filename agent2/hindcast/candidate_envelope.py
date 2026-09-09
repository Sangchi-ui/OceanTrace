"""
Stage A Hindcasting: Backward Candidate Origin Search Envelope.
Traces reverse physical advection from observed slick coordinates to establish
the plausible historical spatial-temporal envelope without naive reverse diffusion claims.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from shapely.geometry import mapping, Polygon, MultiPolygon, shape

from agent2.adapters.forcing_base import EnvironmentalForcingProvider
from agent2.contracts.spill_observation import SpillObservation
from agent2.geospatial.geodesy import buffer_points_to_envelope
from agent2.physics.oil_properties import OilProperties, get_oil_properties
from agent2.physics.particle_cloud import ParticleCloud
from agent2.physics.transport_engine import LagrangianTransportEngine


class CandidateOriginSeed:
    """A discrete hypothesized source release location and time."""
    def __init__(
        self,
        candidate_id: str,
        release_timestamp: datetime,
        lon: float,
        lat: float,
        stage_a_distance_km: float = 0.0
    ):
        self.candidate_id = candidate_id
        self.release_timestamp = release_timestamp
        self.lon = round(float(lon), 6)
        self.lat = round(float(lat), 6)
        self.stage_a_distance_km = round(float(stage_a_distance_km), 3)


class CandidateEnvelopeGenerator:
    """
    Constructs Stage A backward envelope and candidate hypothesis seeds.
    """

    def __init__(
        self,
        forcing_provider: EnvironmentalForcingProvider,
        oil_props: Optional[OilProperties] = None,
        max_lookback_hours: int = 48,
        time_step_hours: int = 6,
        cross_track_samples: int = 3,
        seed: Optional[int] = 42
    ):
        self.forcing = forcing_provider
        self.oil_props = oil_props or get_oil_properties()
        self.max_lookback_hours = max_lookback_hours
        self.time_step_hours = max(time_step_hours, 2)
        self.cross_track_samples = cross_track_samples
        self.seed = seed

    def generate(
        self, spill: SpillObservation
    ) -> Tuple[Dict[str, Any], List[CandidateOriginSeed]]:
        """
        Runs backward advection from observed slick centroid across historical horizons.
        Returns:
          - envelope_geojson: GeoJSON Polygon representing search corridor
          - candidate_seeds: List of candidate origin (lon, lat, release_time) hypotheses
        """
        obs_time = spill.observation_timestamp
        c_lon, c_lat = spill.centroid

        # We test lower, mean, and upper windage bounds to bound the advection corridor
        windage_bounds = [
            self.oil_props.min_windage,
            self.oil_props.default_windage,
            self.oil_props.max_windage
        ]

        # Horizons to test backward: e.g. -6h, -12h, -18h, ..., -max_lookback_hours
        lookbacks = list(range(
            self.time_step_hours,
            self.max_lookback_hours + 1,
            self.time_step_hours
        ))

        all_corridor_points: List[Tuple[float, float]] = [(c_lon, c_lat)]
        candidate_seeds: List[CandidateOriginSeed] = []
        cand_counter = 1

        for w_val in windage_bounds:
            engine = LagrangianTransportEngine(
                forcing_provider=self.forcing,
                oil_props=self.oil_props,
                timestep_minutes=20.0,
                windage=w_val,
                horizontal_diffusivity=0.0,  # Advection only for Stage A
                seed=self.seed
            )

            # Trace backward particle from centroid
            p_lon, p_lat = c_lon, c_lat
            prev_t = obs_time

            for h in lookbacks:
                target_t = obs_time - timedelta(hours=h)
                # Single particle cloud for tracking trajectory
                cloud = ParticleCloud(
                    lons=np.array([p_lon]),
                    lats=np.array([p_lat])
                )
                engine.simulate(
                    cloud=cloud,
                    start_time=prev_t,
                    end_time=target_t,
                    apply_diffusion=False,
                    reverse=True
                )
                p_lon, p_lat = cloud.get_centroid()
                prev_t = target_t

                all_corridor_points.append((p_lon, p_lat))

                # If middle windage (or primary track), sample cross-track candidate seeds
                if abs(w_val - self.oil_props.default_windage) < 1e-4:
                    # Central seed
                    seed_id = f"HYP_{h:02d}H_C0"
                    candidate_seeds.append(CandidateOriginSeed(
                        candidate_id=seed_id,
                        release_timestamp=target_t,
                        lon=p_lon,
                        lat=p_lat
                    ))

                    # Cross-track spatial dispersion seeds (proportional to lookback time)
                    spread_scale_deg = (h / 24.0) * 0.05
                    if self.cross_track_samples > 1 and spread_scale_deg > 0.005:
                        candidate_seeds.append(CandidateOriginSeed(
                            candidate_id=f"HYP_{h:02d}H_N1",
                            release_timestamp=target_t,
                            lon=p_lon,
                            lat=p_lat + spread_scale_deg
                        ))
                        candidate_seeds.append(CandidateOriginSeed(
                            candidate_id=f"HYP_{h:02d}H_S1",
                            release_timestamp=target_t,
                            lon=p_lon,
                            lat=p_lat - spread_scale_deg
                        ))
                        candidate_seeds.append(CandidateOriginSeed(
                            candidate_id=f"HYP_{h:02d}H_E1",
                            release_timestamp=target_t,
                            lon=p_lon + spread_scale_deg,
                            lat=p_lat
                        ))
                        candidate_seeds.append(CandidateOriginSeed(
                            candidate_id=f"HYP_{h:02d}H_W1",
                            release_timestamp=target_t,
                            lon=p_lon - spread_scale_deg,
                            lat=p_lat
                        ))

        # Build Stage A corridor envelope polygon
        pts_arr = np.array(all_corridor_points, dtype=np.float64)
        corridor_poly = buffer_points_to_envelope(
            pts_arr,
            buffer_radius_km=4.0,  # 4 km search buffer along trajectory
            simplify_tolerance_km=0.2
        )

        return mapping(corridor_poly), candidate_seeds
