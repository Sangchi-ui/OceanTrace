"""
Stage B Hindcasting: Forward Physics Replay of Candidate Origins.
Simulates forward physical dispersion from each hypothesized origin to the observation timestamp,
evaluating spatial and geometric agreement to rank candidate source hypotheses.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
from shapely.geometry import mapping, shape

from agent2.adapters.forcing_base import EnvironmentalForcingProvider
from agent2.contracts.hindcast_result import CandidateHypothesis, TrajectoryPoint
from agent2.contracts.spill_observation import SpillObservation
from agent2.hindcast.candidate_envelope import CandidateOriginSeed
from agent2.hindcast.scoring import HypothesisScorer, ScoringConfig
from agent2.physics.oil_properties import OilProperties, get_oil_properties
from agent2.physics.particle_cloud import ParticleCloud
from agent2.physics.transport_engine import LagrangianTransportEngine


class ForwardReplayEngine:
    """
    Executes forward simulation and scoring for a batch of candidate origin hypotheses.
    """

    def __init__(
        self,
        forcing_provider: EnvironmentalForcingProvider,
        oil_props: Optional[OilProperties] = None,
        scoring_config: Optional[ScoringConfig] = None,
        particles_per_candidate: int = 150,
        timestep_minutes: float = 20.0,
        seed: Optional[int] = 42
    ):
        self.forcing = forcing_provider
        self.oil_props = oil_props or get_oil_properties()
        self.scorer = HypothesisScorer(scoring_config)
        self.particles_per_candidate = particles_per_candidate
        self.timestep_minutes = timestep_minutes
        self.seed = seed

    def replay_candidate(
        self,
        seed: CandidateOriginSeed,
        observed_spill: SpillObservation
    ) -> CandidateHypothesis:
        """Simulates forward transport of a single candidate seed to observation time."""
        t_release = seed.release_timestamp
        t_obs = observed_spill.observation_timestamp

        engine = LagrangianTransportEngine(
            forcing_provider=self.forcing,
            oil_props=self.oil_props,
            timestep_minutes=self.timestep_minutes,
            windage=self.oil_props.default_windage,
            horizontal_diffusivity=self.oil_props.horizontal_diffusivity_m2_s,
            seed=self.seed
        )

        # Initial point cloud at release coordinate (radius ~0.4 km)
        cloud = ParticleCloud.from_point(
            lon=seed.lon,
            lat=seed.lat,
            radius_km=0.4,
            num_particles=self.particles_per_candidate,
            seed=self.seed
        )

        # Run forward transport to t_obs
        snapshots = engine.simulate(
            cloud=cloud,
            start_time=t_release,
            end_time=t_obs,
            record_interval_minutes=60.0,
            apply_diffusion=True,
            reverse=False
        )

        # Final simulated state at t_obs
        sim_centroid = cloud.get_centroid()
        sim_envelope = cloud.to_envelope_polygon(percentile=90.0, buffer_km=0.5)
        sim_area_km2 = float(cloud.get_spread_radius_km()**2 * np.pi)

        # Evaluate against actual observed spill
        scores = self.scorer.evaluate(
            obs_geom=observed_spill.geometry,
            obs_centroid=observed_spill.centroid,
            obs_area_km2=observed_spill.area_km2,
            sim_geom=sim_envelope,
            sim_centroid=sim_centroid,
            sim_area_km2=sim_area_km2
        )

        # Format trajectory track
        trajectory_points = [
            TrajectoryPoint(timestamp=snap[0], longitude=snap[1][0], latitude=snap[1][1])
            for snap in snapshots
        ]

        return CandidateHypothesis(
            candidate_id=seed.candidate_id,
            release_timestamp=t_release,
            origin_coordinates=(seed.lon, seed.lat),
            simulated_final_centroid=sim_centroid,
            simulated_final_area_km2=round(sim_area_km2, 3),
            centroid_distance_km=scores["centroid_distance_km"],
            centroid_score=scores["centroid_score"],
            iou_score=scores["iou_score"],
            area_score=scores["area_score"],
            shape_score=scores["shape_score"],
            composite_score=scores["composite_score"],
            trajectory=trajectory_points
        )

    def replay_all(
        self,
        seeds: List[CandidateOriginSeed],
        observed_spill: SpillObservation
    ) -> List[CandidateHypothesis]:
        """Replays all candidate origin seeds and sorts by composite score descending."""
        hypotheses: List[CandidateHypothesis] = []
        for seed in seeds:
            hyp = self.replay_candidate(seed, observed_spill)
            hypotheses.append(hyp)

        # Sort descending by composite score
        hypotheses.sort(key=lambda h: h.composite_score, reverse=True)
        return hypotheses
