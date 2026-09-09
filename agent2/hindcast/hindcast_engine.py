"""
Orchestrator for the Two-Stage Hindcasting Engine of Agent 2.
Coordinates backward envelope generation (Stage A), candidate origin hypothesis forward replay (Stage B),
scoring (Stage C), and probable origin region estimation (Stage D).
"""

from typing import Any, Dict, Optional
import time

from agent2.adapters.forcing_base import EnvironmentalForcingProvider
from agent2.contracts.hindcast_result import HindcastResult
from agent2.contracts.spill_observation import SpillObservation
from agent2.hindcast.candidate_envelope import CandidateEnvelopeGenerator
from agent2.hindcast.forward_replay import ForwardReplayEngine
from agent2.hindcast.origin_estimator import OriginRegionEstimator
from agent2.hindcast.scoring import ScoringConfig
from agent2.physics.oil_properties import OilProperties, get_oil_properties


class HindcastingEngine:
    """
    Two-Stage Physics-Based Hindcasting Engine for estimating probable oil spill origins.
    """

    def __init__(
        self,
        forcing_provider: EnvironmentalForcingProvider,
        oil_props: Optional[OilProperties] = None,
        max_lookback_hours: int = 48,
        time_step_hours: int = 6,
        particles_per_candidate: int = 150,
        scoring_config: Optional[ScoringConfig] = None,
        timestep_minutes: float = 20.0,
        seed: Optional[int] = 42
    ):
        self.forcing = forcing_provider
        self.oil_props = oil_props or get_oil_properties()
        self.max_lookback_hours = max_lookback_hours
        self.time_step_hours = time_step_hours
        self.particles_per_candidate = particles_per_candidate
        self.scoring_config = scoring_config or ScoringConfig()
        self.timestep_minutes = timestep_minutes
        self.seed = seed

    def run(self, spill: SpillObservation) -> HindcastResult:
        """Executes complete two-stage hindcast analysis."""
        t_start = time.time()

        # 1. Stage A: Generate backward candidate search envelope & discrete origin seeds
        envelope_gen = CandidateEnvelopeGenerator(
            forcing_provider=self.forcing,
            oil_props=self.oil_props,
            max_lookback_hours=self.max_lookback_hours,
            time_step_hours=self.time_step_hours,
            seed=self.seed
        )
        corridor_geojson, candidate_seeds = envelope_gen.generate(spill)

        # 2. Stage B: Forward replay of candidate origins to observation time
        replay_engine = ForwardReplayEngine(
            forcing_provider=self.forcing,
            oil_props=self.oil_props,
            scoring_config=self.scoring_config,
            particles_per_candidate=self.particles_per_candidate,
            timestep_minutes=self.timestep_minutes,
            seed=self.seed
        )
        ranked_hypotheses = replay_engine.replay_all(candidate_seeds, spill)

        # 3. Stage D: Estimate probable origin region and release time window
        probable_origin = OriginRegionEstimator.estimate(ranked_hypotheses)
        best_candidate = ranked_hypotheses[0] if ranked_hypotheses else None

        elapsed = round(time.time() - t_start, 2)

        return HindcastResult(
            enabled=True,
            status="SUCCESS",
            probable_origin=probable_origin,
            candidate_hypotheses=ranked_hypotheses,
            best_candidate=best_candidate,
            historical_envelope_geojson=corridor_geojson,
            diagnostics={
                "runtime_seconds": elapsed,
                "candidates_evaluated": len(ranked_hypotheses),
                "max_lookback_hours": self.max_lookback_hours,
                "time_step_hours": self.time_step_hours,
                "best_candidate_id": best_candidate.candidate_id if best_candidate else None,
                "best_candidate_score": best_candidate.composite_score if best_candidate else 0.0
            }
        )
