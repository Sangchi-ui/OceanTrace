"""
Hindcasting engine package for Agent 2.
"""

from agent2.hindcast.scoring import HypothesisScorer, ScoringConfig
from agent2.hindcast.candidate_envelope import (
    CandidateOriginSeed,
    CandidateEnvelopeGenerator,
)
from agent2.hindcast.forward_replay import ForwardReplayEngine
from agent2.hindcast.origin_estimator import OriginRegionEstimator
from agent2.hindcast.hindcast_engine import HindcastingEngine

__all__ = [
    "HypothesisScorer",
    "ScoringConfig",
    "CandidateOriginSeed",
    "CandidateEnvelopeGenerator",
    "ForwardReplayEngine",
    "OriginRegionEstimator",
    "HindcastingEngine",
]
