"""
Agent 2 data contracts module.
"""

from agent2.contracts.spill_observation import SpillObservation
from agent2.contracts.hindcast_result import (
    TrajectoryPoint,
    CandidateHypothesis,
    OriginRegion,
    HindcastResult,
)
from agent2.contracts.forecast_result import (
    EnsembleTrajectory,
    ForecastHorizonResult,
    ForecastResult,
)
from agent2.contracts.agent2_result import Agent2Result

__all__ = [
    "SpillObservation",
    "TrajectoryPoint",
    "CandidateHypothesis",
    "OriginRegion",
    "HindcastResult",
    "EnsembleTrajectory",
    "ForecastHorizonResult",
    "ForecastResult",
    "Agent2Result",
]
