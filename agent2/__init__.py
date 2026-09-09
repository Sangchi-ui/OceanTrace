"""
OceanTrace — Agent 2: Hindcasting & Forecasting Package.
Physics-first Lagrangian particle transport, two-stage source estimation,
and ensemble-based future dispersion modeling for marine oil spills.
"""

from agent2.contracts import (
    SpillObservation,
    Agent2Result,
    HindcastResult,
    ForecastResult,
)
from agent2.pipeline import Agent2Pipeline
from agent2.config import Agent2Config

__version__ = "2.0.0"

__all__ = [
    "SpillObservation",
    "Agent2Result",
    "HindcastResult",
    "ForecastResult",
    "Agent2Pipeline",
    "Agent2Config",
    "__version__",
]
