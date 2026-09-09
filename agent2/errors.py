"""
Structured exception hierarchy for Agent 2: Hindcasting & Forecasting.
Ensures meaningful, categorized diagnostic messages without generic fallbacks.
"""


class Agent2Error(Exception):
    """Base exception for all Agent 2 errors."""
    pass


class Agent1ContractError(Agent2Error):
    """Raised when an Agent 1 output payload violates schema expectations."""
    pass


class GeometryValidationError(Agent2Error):
    """Raised when coordinates, CRS, or polygon geometries are invalid or missing."""
    pass


class CRSValidationError(Agent2Error):
    """Raised when coordinate reference system is unsupported or incompatible."""
    pass


class EnvironmentalDataError(Agent2Error):
    """Raised when required current or wind forcing datasets cannot be retrieved."""
    pass


class SimulationError(Agent2Error):
    """Raised when a particle transport simulation fails sanity checks or divergence limits."""
    pass


class ConfigurationError(Agent2Error):
    """Raised when pipeline configuration is invalid or missing required keys."""
    pass
