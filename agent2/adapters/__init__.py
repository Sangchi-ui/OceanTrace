"""
Adapters package for Agent 2.
"""

from agent2.adapters.forcing_base import EnvironmentalForcingProvider
from agent2.adapters.mock_forcing import MockEnvironmentalForcingProvider
from agent2.adapters.cmems_adapter import CMEMSCurrentProvider
from agent2.adapters.era5_adapter import ERA5WindProvider
from agent2.adapters.composite_forcing import CompositeForcingProvider
from agent2.adapters.agent1_adapter import Agent1Adapter

__all__ = [
    "EnvironmentalForcingProvider",
    "MockEnvironmentalForcingProvider",
    "CMEMSCurrentProvider",
    "ERA5WindProvider",
    "CompositeForcingProvider",
    "Agent1Adapter",
]
