"""
Agent 2 API package.
"""

from agent2.api.routes import agent2_router
from agent2.api.main import app

__all__ = ["agent2_router", "app"]
