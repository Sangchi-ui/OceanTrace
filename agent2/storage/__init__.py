"""
OceanTrace Agent 2 Storage Layer.
Provides PostgreSQL and PostGIS spatial persistence and DDL/SQL generation.
"""

from agent2.storage.postgis_adapter import PostGISStorage

__all__ = ["PostGISStorage"]
