"""
Tests for OceanTrace Agent 2 PostgreSQL / PostGIS Adapter and Spatial Storage.
"""

from datetime import datetime, timezone
import os
import tempfile
import pytest

from agent2.contracts.agent2_result import Agent2Result
from agent2.contracts.spill_observation import SpillObservation
from agent2.pipeline import Agent2Pipeline
from agent2.storage.postgis_adapter import PostGISStorage


def test_postgis_ddl_generation():
    ddl = PostGISStorage.generate_ddl_sql()
    assert "CREATE EXTENSION IF NOT EXISTS postgis;" in ddl
    assert "CREATE TABLE IF NOT EXISTS agent2_spills" in ddl
    assert "CREATE TABLE IF NOT EXISTS agent2_origin_regions" in ddl
    assert "CREATE TABLE IF NOT EXISTS agent2_hindcast_trajectories" in ddl
    assert "CREATE TABLE IF NOT EXISTS agent2_forecast_envelopes" in ddl
    assert "CREATE TABLE IF NOT EXISTS agent2_xgboost_residuals" in ddl
    assert "USING GIST (centroid_geom)" in ddl
    assert "USING GIST (envelope_geom)" in ddl
    assert "USING GIST (trajectory_geom)" in ddl


def test_postgis_export_sql_script():
    pipeline = Agent2Pipeline()
    obs = SpillObservation(
        spill_id="slick_postgis_test",
        observation_timestamp=datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc),
        centroid=(104.2, 1.3),
        geometry={
            "type": "Polygon",
            "coordinates": [[[104.19, 1.29], [104.21, 1.29], [104.21, 1.31], [104.19, 1.31], [104.19, 1.29]]]
        },
        area_km2=4.5,
        confidence=0.92,
        source_satellite="Sentinel-1"
    )

    result = pipeline.run(obs, mode="both")
    assert result.ml_residual is not None
    assert "dx_residual_meters" in result.ml_residual

    storage = PostGISStorage()
    sql_script = storage.export_sql_script(result)

    assert "BEGIN;" in sql_script
    assert "COMMIT;" in sql_script
    assert "INSERT INTO agent2_spills" in sql_script
    assert "ST_SetSRID(ST_GeomFromGeoJSON" in sql_script
    assert "INSERT INTO agent2_xgboost_residuals" in sql_script

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, "insert.sql")
        storage.export_to_file(result, out_file)
        assert os.path.exists(out_file)
        with open(out_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert "slick_postgis_test" in content
