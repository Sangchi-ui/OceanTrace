"""
PostgreSQL / PostGIS Storage Adapter for OceanTrace Agent 2.
Implements spatial persistence for spill observations, hindcast trajectories,
origin candidate regions, forecast probability envelopes, and XGBoost ML residuals.
"""

from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Optional
import uuid

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    text
)
from sqlalchemy.orm import declarative_base, sessionmaker

try:
    from geoalchemy2 import Geometry
    GEOALCHEMY_AVAILABLE = True
except ImportError:
    GEOALCHEMY_AVAILABLE = False

from agent2.contracts.agent2_result import Agent2Result

logger = logging.getLogger("OceanTrace.Agent2.PostGIS")

Base = declarative_base()


class SpillRecord(Base):
    __tablename__ = "agent2_spills"

    id = Column(String(64), primary_key=True)
    spill_id = Column(String(64), index=True, nullable=False)
    observation_timestamp = Column(DateTime(timezone=True), nullable=False)
    area_km2 = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    source_satellite = Column(String(64), nullable=True)
    if GEOALCHEMY_AVAILABLE:
        centroid_geom = Column(Geometry("POINT", srid=4326), nullable=True)
        polygon_geom = Column(Geometry("POLYGON", srid=4326), nullable=True)
    else:
        centroid_geom = Column(Text, nullable=True)
        polygon_geom = Column(Text, nullable=True)


class OriginRegionRecord(Base):
    __tablename__ = "agent2_origin_regions"

    id = Column(String(64), primary_key=True)
    analysis_id = Column(String(64), index=True, nullable=False)
    spill_id = Column(String(64), index=True, nullable=False)
    confidence = Column(Float, nullable=False)
    area_km2 = Column(Float, nullable=False)
    time_window_start = Column(DateTime(timezone=True), nullable=False)
    time_window_end = Column(DateTime(timezone=True), nullable=False)
    if GEOALCHEMY_AVAILABLE:
        envelope_geom = Column(Geometry("POLYGON", srid=4326), nullable=True)
    else:
        envelope_geom = Column(Text, nullable=True)


class HindcastTrajectoryRecord(Base):
    __tablename__ = "agent2_hindcast_trajectories"

    id = Column(String(64), primary_key=True)
    analysis_id = Column(String(64), index=True, nullable=False)
    candidate_id = Column(String(64), nullable=False)
    rank = Column(Integer, nullable=False)
    composite_score = Column(Float, nullable=False)
    centroid_distance_km = Column(Float, nullable=False)
    if GEOALCHEMY_AVAILABLE:
        trajectory_geom = Column(Geometry("LINESTRING", srid=4326), nullable=True)
    else:
        trajectory_geom = Column(Text, nullable=True)


class ForecastEnvelopeRecord(Base):
    __tablename__ = "agent2_forecast_envelopes"

    id = Column(String(64), primary_key=True)
    analysis_id = Column(String(64), index=True, nullable=False)
    horizon_hours = Column(Integer, nullable=False)
    target_timestamp = Column(DateTime(timezone=True), nullable=False)
    uncertainty_km = Column(Float, nullable=False)
    area_km2_50 = Column(Float, nullable=True)
    area_km2_90 = Column(Float, nullable=True)
    if GEOALCHEMY_AVAILABLE:
        centroid_geom = Column(Geometry("POINT", srid=4326), nullable=True)
        core_50_geom = Column(Geometry("POLYGON", srid=4326), nullable=True)
        spread_90_geom = Column(Geometry("POLYGON", srid=4326), nullable=True)
    else:
        centroid_geom = Column(Text, nullable=True)
        core_50_geom = Column(Text, nullable=True)
        spread_90_geom = Column(Text, nullable=True)


class XGBoostResidualRecord(Base):
    __tablename__ = "agent2_xgboost_residuals"

    id = Column(String(64), primary_key=True)
    analysis_id = Column(String(64), index=True, nullable=False)
    dx_residual_meters = Column(Float, nullable=False)
    dy_residual_meters = Column(Float, nullable=False)
    uncertainty_scale = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class PostGISStorage:
    """
    PostgreSQL / PostGIS persistence manager and spatial SQL script generator.
    Allows seamless synchronization to live PostgreSQL databases and generation
    of standard PostGIS DDL and spatial insert scripts.
    """

    def __init__(self, connection_url: Optional[str] = None):
        self.connection_url = connection_url
        self.engine = None
        self.Session = None

        if self.connection_url:
            self._init_db()

    def _init_db(self):
        """Initializes SQLAlchemy engine and session factory."""
        try:
            self.engine = create_engine(self.connection_url, echo=False)
            self.Session = sessionmaker(bind=self.engine)
            # Ensure postgis extension
            with self.engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
                conn.commit()
            Base.metadata.create_all(self.engine)
            logger.info("Connected to PostGIS database and verified tables.")
        except Exception as e:
            logger.warning(f"Could not connect to PostGIS at {self.connection_url}: {e}")

    @staticmethod
    def generate_ddl_sql() -> str:
        """
        Generates standard PostGIS DDL schema creation script with GIST spatial indexes.
        """
        return """-- ==========================================================
-- OceanTrace Agent 2: PostgreSQL / PostGIS Spatial Database Schema
-- ==========================================================

-- Enable PostGIS spatial extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- 1. Spill Observations Table
CREATE TABLE IF NOT EXISTS agent2_spills (
    id VARCHAR(64) PRIMARY KEY,
    spill_id VARCHAR(64) NOT NULL,
    observation_timestamp TIMESTAMPTZ NOT NULL,
    area_km2 DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    source_satellite VARCHAR(64),
    centroid_geom GEOMETRY(POINT, 4326),
    polygon_geom GEOMETRY(POLYGON, 4326)
);
CREATE INDEX IF NOT EXISTS idx_agent2_spills_centroid ON agent2_spills USING GIST (centroid_geom);
CREATE INDEX IF NOT EXISTS idx_agent2_spills_polygon ON agent2_spills USING GIST (polygon_geom);
CREATE INDEX IF NOT EXISTS idx_agent2_spills_spill_id ON agent2_spills (spill_id);

-- 2. Probable Origin Regions Table (Hindcast Output)
CREATE TABLE IF NOT EXISTS agent2_origin_regions (
    id VARCHAR(64) PRIMARY KEY,
    analysis_id VARCHAR(64) NOT NULL,
    spill_id VARCHAR(64) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    area_km2 DOUBLE PRECISION NOT NULL,
    time_window_start TIMESTAMPTZ NOT NULL,
    time_window_end TIMESTAMPTZ NOT NULL,
    envelope_geom GEOMETRY(POLYGON, 4326)
);
CREATE INDEX IF NOT EXISTS idx_agent2_origin_regions_geom ON agent2_origin_regions USING GIST (envelope_geom);
CREATE INDEX IF NOT EXISTS idx_agent2_origin_regions_analysis ON agent2_origin_regions (analysis_id);

-- 3. Hindcast Trajectories Table
CREATE TABLE IF NOT EXISTS agent2_hindcast_trajectories (
    id VARCHAR(64) PRIMARY KEY,
    analysis_id VARCHAR(64) NOT NULL,
    candidate_id VARCHAR(64) NOT NULL,
    rank INTEGER NOT NULL,
    composite_score DOUBLE PRECISION NOT NULL,
    centroid_distance_km DOUBLE PRECISION NOT NULL,
    trajectory_geom GEOMETRY(LINESTRING, 4326)
);
CREATE INDEX IF NOT EXISTS idx_agent2_hindcast_traj_geom ON agent2_hindcast_trajectories USING GIST (trajectory_geom);
CREATE INDEX IF NOT EXISTS idx_agent2_hindcast_analysis ON agent2_hindcast_trajectories (analysis_id);

-- 4. Forecast Envelopes Table (Forecast Output)
CREATE TABLE IF NOT EXISTS agent2_forecast_envelopes (
    id VARCHAR(64) PRIMARY KEY,
    analysis_id VARCHAR(64) NOT NULL,
    horizon_hours INTEGER NOT NULL,
    target_timestamp TIMESTAMPTZ NOT NULL,
    uncertainty_km DOUBLE PRECISION NOT NULL,
    area_km2_50 DOUBLE PRECISION,
    area_km2_90 DOUBLE PRECISION,
    centroid_geom GEOMETRY(POINT, 4326),
    core_50_geom GEOMETRY(POLYGON, 4326),
    spread_90_geom GEOMETRY(POLYGON, 4326)
);
CREATE INDEX IF NOT EXISTS idx_agent2_forecast_centroid ON agent2_forecast_envelopes USING GIST (centroid_geom);
CREATE INDEX IF NOT EXISTS idx_agent2_forecast_core ON agent2_forecast_envelopes USING GIST (core_50_geom);
CREATE INDEX IF NOT EXISTS idx_agent2_forecast_spread ON agent2_forecast_envelopes USING GIST (spread_90_geom);
CREATE INDEX IF NOT EXISTS idx_agent2_forecast_analysis ON agent2_forecast_envelopes (analysis_id);

-- 5. XGBoost Physics Residual Correction Table
CREATE TABLE IF NOT EXISTS agent2_xgboost_residuals (
    id VARCHAR(64) PRIMARY KEY,
    analysis_id VARCHAR(64) NOT NULL,
    dx_residual_meters DOUBLE PRECISION NOT NULL,
    dy_residual_meters DOUBLE PRECISION NOT NULL,
    uncertainty_scale DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_agent2_xgboost_analysis ON agent2_xgboost_residuals (analysis_id);
"""

    def export_sql_script(self, result: Agent2Result) -> str:
        """
        Converts an Agent2Result into valid PostGIS SQL insert statements using
        ST_SetSRID(ST_GeomFromGeoJSON(...), 4326).
        """
        sql_lines = [
            "-- OceanTrace Agent 2 Analysis Spatial Export",
            f"-- Analysis ID: {result.analysis_id}",
            f"-- Generated: {result.execution_timestamp.isoformat()}",
            "BEGIN;",
            ""
        ]

        ref = result.input_reference
        spill_id = ref.get("spill_id", "unknown_spill")
        obs_time = ref.get("observation_timestamp", datetime.now(timezone.utc).isoformat())
        area = float(ref.get("area_km2", 0.0))
        conf = float(ref.get("confidence", 0.9))
        sat = ref.get("source_satellite", "Sentinel-1")
        centroid = ref.get("centroid", [0.0, 0.0])

        point_geojson = json.dumps({"type": "Point", "coordinates": [centroid[0], centroid[1]]})
        spill_rec_id = f"spill_{uuid.uuid4().hex[:8]}"

        sql_lines.append(
            f"INSERT INTO agent2_spills (id, spill_id, observation_timestamp, area_km2, confidence, source_satellite, centroid_geom) "
            f"VALUES ('{spill_rec_id}', '{spill_id}', '{obs_time}', {area}, {conf}, '{sat}', "
            f"ST_SetSRID(ST_GeomFromGeoJSON('{point_geojson}'), 4326)) "
            f"ON CONFLICT (id) DO NOTHING;\n"
        )

        # Origin Region
        if result.hindcast.enabled and result.hindcast.probable_origin:
            orig = result.hindcast.probable_origin
            orig_id = f"orig_{uuid.uuid4().hex[:8]}"
            orig_geom = json.dumps(orig.geometry_geojson)
            sql_lines.append(
                f"INSERT INTO agent2_origin_regions (id, analysis_id, spill_id, confidence, area_km2, time_window_start, time_window_end, envelope_geom) "
                f"VALUES ('{orig_id}', '{result.analysis_id}', '{spill_id}', {orig.confidence}, {orig.area_km2}, "
                f"'{orig.time_window_start.isoformat()}', '{orig.time_window_end.isoformat()}', "
                f"ST_SetSRID(ST_GeomFromGeoJSON('{orig_geom}'), 4326));\n"
            )

        # Hindcast trajectories
        if result.hindcast.enabled and result.hindcast.candidate_hypotheses:
            for rank_idx, cand in enumerate(result.hindcast.candidate_hypotheses[:5]):
                if len(cand.trajectory) > 1:
                    traj_id = f"traj_{uuid.uuid4().hex[:8]}"
                    coords = [[pt.longitude, pt.latitude] for pt in cand.trajectory]
                    traj_geojson = json.dumps({"type": "LineString", "coordinates": coords})
                    sql_lines.append(
                        f"INSERT INTO agent2_hindcast_trajectories (id, analysis_id, candidate_id, rank, composite_score, centroid_distance_km, trajectory_geom) "
                        f"VALUES ('{traj_id}', '{result.analysis_id}', '{cand.candidate_id}', {rank_idx + 1}, {cand.composite_score}, {cand.centroid_distance_km}, "
                        f"ST_SetSRID(ST_GeomFromGeoJSON('{traj_geojson}'), 4326));\n"
                    )

        # Forecast Horizons
        if result.forecast.enabled and result.forecast.horizons:
            for hor in result.forecast.horizons:
                f_id = f"fc_{uuid.uuid4().hex[:8]}"
                c_geom = json.dumps({"type": "Point", "coordinates": [hor.centroid[0], hor.centroid[1]]})
                core_geom = json.dumps(hor.core_envelope_50_geojson)
                spread_geom = json.dumps(hor.spread_envelope_90_geojson)
                sql_lines.append(
                    f"INSERT INTO agent2_forecast_envelopes (id, analysis_id, horizon_hours, target_timestamp, uncertainty_km, area_km2_50, area_km2_90, centroid_geom, core_50_geom, spread_90_geom) "
                    f"VALUES ('{f_id}', '{result.analysis_id}', {hor.horizon_hours}, '{hor.target_timestamp.isoformat()}', {hor.uncertainty_radius_km}, "
                    f"{hor.area_km2_core_50}, {hor.area_km2_spread_90}, "
                    f"ST_SetSRID(ST_GeomFromGeoJSON('{c_geom}'), 4326), "
                    f"ST_SetSRID(ST_GeomFromGeoJSON('{core_geom}'), 4326), "
                    f"ST_SetSRID(ST_GeomFromGeoJSON('{spread_geom}'), 4326));\n"
                )

        # XGBoost Residual
        xgb_res = getattr(result, "ml_residual", None) or result.diagnostics.get("xgboost_residual_correction", {})
        if xgb_res:
            xgb_id = f"xgb_{uuid.uuid4().hex[:8]}"
            dx = float(xgb_res.get("dx_residual_meters", 0.0))
            dy = float(xgb_res.get("dy_residual_meters", 0.0))
            unc = float(xgb_res.get("uncertainty_scale", 1.0))
            sql_lines.append(
                f"INSERT INTO agent2_xgboost_residuals (id, analysis_id, dx_residual_meters, dy_residual_meters, uncertainty_scale) "
                f"VALUES ('{xgb_id}', '{result.analysis_id}', {dx}, {dy}, {unc});\n"
            )

        sql_lines.append("COMMIT;\n")
        return "\n".join(sql_lines)

    def export_to_file(self, result: Agent2Result, output_path: str) -> str:
        """Writes PostGIS SQL script to disk."""
        sql_content = self.export_sql_script(result)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(sql_content)
        return output_path
