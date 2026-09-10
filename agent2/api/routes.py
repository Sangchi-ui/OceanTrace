"""
FastAPI Route Handlers for Agent 2: Hindcasting & Forecasting.
Exposes POST /analyze to run physics-based trajectory simulations
from an uploaded Agent 1 result or JSON payload.
"""

import json
import os
import shutil
import tempfile
from typing import Any, Dict, Optional
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, Body

from agent2.adapters.agent1_adapter import Agent1Adapter
from agent2.contracts.agent2_result import Agent2Result
from agent2.pipeline import Agent2Pipeline
from agent2.errors import Agent2Error

from agent2.storage.postgis_adapter import PostGISStorage

agent2_router = APIRouter(tags=["Agent 2 - Hindcasting & Forecasting"])


@agent2_router.get("/health", summary="Agent 2 Health Check")
def agent2_health():
    return {
        "status": "healthy",
        "agent": "Agent 2 - Hindcasting & Forecasting",
        "version": "2.0.0",
        "techstack": {
            "core": "Python",
            "drift_physics": "Lagrangian Transport (RK4 / OpenDrift/OpenOil)",
            "ocean_forcing": "CMEMS (Xarray / NetCDF)",
            "atmospheric_forcing": "ERA5 / NOAA GFS (Xarray / NetCDF)",
            "scientific_compute": "NumPy / SciPy",
            "spatial_processing": "GeoPandas / Shapely",
            "machine_learning": "XGBoost (Mandatory Residual & Uncertainty Calibration)",
            "spatial_database": "PostgreSQL / PostGIS (GeoAlchemy2)",
            "api_framework": "FastAPI"
        }
    }


@agent2_router.get("/postgis/schema", summary="Export PostGIS Schema DDL")
def get_postgis_schema():
    """Returns the standard PostgreSQL / PostGIS spatial table DDL script."""
    return {"ddl_sql": PostGISStorage.generate_ddl_sql()}


@agent2_router.post(
    "/analyze",
    response_model=Agent2Result,
    summary="Execute Hindcasting & Forecasting on Detected Spill"
)
async def analyze_spill(
    payload: Optional[Dict[str, Any]] = Body(default=None, description="SpillEvent or SpillObservation JSON"),
    mode: str = Query("both", pattern="^(both|hindcast|forecast)$", description="Analysis mode"),
    config_path: str = Query("configs/agent2.yaml", description="Path to Agent 2 config")
):
    """
    Takes an Agent 1 SpillEvent or SpillObservation JSON payload,
    simulates backward candidate origins (Hindcast) and multi-horizon forward dispersion (Forecast),
    and returns an Agent2Result with GIS GeoJSON features.
    """
    if not payload:
        raise HTTPException(status_code=400, detail="Missing JSON body payload.")

    try:
        pipeline = Agent2Pipeline(
            config_path=config_path if os.path.exists(config_path) else None
        )
        spill = Agent1Adapter.parse(payload)
        result = pipeline.run(spill, mode=mode)
        return result
    except Agent2Error as ae:
        raise HTTPException(status_code=422, detail=f"Agent 2 validation error: {ae}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 2 execution failed: {e}")


@agent2_router.post(
    "/analyze-file",
    response_model=Agent2Result,
    summary="Execute Hindcasting & Forecasting on Uploaded Detection File"
)
async def analyze_file(
    file: UploadFile = File(...),
    mode: str = Query("both", pattern="^(both|hindcast|forecast)$"),
    config_path: str = Query("configs/agent2.yaml")
):
    """
    Accepts an uploaded spill_event.json or spill.geojson file and runs Agent 2.
    """
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pipeline = Agent2Pipeline(
            config_path=config_path if os.path.exists(config_path) else None
        )
        result = pipeline.run_from_file(temp_path, mode=mode)
        return result
    except Agent2Error as ae:
        raise HTTPException(status_code=422, detail=f"Validation error: {ae}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
