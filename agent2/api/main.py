"""
FastAPI Application for Agent 2: Hindcasting & Forecasting Service.
Can be executed independently via:
    uvicorn agent2.api.main:app --port 8001 --reload
"""

import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agent2.api.routes import agent2_router

app = FastAPI(
    title="OceanTrace — Agent 2 Hindcasting & Forecasting API",
    description="Physics-based Lagrangian particle transport, two-stage source estimation, and ensemble dispersion forecasting",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent2_router, prefix="/api/v1/agent2")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent2.api.main:app", host="0.0.0.0", port=8001, reload=True)
