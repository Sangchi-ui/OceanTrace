import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))


app = FastAPI(
    title="OceanTrace — Agent 1 Sentinel-1 SAR Oil Spill API",
    description="Computational AI Agent for Sentinel-1 SAR Oil Spill Segmentation, Geometry Extraction, and Event Structuring",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
