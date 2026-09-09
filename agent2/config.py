"""
Configuration loader and validator for Agent 2.
Parses YAML files or environment settings into structured configuration objects.
"""

import os
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field

from agent2.errors import ConfigurationError


class HindcastConfig(BaseModel):
    enabled: bool = True
    max_lookback_hours: int = Field(default=48, ge=2, le=168)
    time_step_hours: int = Field(default=6, ge=1, le=24)
    cross_track_samples: int = Field(default=3, ge=1)
    particles_per_candidate: int = Field(default=150, ge=10)
    timestep_minutes: float = Field(default=20.0, ge=1.0)


class ForecastConfig(BaseModel):
    enabled: bool = True
    horizons_hours: List[int] = Field(default=[6, 12, 24])
    particle_count: int = Field(default=500, ge=20)
    ensemble_size: int = Field(default=15, ge=1)
    timestep_minutes: float = Field(default=15.0, ge=1.0)


class WindageConfig(BaseModel):
    enabled: bool = True
    percentage: float = Field(default=0.030, ge=0.01, le=0.06)
    min: float = Field(default=0.025)
    max: float = Field(default=0.044)


class ScoringWeightsConfig(BaseModel):
    centroid_weight: float = Field(default=0.35, ge=0.0)
    iou_weight: float = Field(default=0.35, ge=0.0)
    area_weight: float = Field(default=0.15, ge=0.0)
    shape_weight: float = Field(default=0.15, ge=0.0)
    centroid_sigma_km: float = Field(default=3.5, gt=0.0)


class MockForcingConfig(BaseModel):
    mode: str = "uniform"
    current_u: float = 0.25
    current_v: float = 0.15
    wind_u: float = 4.0
    wind_v: float = 2.0
    gyre_center: List[float] = [80.0, 15.0]
    gyre_radius_km: float = 40.0
    gyre_max_speed: float = 0.5


class ForcingConfig(BaseModel):
    provider: str = "mock"
    mock: MockForcingConfig = Field(default_factory=MockForcingConfig)
    cmems: Dict[str, Any] = Field(default_factory=dict)
    era5: Dict[str, Any] = Field(default_factory=dict)


class Agent2Config(BaseModel):
    version: str = "2.0.0"
    hindcast: HindcastConfig = Field(default_factory=HindcastConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    oil_category: str = "medium_crude"
    windage: WindageConfig = Field(default_factory=WindageConfig)
    scoring: ScoringWeightsConfig = Field(default_factory=ScoringWeightsConfig)
    forcing: ForcingConfig = Field(default_factory=ForcingConfig)
    seed: Optional[int] = 42

    @classmethod
    def from_yaml(cls, path: str) -> "Agent2Config":
        if not os.path.exists(path):
            raise ConfigurationError(f"Configuration file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            
            # Map top-level keys if nested
            cfg_dict = {}
            if "hindcast" in data:
                cfg_dict["hindcast"] = data["hindcast"]
            if "forecast" in data:
                cfg_dict["forecast"] = data["forecast"]
            if "oil" in data:
                cfg_dict["oil_category"] = data["oil"].get("category", "medium_crude")
            if "windage" in data:
                w_data = data["windage"]
                unc = w_data.get("uncertainty", {})
                cfg_dict["windage"] = {
                    "enabled": w_data.get("enabled", True),
                    "percentage": w_data.get("percentage", 0.030),
                    "min": unc.get("min", 0.025),
                    "max": unc.get("max", 0.044)
                }
            if "scoring" in data:
                cfg_dict["scoring"] = data["scoring"]
            if "forcing" in data:
                cfg_dict["forcing"] = data["forcing"]
            if "reproducibility" in data:
                cfg_dict["seed"] = data["reproducibility"].get("seed", 42)

            return cls(**cfg_dict)
        except Exception as e:
            raise ConfigurationError(f"Failed to parse Agent 2 config from {path}: {e}")
