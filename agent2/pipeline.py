"""
High-Level Pipeline Orchestrator for Agent 2: Hindcasting & Forecasting.
Connects input ingestion (Agent1Adapter), environmental forcing, physics engines,
scoring, GeoJSON production, and output serialization.
"""

from datetime import datetime, timezone
import json
import logging
import os
import time
from typing import Any, Dict, Optional, Union
import uuid

from agent2.adapters.agent1_adapter import Agent1Adapter
from agent2.adapters.cmems_adapter import CMEMSCurrentProvider
from agent2.adapters.composite_forcing import CompositeForcingProvider
from agent2.adapters.era5_adapter import ERA5WindProvider
from agent2.adapters.forcing_base import EnvironmentalForcingProvider
from agent2.adapters.mock_forcing import MockEnvironmentalForcingProvider
from agent2.config import Agent2Config
from agent2.contracts.agent2_result import Agent2Result
from agent2.contracts.forecast_result import ForecastResult
from agent2.contracts.hindcast_result import HindcastResult
from agent2.contracts.spill_observation import SpillObservation
from agent2.forecast.forecast_engine import ForecastingEngine
from agent2.geospatial.geojson_export import build_unified_geojson, create_feature, create_feature_collection
from agent2.hindcast.hindcast_engine import HindcastingEngine
from agent2.hindcast.scoring import ScoringConfig
from agent2.physics.oil_properties import get_oil_properties

logger = logging.getLogger("OceanTrace.Agent2.Pipeline")


class Agent2Pipeline:
    """
    Main entry point and orchestrator for Agent 2.
    """

    def __init__(
        self,
        config: Optional[Agent2Config] = None,
        config_path: Optional[str] = None,
        forcing_provider: Optional[EnvironmentalForcingProvider] = None
    ):
        if config is not None:
            self.config = config
        elif config_path and os.path.exists(config_path):
            self.config = Agent2Config.from_yaml(config_path)
        else:
            default_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "configs", "agent2.yaml"
            )
            if os.path.exists(default_path):
                self.config = Agent2Config.from_yaml(default_path)
            else:
                self.config = Agent2Config()

        self.oil_props = get_oil_properties(self.config.oil_category)
        self.forcing_provider = forcing_provider or self._build_forcing_provider()

    def _build_forcing_provider(self) -> EnvironmentalForcingProvider:
        """Instantiates the configured environmental forcing provider."""
        prov_type = self.config.forcing.provider.lower()

        if prov_type == "mock":
            m = self.config.forcing.mock
            return MockEnvironmentalForcingProvider(
                mode=m.mode,
                current_u=m.current_u,
                current_v=m.current_v,
                wind_u=m.wind_u,
                wind_v=m.wind_v,
                gyre_center=(m.gyre_center[0], m.gyre_center[1]),
                gyre_radius_km=m.gyre_radius_km,
                gyre_max_speed=m.gyre_max_speed,
                seed=self.config.seed
            )

        elif prov_type == "cmems":
            return CMEMSCurrentProvider(
                dataset_path=self.config.forcing.cmems.get("dataset_path")
            )

        elif prov_type == "composite":
            c_prov = CMEMSCurrentProvider(
                dataset_path=self.config.forcing.cmems.get("dataset_path")
            )
            w_prov = ERA5WindProvider(
                dataset_path=self.config.forcing.era5.get("dataset_path")
            )
            return CompositeForcingProvider(c_prov, w_prov)

        else:
            logger.warning(f"Unknown forcing provider '{prov_type}', falling back to Mock.")
            return MockEnvironmentalForcingProvider(seed=self.config.seed)

    def run(
        self,
        spill: SpillObservation,
        mode: str = "both"
    ) -> Agent2Result:
        """
        Executes complete Agent 2 analysis for a given SpillObservation.
        mode: 'both', 'hindcast', or 'forecast'
        """
        t_start = time.time()
        analysis_id = f"A2_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        run_hindcast = mode in ["both", "hindcast"] and self.config.hindcast.enabled
        run_forecast = mode in ["both", "forecast"] and self.config.forecast.enabled

        hindcast_res: Optional[HindcastResult] = None
        forecast_res: Optional[ForecastResult] = None

        # 1. Hindcast Execution
        if run_hindcast:
            h_cfg = self.config.hindcast
            scoring_cfg = ScoringConfig(
                weight_centroid=self.config.scoring.centroid_weight,
                weight_iou=self.config.scoring.iou_weight,
                weight_area=self.config.scoring.area_weight,
                weight_shape=self.config.scoring.shape_weight,
                centroid_sigma_km=self.config.scoring.centroid_sigma_km
            )
            hind_engine = HindcastingEngine(
                forcing_provider=self.forcing_provider,
                oil_props=self.oil_props,
                max_lookback_hours=h_cfg.max_lookback_hours,
                time_step_hours=h_cfg.time_step_hours,
                particles_per_candidate=h_cfg.particles_per_candidate,
                scoring_config=scoring_cfg,
                timestep_minutes=h_cfg.timestep_minutes,
                seed=self.config.seed
            )
            hindcast_res = hind_engine.run(spill)
        else:
            hindcast_res = HindcastResult(
                enabled=False,
                status="SKIPPED",
                probable_origin=None,
                candidate_hypotheses=[]
            )

        # 2. Forecast Execution
        if run_forecast:
            f_cfg = self.config.forecast
            fore_engine = ForecastingEngine(
                forcing_provider=self.forcing_provider,
                oil_props=self.oil_props,
                horizons_hours=f_cfg.horizons_hours,
                particle_count=f_cfg.particle_count,
                ensemble_size=f_cfg.ensemble_size,
                timestep_minutes=f_cfg.timestep_minutes,
                seed=self.config.seed
            )
            forecast_res = fore_engine.run(spill)
        else:
            forecast_res = ForecastResult(
                enabled=False,
                status="SKIPPED",
                horizons=[]
            )

        # 3. Construct Unified GIS GeoJSON
        unified_geojson = build_unified_geojson(
            observed_spill=spill,
            hindcast_result=hindcast_res if run_hindcast else None,
            forecast_result=forecast_res if run_forecast else None
        )

        total_runtime = round(time.time() - t_start, 3)

        result = Agent2Result(
            analysis_id=analysis_id,
            agent_version=self.config.version,
            execution_timestamp=datetime.now(timezone.utc),
            input_reference={
                "spill_id": spill.spill_id,
                "observation_timestamp": spill.observation_timestamp.isoformat(),
                "centroid": spill.centroid,
                "area_km2": spill.area_km2,
                "confidence": spill.confidence,
                "source_satellite": spill.source_satellite
            },
            hindcast=hindcast_res,
            forecast=forecast_res,
            forcing_summary=self.forcing_provider.metadata(),
            diagnostics={
                "total_runtime_seconds": total_runtime,
                "mode": mode,
                "seed": self.config.seed,
                "oil_category": self.config.oil_category
            },
            geojson_collection=unified_geojson
        )

        return result

    def run_from_file(
        self,
        input_path: str,
        output_dir: Optional[str] = None,
        mode: str = "both",
        region_index: int = 0
    ) -> Agent2Result:
        """
        Loads an Agent 1 output file, runs Agent 2, and saves JSON and GeoJSON outputs.
        """
        spill = Agent1Adapter.from_file(input_path, region_index=region_index)
        result = self.run(spill, mode=mode)

        if output_dir:
            self.export_artifacts(result, output_dir)

        return result

    def export_artifacts(self, result: Agent2Result, output_dir: str) -> Dict[str, str]:
        """
        Saves Agent2Result JSON and modular GeoJSON files.
        """
        os.makedirs(output_dir, exist_ok=True)
        geojson_dir = os.path.join(output_dir, "geojson")
        os.makedirs(geojson_dir, exist_ok=True)

        paths = {}

        # 1. Main JSON result
        json_path = os.path.join(output_dir, "agent2_result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
        paths["result_json"] = json_path

        # 2. Unified GeoJSON
        unified_path = os.path.join(output_dir, "agent2_analysis.geojson")
        with open(unified_path, "w", encoding="utf-8") as f:
            json.dump(result.geojson_collection, f, indent=2)
        paths["unified_geojson"] = unified_path

        # 3. Individual GeoJSON components
        # Probable Origin Region
        if result.hindcast.enabled and result.hindcast.probable_origin:
            origin_feat = create_feature(
                geometry=result.hindcast.probable_origin.geometry_geojson,
                properties={
                    "type": "probable_origin_region",
                    "confidence": result.hindcast.probable_origin.confidence,
                    "area_km2": result.hindcast.probable_origin.area_km2,
                    "time_window_start": result.hindcast.probable_origin.time_window_start.isoformat(),
                    "time_window_end": result.hindcast.probable_origin.time_window_end.isoformat()
                }
            )
            origin_path = os.path.join(geojson_dir, "origin_region.geojson")
            with open(origin_path, "w", encoding="utf-8") as f:
                json.dump(create_feature_collection([origin_feat]), f, indent=2)
            paths["origin_region_geojson"] = origin_path

        # Hindcast Paths
        if result.hindcast.enabled and result.hindcast.best_candidate:
            best = result.hindcast.best_candidate
            if len(best.trajectory) > 1:
                coords = [[pt.longitude, pt.latitude] for pt in best.trajectory]
                path_feat = create_feature(
                    geometry={"type": "LineString", "coordinates": coords},
                    properties={
                        "candidate_id": best.candidate_id,
                        "composite_score": best.composite_score,
                        "centroid_distance_km": best.centroid_distance_km
                    }
                )
                paths_file = os.path.join(geojson_dir, "hindcast_paths.geojson")
                with open(paths_file, "w", encoding="utf-8") as f:
                    json.dump(create_feature_collection([path_feat]), f, indent=2)
                paths["hindcast_paths_geojson"] = paths_file

        # Forecast Horizons
        if result.forecast.enabled and result.forecast.horizons:
            for hor in result.forecast.horizons:
                h_file = os.path.join(geojson_dir, f"forecast_{hor.horizon_hours}h.geojson")
                f_feats = [
                    create_feature(hor.core_envelope_50_geojson, {"layer": "core_50", "hours": hor.horizon_hours}),
                    create_feature(hor.spread_envelope_90_geojson, {"layer": "spread_90", "hours": hor.horizon_hours}),
                    create_feature({"type": "Point", "coordinates": [hor.centroid[0], hor.centroid[1]]}, {"layer": "centroid"})
                ]
                with open(h_file, "w", encoding="utf-8") as f:
                    json.dump(create_feature_collection(f_feats), f, indent=2)
                paths[f"forecast_{hor.horizon_hours}h_geojson"] = h_file

        return paths
