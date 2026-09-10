import json
import os
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List

from contracts.system_contracts import (
    Agent1Output, DetectionResult, SpillGeometry, Centroid,
    ObservationMetadata, SegmentationMetadata, QualityControl,
    Agent2Input, EnvironmentalData, Agent2Output, InputReference,
    OriginEstimation, OriginTimeWindow, ForecastResult, HorizonForecast,
    UncertaintyResult, SpillEvolution, EnvironmentSummary,
    Agent3Input, AISData, Agent3Output, CandidateVessel, EvidenceScore,
    BehaviourFlags, Explanation, AttributionSummary
)

def adapt_to_agent1_output(spill_event: Any, prob_map: np.ndarray) -> Agent1Output:
    """
    Converts internal SpillEvent to standardized Agent1Output contract.
    """
    event_dict = spill_event.model_dump()
    regions = event_dict.get("spill_regions", [])
    
    # If no regions, create an empty/invalid placeholder
    if not regions:
        return Agent1Output(
            event_id=event_dict["event_id"],
            status="INVALID",
            detection=DetectionResult(
                is_potential_oil_spill=False,
                overall_confidence=0.0,
                pixel_probability_mean=float(np.mean(prob_map)),
                pixel_probability_max=float(np.max(prob_map))
            ),
            geometry=SpillGeometry(
                polygon_geojson={"type": "Polygon", "coordinates": []},
                centroid=Centroid(latitude=0.0, longitude=0.0),
                area_km2=0.0,
                perimeter_km=0.0
            ),
            observation=ObservationMetadata(
                acquisition_time=event_dict["source"]["acquisition_time"] or event_dict["timestamp"],
                sensor=event_dict["source"]["sensor"],
                source_image=event_dict["source"]["image_path"],
                crs="EPSG:4326"
            ),
            segmentation=SegmentationMetadata(
                threshold=0.5,
                component_count=0
            ),
            quality_control=QualityControl(
                status="INVALID",
                warnings=["No spill regions detected"]
            )
        )
        
    # Use the largest region for the top-level contract (simplification for contract mapping)
    best_region = sorted(regions, key=lambda x: x["measurements"]["area_km2"], reverse=True)[0]
    
    # Calculate pixel probabilities over the entire prob_map or just the mask
    # For contract, we'll provide the image-wide max and mean if mask isn't provided,
    # but the region confidence is already the mean over the mask.
    mean_prob = best_region["detection"]["confidence"]
    max_prob = float(np.max(prob_map))
    
    return Agent1Output(
        event_id=event_dict["event_id"],
        status=event_dict["validation"]["status"],
        detection=DetectionResult(
            is_potential_oil_spill=True,
            overall_confidence=event_dict["overall_confidence"],
            pixel_probability_mean=mean_prob,
            pixel_probability_max=max_prob
        ),
        geometry=SpillGeometry(
            polygon_geojson=best_region["geometry"]["polygon_geojson"],
            centroid=Centroid(
                latitude=best_region["geometry"]["centroid"]["latitude"] or 0.0,
                longitude=best_region["geometry"]["centroid"]["longitude"] or 0.0
            ),
            area_km2=best_region["measurements"]["area_km2"],
            perimeter_km=best_region["measurements"]["perimeter_km"]
        ),
        observation=ObservationMetadata(
            acquisition_time=event_dict["source"].get("acquisition_time") or event_dict["timestamp"],
            sensor=event_dict["source"]["sensor"],
            source_image=event_dict["source"]["image_path"],
            crs=best_region["geometry"].get("crs", "EPSG:4326")
        ),
        segmentation=SegmentationMetadata(
            threshold=best_region["detection"]["threshold"],
            component_count=len(regions)
        ),
        quality_control=QualityControl(
            status=event_dict["validation"]["status"],
            warnings=event_dict["validation"]["warnings"]
        )
    )

def adapt_to_agent2_output(agent2_result: Any) -> Agent2Output:
    """
    Converts internal Agent2Result to standardized Agent2Output contract.
    """
    res = agent2_result.model_dump()
    
    h = res.get("hindcast", {})
    f = res.get("forecast", {})
    
    po = h.get("probable_origin", {})
    
    # Map origin
    def to_iso(dt: Any) -> str:
        if isinstance(dt, datetime):
            return dt.isoformat()
        return str(dt) if dt else ""

    origin_estimation = OriginEstimation(
        probable_origin_region=po.get("polygon", {}),
        origin_centroid=Centroid(
            longitude=po.get("centroid", [0.0, 0.0])[0] if po.get("centroid") else 0.0,
            latitude=po.get("centroid", [0.0, 0.0])[1] if po.get("centroid") else 0.0
        ),
        origin_time_window=OriginTimeWindow(
            start=to_iso(po.get("time_window_start", "")),
            end=to_iso(po.get("time_window_end", ""))
        ),
        origin_probability=po.get("confidence", 0.0)
    )
    
    # Map horizons
    horizons = f.get("horizons", [])
    def get_horizon(hours: int) -> HorizonForecast:
        hz = next((x for x in horizons if x["horizon_hours"] == hours), None)
        if not hz:
            return HorizonForecast(
                drift_geometry={},
                centroid=Centroid(latitude=0.0, longitude=0.0),
                probability=0.0
            )
        return HorizonForecast(
            drift_geometry=hz.get("spread_polygon_90", {}),
            centroid=Centroid(
                longitude=hz.get("centroid", [0.0, 0.0])[0],
                latitude=hz.get("centroid", [0.0, 0.0])[1]
            ),
            probability=0.9
        )

    forecast_res = ForecastResult(
        plus_6h=get_horizon(6),
        plus_12h=get_horizon(12),
        plus_24h=get_horizon(24)
    )
    
    # Uncertainty
    uncertainty = UncertaintyResult(
        probability_map={},
        confidence=res.get("ml_residual", {}).get("confidence_scaling", 1.0),
        uncertainty_radius_km=f.get("horizons", [{}])[-1].get("spread_area_km2", 0.0) ** 0.5 if f.get("horizons") else 0.0
    )
    
    # Spill Evolution
    spill_evolution = SpillEvolution(
        initial_area_km2=res.get("input_reference", {}).get("area_km2", 0.0),
        predicted_area_km2={f"+{hz['horizon_hours']}h": hz["spread_area_km2"] for hz in horizons},
        estimated_age_hours=0.0, # Could be derived from time window
        weathering=f.get("weathering_timeseries", [{}])[-1] if f.get("weathering_timeseries") else {},
        thickness_estimate={"mean_mm": 0.1},
        evolution_parameters={}
    )
    
    # Environment
    env = res.get("forcing_summary", {})
    environment = EnvironmentSummary(
        current_source=env.get("currents", "unknown"),
        wind_source=env.get("wind", "unknown"),
        wave_source=None,
        data_timestamp=to_iso(res.get("execution_timestamp", ""))
    )
    
    return Agent2Output(
        event_id=res["analysis_id"],
        input_reference=InputReference(
            agent1_event_id=res.get("input_reference", {}).get("spill_id", ""),
            acquisition_time=res.get("input_reference", {}).get("observation_time", "")
        ),
        origin_estimation=origin_estimation,
        forecast=forecast_res,
        uncertainty=uncertainty,
        spill_evolution=spill_evolution,
        environment=environment,
        quality_control=QualityControl(
            status="VALID",
            warnings=res.get("diagnostics", {}).get("warnings", [])
        )
    )
