from typing import Any, List, Optional, Dict
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Common Components
# ---------------------------------------------------------

class Centroid(BaseModel):
    latitude: float
    longitude: float

class QualityControl(BaseModel):
    status: str
    warnings: List[str] = Field(default_factory=list)

# ---------------------------------------------------------
# AGENT 1 - OIL SPILL DETECTION
# ---------------------------------------------------------

class DetectionResult(BaseModel):
    is_potential_oil_spill: bool
    overall_confidence: float
    pixel_probability_mean: float
    pixel_probability_max: float

class SpillGeometry(BaseModel):
    polygon_geojson: Dict[str, Any]
    centroid: Centroid
    area_km2: float
    perimeter_km: float

class ObservationMetadata(BaseModel):
    acquisition_time: str
    sensor: str
    source_image: str
    crs: str

class SegmentationMetadata(BaseModel):
    threshold: float
    component_count: int

class Agent1Output(BaseModel):
    event_id: str
    status: str
    detection: DetectionResult
    geometry: SpillGeometry
    observation: ObservationMetadata
    segmentation: SegmentationMetadata
    quality_control: QualityControl

# ---------------------------------------------------------
# AGENT 2 - DRIFT MODELLING
# ---------------------------------------------------------

class EnvironmentalData(BaseModel):
    ocean_currents: str
    wind: str
    waves: Optional[str] = None
    sea_surface_temperature: str
    bathymetry: Optional[str] = None
    source: str
    time_range: str

class Agent2Input(BaseModel):
    spill_event: Agent1Output
    environment: EnvironmentalData

class InputReference(BaseModel):
    agent1_event_id: str
    acquisition_time: str

class OriginTimeWindow(BaseModel):
    start: str
    end: str

class OriginEstimation(BaseModel):
    probable_origin_region: Dict[str, Any]
    origin_centroid: Centroid
    origin_time_window: OriginTimeWindow
    origin_probability: float

class HorizonForecast(BaseModel):
    drift_geometry: Dict[str, Any]
    centroid: Centroid
    probability: float

class ForecastResult(BaseModel):
    plus_6h: HorizonForecast
    plus_12h: HorizonForecast
    plus_24h: HorizonForecast

class UncertaintyResult(BaseModel):
    probability_map: Dict[str, Any]
    confidence: float
    uncertainty_radius_km: float

class SpillEvolution(BaseModel):
    initial_area_km2: float
    predicted_area_km2: Dict[str, float]
    estimated_age_hours: float
    weathering: Dict[str, Any]
    thickness_estimate: Dict[str, Any]
    evolution_parameters: Dict[str, Any]

class EnvironmentSummary(BaseModel):
    current_source: str
    wind_source: str
    wave_source: Optional[str] = None
    data_timestamp: str

class Agent2Output(BaseModel):
    event_id: str
    input_reference: InputReference
    origin_estimation: OriginEstimation
    forecast: ForecastResult
    uncertainty: UncertaintyResult
    spill_evolution: SpillEvolution
    environment: EnvironmentSummary
    quality_control: QualityControl

# ---------------------------------------------------------
# AGENT 3 - VESSEL ATTRIBUTION
# ---------------------------------------------------------

class VesselRecord(BaseModel):
    mmsi: str
    timestamp: str
    latitude: float
    longitude: float
    speed: float
    heading: float
    vessel_identity: Optional[str] = None
    vessel_type: Optional[str] = None
    flag: Optional[str] = None
    historical_trajectory: Optional[List[Dict[str, Any]]] = None

class AISData(BaseModel):
    vessels: List[VesselRecord]

class Agent3Input(BaseModel):
    spill_event: Agent1Output
    drift_result: Agent2Output
    ais_data: AISData

class EvidenceScore(BaseModel):
    minimum_distance_km: float
    time_difference_hours: float
    trajectory_match_score: float
    proximity_score: float
    behaviour_score: float

class BehaviourFlags(BaseModel):
    speed_anomaly: bool
    heading_anomaly: bool
    stop_anomaly: bool
    other_anomalies: List[str] = Field(default_factory=list)

class Explanation(BaseModel):
    shap_values: Dict[str, float]
    top_contributing_features: List[str]

class CandidateVessel(BaseModel):
    rank: int
    mmsi: str
    vessel_name: Optional[str] = None
    flag: Optional[str] = None
    vessel_type: Optional[str] = None
    attribution_probability: float
    evidence: EvidenceScore
    behaviour: BehaviourFlags
    trajectory: Dict[str, Any]
    explanation: Explanation

class AttributionSummary(BaseModel):
    top_candidate_mmsi: str
    top_probability: float
    confidence: float

class Agent3Output(BaseModel):
    event_id: str
    candidate_vessels: List[CandidateVessel]
    attribution_summary: AttributionSummary
    quality_control: QualityControl
