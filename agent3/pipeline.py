from typing import Dict, Any
from contracts.system_contracts import (
    Agent3Input, Agent3Output, CandidateVessel, EvidenceScore,
    BehaviourFlags, Explanation, AttributionSummary, QualityControl
)

class Agent3Pipeline:
    def __init__(self):
        pass
        
    def run(self, input_data: Agent3Input) -> Agent3Output:
        """
        Mock implementation of Agent 3 that ranks vessels based on proximity to origin.
        """
        vessels = input_data.ais_data.vessels
        
        origin_lat = input_data.drift_result.origin_estimation.origin_centroid.latitude
        origin_lon = input_data.drift_result.origin_estimation.origin_centroid.longitude
        
        candidates = []
        for v in vessels:
            dist = ((v.latitude - origin_lat)**2 + (v.longitude - origin_lon)**2)**0.5 * 111.0
            
            candidates.append({
                "vessel": v,
                "dist": dist
            })
            
        candidates.sort(key=lambda x: x["dist"])
        
        ranked_vessels = []
        for idx, c in enumerate(candidates):
            v = c["vessel"]
            prob = max(0.01, 0.9 - (idx * 0.1))
            
            ranked_vessels.append(
                CandidateVessel(
                    rank=idx + 1,
                    mmsi=v.mmsi,
                    vessel_name=v.vessel_identity,
                    flag=v.flag,
                    vessel_type=v.vessel_type,
                    attribution_probability=prob,
                    evidence=EvidenceScore(
                        minimum_distance_km=c["dist"],
                        time_difference_hours=0.5,
                        trajectory_match_score=0.85,
                        proximity_score=0.9,
                        behaviour_score=0.7
                    ),
                    behaviour=BehaviourFlags(
                        speed_anomaly=False,
                        heading_anomaly=False,
                        stop_anomaly=True
                    ),
                    trajectory={},
                    explanation=Explanation(
                        shap_values={"proximity": 0.4, "speed": 0.2},
                        top_contributing_features=["proximity_score", "stop_anomaly"]
                    )
                )
            )
            
        summary = AttributionSummary(
            top_candidate_mmsi=ranked_vessels[0].mmsi if ranked_vessels else "UNKNOWN",
            top_probability=ranked_vessels[0].attribution_probability if ranked_vessels else 0.0,
            confidence=0.85
        )
        
        return Agent3Output(
            event_id=input_data.spill_event.event_id,
            candidate_vessels=ranked_vessels,
            attribution_summary=summary,
            quality_control=QualityControl(
                status="VALID",
                warnings=[]
            )
        )
