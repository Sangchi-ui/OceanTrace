import json
import os
import sys
from pydantic import ValidationError

from contracts.system_contracts import (
    Agent1Output, Agent2Input, Agent2Output, Agent3Input, Agent3Output, AISData
)
from agent1.predict import run_inference
from agent2.pipeline import Agent2Pipeline
from agent3.pipeline import Agent3Pipeline
from agent2.adapters.agent1_adapter import Agent1Adapter

def validate_json_with_schema(json_data, schema_model):
    try:
        model = schema_model.model_validate(json_data)
        return True, model, []
    except ValidationError as e:
        errors = []
        for err in e.errors():
            loc = ".".join(str(l) for l in err["loc"])
            expected_type = err.get("type")
            msg = err.get("msg")
            val = json_data
            for l in err["loc"]:
                if isinstance(val, dict) and l in val:
                    val = val[l]
                else:
                    val = None
                    break
            errors.append({
                "field": loc,
                "expected": expected_type,
                "current_val": val,
                "msg": msg
            })
        return False, None, errors

def print_audit_result(edge_name, passed, errors):
    print(f"\n{edge_name}: {'PASS' if passed else 'FAIL'}")
    if not passed:
        for err in errors:
            print(f"  - Missing/Invalid Field: {err['field']}")
            print(f"    Expected Type: {err['expected']}")
            print(f"    Current Value: {err['current_val']}")
            print(f"    Error: {err['msg']}")

def main():
    print("="*50)
    print("MULTI-AGENT CONTRACT AUDIT")
    print("="*50)

    # Use existing sample image if it exists, otherwise use a placeholder
    sample_img = "data/sample/sample_sentinel1.tif"
    output_dir = "outputs/audit_run"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- Running Agent 1 (Detection) ---")
    try:
        # We run inference, which now saves Agent1Output.json
        run_inference(sample_img, output_dir=output_dir)
        with open(os.path.join(output_dir, "Agent1Output.json"), "r") as f:
            a1_out_json = json.load(f)
            
        pass_a1, a1_model, errs_a1 = validate_json_with_schema(a1_out_json, Agent1Output)
        print_audit_result("AGENT 1 -> AGENT 2", pass_a1, errs_a1)
    except Exception as e:
        print(f"AGENT 1 -> AGENT 2: FAIL\n  - Exception during Agent 1 execution: {e}")
        pass_a1 = False
        a1_model = None
        a1_out_json = None

    if not pass_a1:
        print("\nStopping audit due to Agent 1 failure.")
        sys.exit(1)
        
    print("\n--- Running Agent 2 (Drift Modelling) ---")
    try:
        # Run Agent 2 using Agent1Output
        a2_pipeline = Agent2Pipeline()
        # Parse it through the updated Agent1Adapter
        obs = Agent1Adapter.parse(a1_out_json)
        
        a2_result = a2_pipeline.run(obs, mode="both")
        a2_pipeline.export_artifacts(a2_result, output_dir=output_dir)
        
        with open(os.path.join(output_dir, "Agent2Output.json"), "r") as f:
            a2_out_json = json.load(f)
            
        pass_a2, a2_model, errs_a2 = validate_json_with_schema(a2_out_json, Agent2Output)
        print_audit_result("AGENT 2 -> AGENT 3", pass_a2, errs_a2)
    except Exception as e:
        print(f"AGENT 2 -> AGENT 3: FAIL\n  - Exception during Agent 2 execution: {e}")
        pass_a2 = False
        a2_model = None

    if not pass_a2:
        print("\nStopping audit due to Agent 2 failure.")
        sys.exit(1)

    print("\n--- Running Agent 3 (Vessel Attribution) ---")
    try:
        # Mock AIS Data
        mock_ais = {
            "vessels": [
                {
                    "mmsi": "123456789",
                    "timestamp": "2026-09-10T12:00:00Z",
                    "latitude": a2_model.origin_estimation.origin_centroid.latitude + 0.01,
                    "longitude": a2_model.origin_estimation.origin_centroid.longitude + 0.01,
                    "speed": 12.5,
                    "heading": 45.0,
                    "vessel_identity": "TANKER_A",
                    "vessel_type": "Tanker"
                },
                {
                    "mmsi": "987654321",
                    "timestamp": "2026-09-10T12:00:00Z",
                    "latitude": a2_model.origin_estimation.origin_centroid.latitude + 0.5,
                    "longitude": a2_model.origin_estimation.origin_centroid.longitude - 0.2,
                    "speed": 18.0,
                    "heading": 120.0,
                    "vessel_identity": "CARGO_B",
                    "vessel_type": "Cargo"
                }
            ]
        }
        
        a3_in_json = {
            "spill_event": a1_out_json,
            "drift_result": a2_out_json,
            "ais_data": mock_ais
        }
        
        pass_a3in, a3in_model, errs_a3in = validate_json_with_schema(a3_in_json, Agent3Input)
        if not pass_a3in:
            print("Failed to construct valid Agent3Input:")
            for err in errs_a3in:
                print(f"  {err['field']}: {err['msg']}")
            sys.exit(1)
            
        a3_pipeline = Agent3Pipeline()
        a3_model = a3_pipeline.run(a3in_model)
        
        a3_out_json = json.loads(a3_model.model_dump_json())
        with open(os.path.join(output_dir, "Agent3Output.json"), "w") as f:
            json.dump(a3_out_json, f, indent=2)
            
        pass_a3, a3_model_validated, errs_a3 = validate_json_with_schema(a3_out_json, Agent3Output)
        print_audit_result("AGENT 3 -> UI", pass_a3, errs_a3)
        
    except Exception as e:
        print(f"AGENT 3 -> UI: FAIL\n  - Exception during Agent 3 execution: {e}")
        pass_a3 = False
        
    print("\n" + "="*50)
    if pass_a1 and pass_a2 and pass_a3:
        print("ALL CONTRACT AUDITS PASSED SUCCESSFULLY.")
    else:
        print("ONE OR MORE CONTRACT AUDITS FAILED.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
