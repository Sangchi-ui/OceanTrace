"""
Command-Line Interface (CLI) for Agent 2: Hindcasting & Forecasting.
Usage:
    python -m agent2.cli --input outputs/spill_event.json --output outputs/agent2 --mode both
"""

import argparse
import os
import sys

# Ensure root directory in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent2.pipeline import Agent2Pipeline


def main():
    parser = argparse.ArgumentParser(
        description="OceanTrace - Agent 2: Physics-Based Hindcasting & Forecasting Pipeline"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to input Agent 1 detection file (spill_event.json or spill.geojson)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="outputs/agent2",
        help="Directory to save output predictions, diagnostics, and GeoJSON artifacts"
    )
    parser.add_argument(
        "--mode", "-m", type=str, default="both", choices=["both", "hindcast", "forecast"],
        help="Analysis mode: 'both' (default), 'hindcast', or 'forecast'"
    )
    parser.add_argument(
        "--config", "-c", type=str, default="configs/agent2.yaml",
        help="Path to Agent 2 configuration YAML file"
    )
    parser.add_argument(
        "--region-index", "-r", type=int, default=0,
        help="Index of spill region to analyze if input contains multiple slicks (default: 0)"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("OceanTrace Agent 2: Hindcasting & Forecasting Pipeline")
    print(f"Input:        {args.input}")
    print(f"Output dir:   {args.output}")
    print(f"Mode:         {args.mode}")
    print(f"Config:       {args.config}")
    print("="*60 + "\n")

    pipeline = Agent2Pipeline(config_path=args.config if os.path.exists(args.config) else None)
    result = pipeline.run_from_file(
        input_path=args.input,
        output_dir=args.output,
        mode=args.mode,
        region_index=args.region_index
    )

    print("\n" + "="*60)
    print("Agent 2 Analysis Completed Successfully!")
    print(f"Analysis ID:          {result.analysis_id}")
    print(f"Execution Timestamp:  {result.execution_timestamp.isoformat()}")

    if result.hindcast.enabled and result.hindcast.probable_origin:
        po = result.hindcast.probable_origin
        print("\n--- HINDCAST RESULTS ---")
        print(f"Probable Origin Area: {po.area_km2} km^2")
        print(f"Origin Centroid:      ({po.centroid[0]:.4f}, {po.centroid[1]:.4f})")
        print(f"Time Window Start:    {po.time_window_start.isoformat()}")
        print(f"Time Window End:      {po.time_window_end.isoformat()}")
        print(f"Origin Confidence:    {po.confidence:.2%}")
        print(f"Uncertainty Radius:   {po.uncertainty_radius_km:.2f} km")

    if result.forecast.enabled and result.forecast.horizons:
        print("\n--- FORECAST RESULTS ---")
        for hor in result.forecast.horizons:
            print(f"+{hor.horizon_hours:02d}h Forecast: Centroid ({hor.centroid[0]:.4f}, {hor.centroid[1]:.4f}), "
                  f"Spread Area: {hor.spread_area_km2:.2f} km^2, Drift: {hor.drift_distance_km:.2f} km")

    print(f"\nArtifacts exported to: '{args.output}/'")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
