"""
Generates synthetic & calibrated historical oil spill drift trajectory datasets
for training the Agent 2 XGBoost Physics Residual and Uncertainty Model.
Simulates real-world Lagrangian buoy / satellite spill tracking with Stokes drift,
Coriolis deflection, wind shear, and sub-grid hydrodynamic turbulence.
"""

import argparse
import os
from typing import Optional
import numpy as np
import pandas as pd


def generate_drift_dataset(
    n_samples: int = 2500,
    output_path: str = "data/drift_trajectories_sample.csv",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic paired physics-versus-observed drift trajectory dataset.
    Features:
      - u_current, v_current: Surface ocean current velocity (m/s)
      - u_wind, v_wind: 10m atmospheric wind velocity (m/s)
      - windage: Oil leeway coefficient [0.025 - 0.045]
      - sst: Sea Surface Temperature (deg C)
      - duration_hours: Drift tracking period [1 to 72 hours]
      - observed_area_km2: Observed slick area in km^2
      - aspect_ratio: Slick elongation ratio (length / width)
      - oil_density: Crude oil density (kg/m^3)
      - oil_viscosity: Oil kinematic viscosity (cSt)
    Targets:
      - dx_residual_meters: Observed X displacement - Physics model simulated X displacement
      - dy_residual_meters: Observed Y displacement - Physics model simulated Y displacement
      - uncertainty_scale: Ratio of observed spread to standard Gaussian diffusion
    """
    rng = np.random.RandomState(seed)

    # 1. Metocean environmental features
    u_curr = rng.uniform(-0.9, 0.9, n_samples)
    v_curr = rng.uniform(-0.9, 0.9, n_samples)
    u_wind = rng.uniform(-18.0, 18.0, n_samples)
    v_wind = rng.uniform(-18.0, 18.0, n_samples)
    wind_speed = np.sqrt(u_wind**2 + v_wind**2)

    # Oil parameters
    windage = rng.uniform(0.025, 0.045, n_samples)
    sst = rng.uniform(14.0, 32.0, n_samples)
    duration_hours = rng.uniform(2.0, 72.0, n_samples)
    dur_sec = duration_hours * 3600.0

    observed_area_km2 = rng.uniform(0.5, 60.0, n_samples)
    aspect_ratio = rng.uniform(1.0, 4.5, n_samples)
    oil_density = rng.uniform(820.0, 960.0, n_samples)
    oil_viscosity = rng.uniform(5.0, 180.0, n_samples)

    # 2. Physics-based drift displacement
    # Standard Lagrangian advection: x_phys = (u_curr + windage * u_wind) * dt
    # But real ocean drift contains:
    #   a) Wave-induced Stokes drift (~1.0% to 1.6% of wind speed aligned with wind)
    #   b) Coriolis deflection of surface leeway (~10-15 degrees to the right in NH)
    #   c) Thermal stratification / SST viscosity dampening
    stokes_factor = 0.012 * (1.0 + 0.15 * np.tanh(wind_speed / 10.0))
    stokes_dx = stokes_factor * u_wind * dur_sec
    stokes_dy = stokes_factor * v_wind * dur_sec

    # Coriolis deflection on windage (deflects velocity vector slightly)
    coriolis_deflection_deg = 12.0  # degrees
    theta_rad = np.radians(coriolis_deflection_deg)
    u_wind_deflected = u_wind * np.cos(theta_rad) + v_wind * np.sin(theta_rad)
    v_wind_deflected = -u_wind * np.sin(theta_rad) + v_wind * np.cos(theta_rad)
    coriolis_dx = (windage * (u_wind_deflected - u_wind)) * dur_sec
    coriolis_dy = (windage * (v_wind_deflected - v_wind)) * dur_sec

    # Sub-grid turbulent noise
    noise_dx = rng.normal(0.0, 150.0 * np.sqrt(duration_hours), n_samples)
    noise_dy = rng.normal(0.0, 150.0 * np.sqrt(duration_hours), n_samples)

    # Viscosity and SST effect on weathering spreading
    temp_visc_factor = np.clip(1.0 + 0.02 * (25.0 - sst) + 0.001 * (oil_viscosity - 30.0), 0.7, 1.6)

    dx_residual = (stokes_dx + coriolis_dx) * temp_visc_factor + noise_dx
    dy_residual = (stokes_dy + coriolis_dy) * temp_visc_factor + noise_dy

    # Uncertainty scale factor
    # Higher wind shear + longer duration = broader ensemble spread
    uncertainty_scale = 1.0 + 0.04 * wind_speed * (duration_hours / 24.0) + 0.05 * rng.randn(n_samples)
    uncertainty_scale = np.clip(uncertainty_scale, 0.5, 3.5)

    df = pd.DataFrame({
        "u_current": np.round(u_curr, 4),
        "v_current": np.round(v_curr, 4),
        "u_wind": np.round(u_wind, 3),
        "v_wind": np.round(v_wind, 3),
        "windage": np.round(windage, 4),
        "sst": np.round(sst, 2),
        "duration_hours": np.round(duration_hours, 2),
        "observed_area_km2": np.round(observed_area_km2, 2),
        "aspect_ratio": np.round(aspect_ratio, 2),
        "oil_density": np.round(oil_density, 1),
        "oil_viscosity": np.round(oil_viscosity, 1),
        "dx_residual_meters": np.round(dx_residual, 2),
        "dy_residual_meters": np.round(dy_residual, 2),
        "uncertainty_scale": np.round(uncertainty_scale, 3)
    })

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Generated {len(df)} drift samples saved to '{output_path}'")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate drift trajectory dataset for Agent 2 XGBoost")
    parser.add_argument("--samples", "-n", type=int, default=2500, help="Number of trajectory records")
    parser.add_argument("--output", "-o", type=str, default="data/drift_trajectories_sample.csv", help="Output CSV path")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_drift_dataset(n_samples=args.samples, output_path=args.output, seed=args.seed)
