"""
Forecast Ensemble Perturbation Generator for Agent 2.
Produces systematic environmental and physical parameter perturbations
to quantify forecasting uncertainty without claiming unvalidated calibration.
"""

from typing import Any, Dict, List, Optional
import numpy as np


class EnsembleMemberConfig:
    """Configuration for a single perturbed ensemble realization."""
    def __init__(
        self,
        member_id: int,
        description: str,
        current_scale: float = 1.0,
        current_angle_deg: float = 0.0,
        wind_scale: float = 1.0,
        wind_angle_deg: float = 0.0,
        windage: float = 0.030,
        diffusivity_factor: float = 1.0
    ):
        self.member_id = member_id
        self.description = description
        self.current_scale = current_scale
        self.current_angle_deg = current_angle_deg
        self.wind_scale = wind_scale
        self.wind_angle_deg = wind_angle_deg
        self.windage = windage
        self.diffusivity_factor = diffusivity_factor


class EnsembleGenerator:
    """
    Generates a structured ensemble of environmental and physical perturbations.
    """

    @classmethod
    def generate_members(
        cls,
        ensemble_size: int = 15,
        base_windage: float = 0.030,
        min_windage: float = 0.025,
        max_windage: float = 0.044,
        seed: Optional[int] = 42
    ) -> List[EnsembleMemberConfig]:
        """
        Creates ensemble members spanning windage sensitivity, current uncertainty,
        and atmospheric wind velocity variations.
        """
        rng = np.random.RandomState(seed)
        members: List[EnsembleMemberConfig] = []

        # Member 0 is always unperturbed baseline
        members.append(EnsembleMemberConfig(
            member_id=0,
            description="Baseline Unperturbed Control",
            current_scale=1.0,
            current_angle_deg=0.0,
            wind_scale=1.0,
            wind_angle_deg=0.0,
            windage=base_windage,
            diffusivity_factor=1.0
        ))

        # Distinct physical sensitivity benchmarks
        members.append(EnsembleMemberConfig(
            member_id=1,
            description="Low Windage Bound (2.5%)",
            windage=min_windage
        ))
        members.append(EnsembleMemberConfig(
            member_id=2,
            description="High Windage Bound (4.4%)",
            windage=max_windage
        ))
        members.append(EnsembleMemberConfig(
            member_id=3,
            description="Accelerated Current (+15%)",
            current_scale=1.15,
            windage=base_windage
        ))
        members.append(EnsembleMemberConfig(
            member_id=4,
            description="Decelerated Current (-15%)",
            current_scale=0.85,
            windage=base_windage
        ))

        # Stochastic perturbation members
        for idx in range(5, ensemble_size):
            # Normal variations around forcing fields
            curr_s = float(rng.normal(1.0, 0.08))
            curr_deg = float(rng.normal(0.0, 4.0))
            wind_s = float(rng.normal(1.0, 0.12))
            wind_deg = float(rng.normal(0.0, 8.0))
            w_val = float(rng.uniform(min_windage, max_windage))
            diff_f = float(rng.uniform(0.7, 1.4))

            members.append(EnsembleMemberConfig(
                member_id=idx,
                description=f"Stochastic Perturbation realization #{idx}",
                current_scale=max(curr_s, 0.5),
                current_angle_deg=curr_deg,
                wind_scale=max(wind_s, 0.5),
                wind_angle_deg=wind_deg,
                windage=w_val,
                diffusivity_factor=diff_f
            ))

        return members[:ensemble_size]
