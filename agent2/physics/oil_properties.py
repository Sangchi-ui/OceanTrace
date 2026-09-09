"""
Oil weathering and physical property classifications for Agent 2.
Avoids false assumptions about exact chemical composition from SAR imagery
by providing standardized scenarios (Light, Medium, Heavy Crude) with documented parameters.
"""

from enum import Enum
from typing import Dict
from pydantic import BaseModel, Field


class OilCategory(str, Enum):
    LIGHT_CRUDE = "light_crude"
    MEDIUM_CRUDE = "medium_crude"
    HEAVY_CRUDE = "heavy_crude"
    DIESEL = "diesel"


class OilProperties(BaseModel):
    """Physical properties of spilled oil for transport and spreading simulation."""
    category: OilCategory = Field(default=OilCategory.MEDIUM_CRUDE)
    density_kg_m3: float = Field(default=865.0, description="Oil density at 15C in kg/m^3")
    viscosity_cst: float = Field(default=25.0, description="Kinematic viscosity in centistokes at 20C")
    pour_point_c: float = Field(default=-10.0, description="Pour point in degrees Celsius")
    default_windage: float = Field(default=0.030, ge=0.01, le=0.06, description="Default windage coefficient")
    min_windage: float = Field(default=0.025, description="Lower windage sensitivity limit")
    max_windage: float = Field(default=0.044, description="Upper windage sensitivity limit")
    horizontal_diffusivity_m2_s: float = Field(default=2.5, description="Horizontal eddy diffusivity coefficient D_h")
    evaporation_half_life_hours: float = Field(default=36.0, description="Approximate evaporation half-life in hours")


# Standardized scenario definitions
OIL_SCENARIOS: Dict[OilCategory, OilProperties] = {
    OilCategory.LIGHT_CRUDE: OilProperties(
        category=OilCategory.LIGHT_CRUDE,
        density_kg_m3=825.0,
        viscosity_cst=8.0,
        pour_point_c=-20.0,
        default_windage=0.032,
        min_windage=0.028,
        max_windage=0.044,
        horizontal_diffusivity_m2_s=3.0,
        evaporation_half_life_hours=18.0
    ),
    OilCategory.MEDIUM_CRUDE: OilProperties(
        category=OilCategory.MEDIUM_CRUDE,
        density_kg_m3=865.0,
        viscosity_cst=25.0,
        pour_point_c=-10.0,
        default_windage=0.030,
        min_windage=0.025,
        max_windage=0.040,
        horizontal_diffusivity_m2_s=2.5,
        evaporation_half_life_hours=36.0
    ),
    OilCategory.HEAVY_CRUDE: OilProperties(
        category=OilCategory.HEAVY_CRUDE,
        density_kg_m3=920.0,
        viscosity_cst=150.0,
        pour_point_c=5.0,
        default_windage=0.028,
        min_windage=0.022,
        max_windage=0.035,
        horizontal_diffusivity_m2_s=1.8,
        evaporation_half_life_hours=96.0
    ),
    OilCategory.DIESEL: OilProperties(
        category=OilCategory.DIESEL,
        density_kg_m3=835.0,
        viscosity_cst=4.0,
        pour_point_c=-25.0,
        default_windage=0.033,
        min_windage=0.029,
        max_windage=0.045,
        horizontal_diffusivity_m2_s=3.5,
        evaporation_half_life_hours=12.0
    )
}


def get_oil_properties(category_or_name: str | OilCategory = OilCategory.MEDIUM_CRUDE) -> OilProperties:
    """Retrieves standard OilProperties by name or category enum."""
    if isinstance(category_or_name, str):
        try:
            cat = OilCategory(category_or_name.lower())
        except ValueError:
            cat = OilCategory.MEDIUM_CRUDE
    else:
        cat = category_or_name
    return OIL_SCENARIOS.get(cat, OIL_SCENARIOS[OilCategory.MEDIUM_CRUDE])
