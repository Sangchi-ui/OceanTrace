"""
OceanTrace — Agent 1: Sentinel-1 SAR Oil Spill Detection Pipeline.
Deep learning segmentation, geometric characterization, and quality validation.
"""

from agent1.schemas.spill_event import (
    SpillEvent,
    SpillRegionRecord,
    GeometryMetadata,
    MeasurementsMetadata,
    DetectionMetadata,
    SourceMetadata,
    Centroid,
    ValidationMetadata,
)
__version__ = "1.0.0"


def __getattr__(name: str):
    if name == "run_inference":
        from agent1.predict import run_inference
        return run_inference
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "SpillEvent",
    "SpillRegionRecord",
    "GeometryMetadata",
    "MeasurementsMetadata",
    "DetectionMetadata",
    "SourceMetadata",
    "Centroid",
    "ValidationMetadata",
    "run_inference",
    "__version__",
]
