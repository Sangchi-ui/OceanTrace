from typing import Any


class SpillValidator:
    """
    Automated Quality Control Validation Layer for Oil Spill Detection Results.
    Checks for unrealistic region sizes, low model confidence, missing georeferencing,
    excessive fragmentation, or empty segmentations.
    """

    def __init__(
        self,
        min_area_km2: float = 0.001,
        max_area_km2: float = 5000.0,
        min_confidence: float = 0.3,
        max_fragmentation_count: int = 20
    ):
        self.min_area_km2 = min_area_km2
        self.max_area_km2 = max_area_km2
        self.min_confidence = min_confidence
        self.max_fragmentation_count = max_fragmentation_count

    def validate_region(self, region_data: dict[str, Any], meta: dict[str, Any]) -> tuple[str, list[str]]:
        """
        Validates a single detected spill region record.
        """
        warnings = []
        status = "VALID"

        confidence = region_data.get("detection", {}).get("confidence", 0.0)
        area_km2 = region_data.get("measurements", {}).get("area_km2", None)
        region_data.get("measurements", {}).get("area_pixels", 0)
        is_geographic = region_data.get(
            "geometry", {}).get("is_geographic", False)

        if confidence < self.min_confidence:
            warnings.append(
                f"Low model confidence score ({confidence:.3f} < {self.min_confidence:.3f}).")
            status = "WARNING"

        if not is_geographic:
            warnings.append(
                "Source image lacks CRS/georeferencing metadata; measurements are in pixel space.")
            status = "WARNING" if status == "VALID" else status
        else:
            if area_km2 is not None:
                if area_km2 < self.min_area_km2:
                    warnings.append(
                        f"Detected area ({area_km2:.4f} km²) is below minimum threshold ({self.min_area_km2} km²).")
                    status = "WARNING"
                elif area_km2 > self.max_area_km2:
                    warnings.append(
                        f"Unrealistically large spill area ({area_km2:.2f} km² > {self.max_area_km2} km²). Potential look-alike or low wind speed area.")
                    status = "INVALID"

        return status, warnings

    def validate_event(
        self,
        num_regions: int,
        region_records: list[dict[str, Any]],
        meta: dict[str, Any]
    ) -> tuple[str, list[str]]:
        """
        Validates the overall image SpillEvent result.
        """
        event_warnings = []
        overall_status = "VALID"

        if num_regions == 0:
            event_warnings.append(
                "No oil spill candidates detected in input image.")
            return "VALID", event_warnings  # Clean image is a valid result

        if num_regions > self.max_fragmentation_count:
            event_warnings.append(
                f"Excessive segmentation fragmentation ({num_regions} separate regions). High risk of False Positives / look-alikes.")
            overall_status = "WARNING"

        if not meta.get("is_georeferenced", False):
            event_warnings.append(
                "Input raster is not georeferenced (missing CRS and affine transform).")
            if overall_status == "VALID":
                overall_status = "WARNING"

        # Aggregate region warnings
        invalid_count = 0
        warning_count = 0
        for rec in region_records:
            st = rec.get("validation", {}).get("status", "VALID")
            if st == "INVALID":
                invalid_count += 1
            elif st == "WARNING":
                warning_count += 1

        if invalid_count > 0:
            overall_status = "INVALID"
            event_warnings.append(
                f"{invalid_count} region(s) failed validation checks (INVALID).")
        elif warning_count > 0 and overall_status == "VALID":
            overall_status = "WARNING"

        return overall_status, event_warnings
