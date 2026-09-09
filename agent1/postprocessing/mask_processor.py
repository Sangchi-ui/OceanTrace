from typing import Any

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, opening


class MaskProcessor:
    """
    Post-processing module for segmentation probability maps.
    Applies thresholding, morphological filtering, connected-component analysis,
    and removes noise artifacts.
    """

    def __init__(
        self,
        probability_threshold: float = 0.5,
        morphology_kernel_size: int = 3,
        min_region_pixels: int = 50,
        remove_small_objects: bool = True
    ):
        self.probability_threshold = probability_threshold
        self.morphology_kernel_size = morphology_kernel_size
        self.min_region_pixels = min_region_pixels
        self.remove_small_objects = remove_small_objects

    def threshold_probability_map(self, prob_map: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """
        Thresholds 2D probability map [0.0, 1.0] into a binary mask (uint8: 0 or 1).
        """
        th = threshold if threshold is not None else self.probability_threshold
        binary_mask = (prob_map >= th).astype(np.uint8)
        return binary_mask

    def apply_morphology(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Applies morphological opening to eliminate isolated noise pixels
        and morphological closing to bridge gaps within candidate slicks.
        """
        selem = disk(self.morphology_kernel_size)
        opened = opening(binary_mask, selem)
        closed = closing(opened, selem)
        return closed.astype(np.uint8)

    def extract_connected_components(
        self,
        cleaned_mask: np.ndarray,
        prob_map: np.ndarray
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """
        Performs connected-component analysis.
        Filters out regions with pixel count < min_region_pixels.
        Computes region confidence (mean probability score).

        Returns:
            final_mask: binary mask containing only validated components.
            regions: list of dicts with region properties.
        """
        labeled_mask, _num_features = label(cleaned_mask, return_num=True)
        final_mask = np.zeros_like(cleaned_mask, dtype=np.uint8)
        regions_data = []

        props = regionprops(labeled_mask, intensity_image=prob_map)

        region_idx = 1
        for prop in props:
            area_px = prop.area
            if self.remove_small_objects and area_px < self.min_region_pixels:
                continue

            # Add valid region pixels to final mask
            coords = prop.coords
            final_mask[coords[:, 0], coords[:, 1]] = 1

            min_row, min_col, max_row, max_col = prop.bbox
            width_px = float(max_col - min_col)
            height_px = float(max_row - min_row)
            aspect_ratio = width_px / max(height_px, 1e-5)

            # Calculate confidence as mean probability inside region
            region_probs = prob_map[coords[:, 0], coords[:, 1]]
            mean_confidence = float(np.mean(region_probs)) if len(
                region_probs) > 0 else float(self.probability_threshold)

            # Centroid in pixel coordinates (Y, X)
            cy_px, cx_px = prop.centroid

            region_info = {
                "spill_id": f"SPILL_{region_idx:03d}",
                "area_pixels": int(area_px),
                "confidence": round(mean_confidence, 4),
                "centroid_pixel": {"x": float(cx_px), "y": float(cy_px)},
                "bbox_pixel": [int(min_col), int(min_row), int(max_col), int(max_row)],
                "width_pixels": width_px,
                "height_pixels": height_px,
                "aspect_ratio": round(aspect_ratio, 3),
                "region_mask_coords": coords
            }
            regions_data.append(region_info)
            region_idx += 1

        return final_mask, regions_data

    def process(
        self,
        prob_map: np.ndarray,
        threshold: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """
        Full Post-Processing Pipeline:
        Probability map -> Threshold -> Morphology -> Connected Component Extraction.

        Returns:
            raw_binary_mask: initial thresholded mask
            cleaned_mask: morphology + small object filtered mask
            regions_data: list of region properties
        """
        raw_mask = self.threshold_probability_map(
            prob_map, threshold=threshold)
        morph_mask = self.apply_morphology(raw_mask)
        final_mask, regions = self.extract_connected_components(
            morph_mask, prob_map)

        return raw_mask, final_mask, regions
