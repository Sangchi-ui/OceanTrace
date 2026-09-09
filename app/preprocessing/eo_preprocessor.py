import os
from typing import Any

import numpy as np
import torch
from PIL import Image


class EOPreprocessor:
    """
    Preprocessor for multi-spectral Earth Observation (EO) imagery (e.g. RGB).
    Handles reading PNG/TIFF, scaling to [0, 1], and creating PyTorch tensors.
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (256, 256),
        normalize: bool = True
    ):
        self.target_size = target_size
        self.normalize = normalize

    def preprocess_eo(self, file_path: str) -> tuple[np.ndarray, torch.Tensor, dict[str, Any]]:
        """
        Input file -> Read RGB raster -> Resize -> Normalize -> Tensor.

        Returns:
            arr_resized: np.ndarray of shape (H, W, C)
            tensor: torch.Tensor of shape (1, C, target_H, target_W)
            meta: geospatial metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image not found: {file_path}")

        pil_img = Image.open(file_path).convert("RGB")

        meta = {
            "file_path": file_path,
            "width": pil_img.width,
            "height": pil_img.height,
            "is_georeferenced": False
        }

        # Resize
        pil_resized = pil_img.resize(
            (self.target_size[1], self.target_size[0]), resample=Image.BILINEAR)
        arr_resized = np.array(pil_resized, dtype=np.float32)

        if self.normalize:
            arr_resized = arr_resized / 255.0

        # Create tensor: PyTorch expects [C, H, W]
        tensor = torch.from_numpy(arr_resized).permute(
            2, 0, 1).unsqueeze(0)  # [1, C, H, W]

        return arr_resized, tensor, meta

    def read_raster(self, file_path: str) -> tuple[np.ndarray, dict[str, Any]]:
        pil_img = Image.open(file_path).convert("RGB")
        return np.array(pil_img, dtype=np.float32) / 255.0, {}
