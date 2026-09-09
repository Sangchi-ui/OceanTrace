import os
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


class SARPreprocessor:
    """
    Preprocessor for Sentinel-1 Synthetic Aperture Radar (SAR) imagery.
    Handles reading GeoTIFF/TIFF/PNG/JPEG rasters, no-data replacement,
    decibel scaling, speckle reduction, min-max normalization, and PyTorch tensor creation.
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (256, 256),
        speckle_filter: str = "lee",
        speckle_kernel_size: int = 3,
        db_min: float = -30.0,
        db_max: float = 0.0,
        normalize: bool = True
    ):
        self.target_size = target_size
        self.speckle_filter = speckle_filter
        self.speckle_kernel_size = speckle_kernel_size
        self.db_min = db_min
        self.db_max = db_max
        self.normalize = normalize

    def read_raster(self, file_path: str, target_size: tuple[int, int] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reads a raster image from disk. Returns raw numpy array (H, W) and geospatial metadata dict.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Raster file not found: {file_path}")

        meta = {
            "file_path": file_path,
            "crs": None,
            "transform": None,
            "width": None,
            "height": None,
            "is_georeferenced": False
        }

        ext = os.path.splitext(file_path)[1].lower()

        if RASTERIO_AVAILABLE and ext in [".tif", ".tiff", ".gtiff"]:
            with rasterio.open(file_path) as src:
                if target_size is None:
                    img = src.read().astype(np.float32)
                else:
                    img = src.read(
                        out_shape=(src.count, target_size[0], target_size[1]),
                        resampling=rasterio.enums.Resampling.bilinear,
                    ).astype(np.float32)

                meta["crs"] = str(src.crs) if src.crs else None
                meta["transform"] = src.transform
                meta["width"] = src.width
                meta["height"] = src.height
                meta["is_georeferenced"] = (
                    src.crs is not None and src.transform is not None)
                if hasattr(src, 'nodata') and src.nodata is not None:
                    img[img == src.nodata] = np.nan
        else:
            # Fallback to PIL for PNG, JPG, or if rasterio is missing
            pil_img = Image.open(file_path).convert("L")
            img = np.array(pil_img, dtype=np.float32)[None, ...]
            meta["width"] = img.shape[1]
            meta["height"] = img.shape[0]
            meta["is_georeferenced"] = False

        return img, meta

    def handle_nodata(self, img: np.ndarray) -> np.ndarray:
        """
        Replaces NaNs, Infs, or negative values with background median/minimum.
        """
        mask_invalid = np.isnan(img) | np.isinf(img)
        if np.all(mask_invalid):
            return np.zeros_like(img, dtype=np.float32)

        valid_vals = img[~mask_invalid]
        min_val = np.min(valid_vals) if len(valid_vals) > 0 else 0.0
        img[mask_invalid] = min_val
        # SAR GeoTIFFs commonly store calibrated backscatter in dB, which is
        # negative. Preserve those values; only clamp linear intensity data.
        if np.min(valid_vals) >= 0.0:
            return np.maximum(img, 0.0)
        return img.astype(np.float32)

    def convert_to_db(self, img: np.ndarray) -> np.ndarray:
        """
        Converts linear intensity values to decibels (dB): 10 * log10(intensity + eps).
        If image max is small (< 100), assumes linear intensity. If already large or in dB, returns as-is.
        """
        if np.min(img) < 0.0:
            # Already calibrated dB values, such as Sentinel-1 backscatter.
            return img.astype(np.float32)
        if np.max(img) > 0 and np.max(img) <= 1.0:
            # Normalized intensity
            img = 10.0 * np.log10(img + 1e-6)
        elif np.max(img) > 1.0 and np.max(img) < 65535.0:
            # Raw DN / intensity
            img = 10.0 * np.log10(img + 1e-6)
        return img

    def reduce_speckle(self, img: np.ndarray) -> np.ndarray:
        """
        Applies speckle filtering. Supports Lee filter or Median filter.
        """
        if img.ndim == 3:
            return np.stack([
                self.reduce_speckle(band) for band in img
            ], axis=0)
        if self.speckle_filter == "lee":
            return self._lee_filter(img, size=self.speckle_kernel_size)
        elif self.speckle_filter == "median":
            return ndimage.median_filter(img, size=self.speckle_kernel_size)
        return img

    def _lee_filter(self, img: np.ndarray, size: int = 3) -> np.ndarray:
        """
        Enhanced Lee Speckle Filter implementation for SAR imagery.
        """
        img_mean = ndimage.uniform_filter(img, (size, size))
        img_sqr_mean = ndimage.uniform_filter(img**2, (size, size))
        img_var = img_sqr_mean - img_mean**2
        img_var = np.maximum(img_var, 0)

        overall_var = np.var(img)
        if overall_var == 0:
            return img

        weights = img_var / (img_var + overall_var + 1e-6)
        img_filtered = img_mean + weights * (img - img_mean)
        return img_filtered.astype(np.float32)

    def normalize_sar(self, img: np.ndarray) -> np.ndarray:
        """
        Normalizes SAR intensity to [0.0, 1.0] range using min-max scaling or percentile clipping.
        """
        if not self.normalize:
            return img

        norm_img = np.zeros_like(img, dtype=np.float32)
        for band_index in range(img.shape[0]):
            band = img[band_index]
            p2, p98 = np.percentile(band, (1, 99))
            if p98 > p2:
                band_clipped = np.clip(band, p2, p98)
                norm_img[band_index] = (band_clipped - p2) / (p98 - p2)
            else:
                min_v, max_v = np.min(band), np.max(band)
                if max_v > min_v:
                    norm_img[band_index] = (band - min_v) / (max_v - min_v)

        return norm_img.astype(np.float32)

    def preprocess_sar(self, file_path: str) -> tuple[np.ndarray, torch.Tensor, dict[str, Any]]:
        """
        Full SAR Preprocessing Pipeline:
        Input file -> Read raster -> Handle invalid pixels -> Convert dB -> Speckle Filter -> Normalize -> Tensor.

        Returns:
            processed_2d_np: np.ndarray of shape (H, W) normalized
            tensor: torch.Tensor of shape (1, 1, target_H, target_W)
            meta: geospatial metadata
        """
        img, meta = self.read_raster(file_path, target_size=self.target_size)
        img = self.handle_nodata(img)
        img_db = self.convert_to_db(img)
        img_filtered = self.reduce_speckle(img_db)
        img_norm = self.normalize_sar(img_filtered)

        # PyTorch Tensor [1, C, H, W]
        tensor = torch.from_numpy(img_norm).unsqueeze(0)

        return img_norm, tensor, meta
