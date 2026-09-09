import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from app.preprocessing.sar_preprocessor import SARPreprocessor


class SARDataset(Dataset):
    """
    PyTorch Dataset for Sentinel-1 SAR Oil Spill Semantic Segmentation.

    Expected SAR Input Representation:
    - Single-channel C-band Synthetic Aperture Radar (SAR) backscatter intensity (e.g., VV or VH polarization).
    - Image values can be linear intensity or raw Digital Numbers (DN), which are log-scaled to decibels (dB)
      and speckle-filtered during preprocessing.
    - Ground truth masks are single-channel binary masks (0 = ocean background, 1 / 255 = potential oil slick).
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str | None = None,
        target_size: tuple[int, int] = (256, 256),
        preprocessor: SARPreprocessor | None = None,
        is_train: bool = True,
        positive_crop_probability: float = 0.0,
        mask_format: str = "mados",
        speckle_noise_probability: float = 0.25,
        intensity_jitter_probability: float = 0.25,
        cache_dir: str | None = None,
        in_channels: int = 2
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.preprocessor = preprocessor or SARPreprocessor(
            target_size=target_size)
        self.is_train = is_train
        self.positive_crop_probability = positive_crop_probability
        self.mask_format = mask_format
        self.speckle_noise_probability = speckle_noise_probability
        self.intensity_jitter_probability = intensity_jitter_probability
        self.cache_dir = cache_dir
        self.in_channels = in_channels
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.image_paths = sorted(
            glob.glob(os.path.join(image_dir, "*.tif")) +
            glob.glob(os.path.join(image_dir, "*.tiff")) +
            glob.glob(os.path.join(image_dir, "*.png")) +
            glob.glob(os.path.join(image_dir, "*.jpg"))
        )

        self.mask_paths = []
        if mask_dir and os.path.exists(mask_dir):
            for img_path in self.image_paths:
                base_name = os.path.splitext(os.path.basename(img_path))[0]

                candidates = [
                    os.path.join(mask_dir, f"{base_name}.png"),
                    os.path.join(mask_dir, f"{base_name}_mask.png"),
                    os.path.join(mask_dir, f"{base_name}.tif")
                ]

                if "_rgb_" in base_name:
                    cl_name = base_name.replace("_rgb_", "_cl_")
                    candidates.extend([
                        os.path.join(mask_dir, f"{cl_name}.tif"),
                        os.path.join(mask_dir, f"{cl_name}.png")
                    ])

                mask_candidate = None
                for c in candidates:
                    if os.path.exists(c):
                        mask_candidate = c
                        break

                if mask_candidate:
                    self.mask_paths.append(mask_candidate)
                else:
                    self.mask_paths.append(None)
                    print(f"WARNING: No mask found for {img_path}")
        else:
            self.mask_paths = [None] * len(self.image_paths)

        # Precompute positive samples for balanced batch sampling (train only)
        self.is_positive = []
        if self.is_train and mask_dir and os.path.exists(mask_dir):
            for image_index, mp in enumerate(self.mask_paths):
                if mp and os.path.exists(mp):
                    try:
                        image_name = os.path.basename(self.image_paths[image_index])
                        if self.mask_format != "mados" and image_name.startswith("clean_"):
                            self.is_positive.append(False)
                            continue
                        cache_path = None
                        if self.cache_dir:
                            cache_name = os.path.splitext(image_name)[0] + ".pt"
                            cache_path = os.path.join(self.cache_dir, cache_name)
                        if cache_path and os.path.exists(cache_path):
                            cached_mask = torch.load(cache_path, map_location="cpu")["mask"]
                            is_pos = bool(cached_mask.any())
                        else:
                            with Image.open(mp) as mk_im:
                                preview = mk_im.resize((64, 64), resample=Image.NEAREST)
                                arr = np.array(preview)
                                is_pos = bool((arr == 6).any() if self.mask_format == "mados"
                                              else (arr > 0).any())
                        self.is_positive.append(is_pos)
                    except Exception:
                        self.is_positive.append(False)
                else:
                    self.is_positive.append(False)

    def get_sample_weights(self) -> torch.Tensor:
        """
        Calculate sample weights so positive (oil) and negative (clean) samples
        have equal 50/50 probability of being sampled in training batches.
        """
        pos_count = sum(self.is_positive)
        neg_count = len(self.is_positive) - pos_count
        pos_weight = 0.5 / max(pos_count, 1)
        neg_weight = 0.5 / max(neg_count, 1)

        weights = [
            pos_weight if is_pos else neg_weight for is_pos in self.is_positive]
        return torch.tensor(weights, dtype=torch.double)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        cache_path = None
        if self.cache_dir:
            cache_name = os.path.splitext(os.path.basename(img_path))[0] + ".pt"
            cache_path = os.path.join(self.cache_dir, cache_name)

        if cache_path and os.path.exists(cache_path):
            cached = torch.load(cache_path, map_location="cpu")
            img_tensor = cached["image"]
            mask_tensor = cached["mask"]
        else:
            _, tensor_img, _meta = self.preprocessor.preprocess_sar(img_path)
            img_tensor = tensor_img.squeeze(0)

            if mask_path and os.path.exists(mask_path):
                mask_pil = Image.open(mask_path)
                mask_resized = mask_pil.resize(
                    (self.target_size[1], self.target_size[0]), resample=Image.NEAREST)
                mask_np = np.array(mask_resized)
                mask_arr = (mask_np == 6).astype(np.float32) if self.mask_format == "mados" else (mask_np > 0).astype(np.float32)
                mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)
            else:
                mask_tensor = torch.zeros(
                    (1, self.target_size[0], self.target_size[1]), dtype=torch.float32)

            if cache_path:
                torch.save({"image": img_tensor, "mask": mask_tensor}, cache_path)

        # Enforce consistent number of input channels
        if img_tensor.shape[0] == 1 and self.in_channels > 1:
            img_tensor = img_tensor.repeat(self.in_channels, 1, 1)
        elif img_tensor.shape[0] > self.in_channels:
            img_tensor = img_tensor[:self.in_channels]

        # Zoom positive regions so the small oil class contributes stronger
        # spatial gradients during training.
        if self.is_train and mask_tensor.any() and torch.rand(1).item() < self.positive_crop_probability:
            height, width = self.target_size
            ys, xs = torch.where(mask_tensor[0] > 0.5)
            crop_height = max(int(height * 0.7), 32)
            crop_width = max(int(width * 0.7), 32)
            center_y = int(ys.float().mean().item())
            center_x = int(xs.float().mean().item())
            top = min(max(center_y - crop_height // 2, 0), height - crop_height)
            left = min(max(center_x - crop_width // 2, 0), width - crop_width)
            img_tensor = img_tensor[:, top:top + crop_height, left:left + crop_width]
            mask_tensor = mask_tensor[:, top:top + crop_height, left:left + crop_width]
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0), size=self.target_size,
                mode="bilinear", align_corners=False).squeeze(0)
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0), size=self.target_size,
                mode="nearest").squeeze(0)

        # Spatial augmentations during training
        if self.is_train:
            if torch.rand(1).item() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[-1])
                mask_tensor = torch.flip(mask_tensor, dims=[-1])
            if torch.rand(1).item() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[-2])
                mask_tensor = torch.flip(mask_tensor, dims=[-2])
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
                img_tensor = torch.rot90(img_tensor, k, dims=[-2, -1])
                mask_tensor = torch.rot90(mask_tensor, k, dims=[-2, -1])

            if torch.rand(1).item() < self.speckle_noise_probability:
                noise = torch.randn_like(img_tensor) * 0.04
                img_tensor = torch.clamp(img_tensor + noise, 0.0, 1.0)

            if torch.rand(1).item() < self.intensity_jitter_probability:
                contrast = 0.9 + 0.2 * torch.rand(1).item()
                brightness = (torch.rand(1).item() - 0.5) * 0.1
                img_tensor = torch.clamp(img_tensor * contrast + brightness, 0.0, 1.0)

        return img_tensor, mask_tensor, img_path
