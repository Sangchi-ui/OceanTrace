import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from app.models.unet import get_segmentation_model
from training.dataset import SARDataset


def checkpoint_features(path: str) -> list[int]:
    state = torch.load(path, map_location="cpu")
    first_width = state["downs.0.double_conv.0.weight"].shape[0]
    return [first_width * (2 ** index) for index in range(4)]


def has_classification_head(path: str) -> bool:
    return any(key.startswith("classification_head.") for key in torch.load(path, map_location="cpu"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SAR prediction sanity-check overlays")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--output", default="outputs/current_model_overlays.png")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        cfg = yaml.safe_load(file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_size = tuple(cfg["preprocessing"].get("target_size", [256, 256]))
    model_path = cfg["training"]["best_model_path"]
    model = get_segmentation_model(
        in_channels=cfg["model"].get("input_channels", 2),
        num_classes=1,
        features=checkpoint_features(model_path),
        classification_head=has_classification_head(model_path),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    dataset = SARDataset(
        "data/raw/val/images", "data/raw/val/masks", target_size=target_size,
        is_train=False, mask_format=cfg["training"].get("mask_format", "binary"),
    )
    threshold = cfg["postprocessing"].get("probability_threshold", 0.15)
    samples = []
    with torch.no_grad():
        for images, masks, paths in DataLoader(dataset, batch_size=1, shuffle=False):
            probability = model.predict(images.to(device))[0, 0].cpu().numpy()
            mask = masks[0, 0].numpy()
            prediction = probability >= threshold
            samples.append((paths[0], images[0, 0].numpy(), mask, prediction))
            if len(samples) == args.count:
                break

    columns = 4
    figure, axes = plt.subplots(len(samples), columns, figsize=(12, 3 * len(samples)), squeeze=False)
    for row, (path, image, mask, prediction) in enumerate(samples):
        overlay = np.repeat(image[..., None], 3, axis=-1)
        overlay[mask > 0.5] = [1.0, 0.0, 0.0]
        overlay[prediction] = [0.0, 1.0, 0.0]
        overlay[(mask > 0.5) & prediction] = [1.0, 1.0, 0.0]
        for axis in axes[row]:
            axis.axis("off")
        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_title(os.path.basename(path))
        axes[row, 1].imshow(mask, cmap="gray")
        axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(prediction, cmap="gray")
        axes[row, 2].set_title("Prediction")
        axes[row, 3].imshow(np.clip(overlay, 0, 1))
        axes[row, 3].set_title("Red=GT Green=prediction Yellow=overlap")
    figure.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(f"Saved {len(samples)} overlays to {args.output}")


if __name__ == "__main__":
    main()
