import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from app.models.unet import get_segmentation_model
from training.dataset import SARDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize clean SAR false alarms")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--metrics", default="outputs/final_sar_metrics.csv")
    parser.add_argument("--output", default="outputs/hard_negative_gallery.png")
    args = parser.parse_args()

    with open(args.config) as file:
        cfg = yaml.safe_load(file)
    metrics = pd.read_csv(args.metrics)
    hard_ids = metrics[(metrics["is_empty_gt"]) & (~metrics["correct_clean_patch"])] \
        .sort_values("pred_foreground_pixels", ascending=False)["sample_id"].tolist()
    if not hard_ids:
        print("No clean false alarms found")
        return

    model_path = cfg["training"]["best_model_path"]
    state = torch.load(model_path, map_location="cpu")
    first_width = state["downs.0.double_conv.0.weight"].shape[0]
    features = [first_width * (2 ** index) for index in range(4)]
    has_head = any(key.startswith("classification_head.") for key in state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_segmentation_model(
        in_channels=cfg["model"].get("input_channels", 2),
        num_classes=1,
        features=features,
        classification_head=has_head,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    dataset = SARDataset(
        "data/raw/test/images", "data/raw/test/masks",
        target_size=tuple(cfg["preprocessing"].get("target_size", [256, 256])),
        is_train=False, mask_format="binary",
        cache_dir=os.path.join(cfg["training"].get("cache_dir", "data/cache/sar"), "test"),
    )
    threshold = cfg["postprocessing"].get("probability_threshold", 0.655)
    rows = []
    with torch.no_grad():
        for images, masks, paths in DataLoader(dataset, batch_size=1, shuffle=False):
            if os.path.basename(paths[0]) not in hard_ids:
                continue
            probability = model.predict(images.to(device))[0, 0].cpu().numpy()
            rows.append((paths[0], images[0, 0].numpy(), probability >= threshold, probability))

    figure, axes = plt.subplots(len(rows), 3, figsize=(10, max(3, 3 * len(rows))), squeeze=False)
    for row, (path, image, prediction, probability) in enumerate(rows):
        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_title(os.path.basename(path))
        axes[row, 1].imshow(probability, cmap="inferno", vmin=0, vmax=1)
        axes[row, 1].set_title("Predicted probability")
        axes[row, 2].imshow(prediction, cmap="gray")
        axes[row, 2].set_title(f"False-positive pixels: {int(prediction.sum())}")
        for axis in axes[row]:
            axis.axis("off")
    figure.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(f"Saved {len(rows)} hard-negative panels to {args.output}")


if __name__ == "__main__":
    main()