import argparse
import json
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


def collect_validation_predictions(config_path: str):
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_size = tuple(cfg["preprocessing"].get("target_size", [256, 256]))
    model_path = cfg["training"]["best_model_path"]
    model = get_segmentation_model(
        architecture=cfg["model"].get("architecture", "unet"),
        in_channels=cfg["model"].get("input_channels", 2),
        num_classes=cfg["model"].get("num_classes", 1),
        features=cfg["model"].get("features", [32, 64, 128, 256]),
        classification_head=cfg["model"].get("classification_head", False),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    dataset = SARDataset(
        "data/raw/val/images", "data/raw/val/masks",
        target_size=target_size, is_train=False,
        mask_format=cfg["training"].get("mask_format", "binary"),
        in_channels=cfg["model"].get("input_channels", 2)
    )
    sample_predictions = []
    with torch.no_grad():
        for images, masks, _ in DataLoader(dataset, batch_size=4, shuffle=False):
            probs = model.predict(images.to(device)).cpu().numpy()
            for probability, target in zip(probs, masks.numpy()):
                sample_predictions.append((probability.ravel(), target.ravel()))
    return sample_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune segmentation threshold on validation data")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="outputs/threshold_metrics.json")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        cfg = yaml.safe_load(file)
    sample_predictions = collect_validation_predictions(args.config)
    thresholds = np.linspace(0.01, 0.99, 197)
    rows = []
    for threshold in thresholds:
        tp = fp = fn = 0
        clean_correct = 0
        clean_count = 0
        for probabilities, targets in sample_predictions:
            predictions = probabilities >= threshold
            tp += np.logical_and(predictions, targets >= 0.5).sum()
            fp += np.logical_and(predictions, targets < 0.5).sum()
            fn += np.logical_and(~predictions, targets >= 0.5).sum()
            if not (targets >= 0.5).any():
                clean_count += 1
                clean_correct += int(not predictions.any())
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        iou = tp / (tp + fp + fn + 1e-7)
        clean_accuracy = clean_correct / clean_count if clean_count else 0.0
        rows.append({"threshold": float(threshold), "precision": float(precision), "recall": float(recall), "f1": float(f1), "iou": float(iou), "false_positives": int(fp), "clean_accuracy": float(clean_accuracy), "clean_count": clean_count})

    clean_target = cfg["training"].get("clean_accuracy_target", 0.85)
    eligible = [row for row in rows if row["clean_accuracy"] >= clean_target]
    best = max(eligible or rows, key=lambda row: row["f1"])
    best["clean_accuracy_target"] = clean_target
    best["clean_accuracy_constraint_met"] = bool(eligible)
    validation_pixels = sum(target.size for _, target in sample_predictions)
    positive_pixels = sum(int((target >= 0.5).sum()) for _, target in sample_predictions)
    result = {"best": best, "points": rows, "validation_pixels": validation_pixels, "positive_pixels": positive_pixels}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as file:
        json.dump(result, file, indent=2)

    precision = [row["precision"] for row in rows]
    recall = [row["recall"] for row in rows]
    clean_acc = [row["clean_accuracy"] for row in rows]
    threshold_vals = [row["threshold"] for row in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(recall, precision, label="Validation PR curve")
    ax1.scatter([best["recall"]], [best["precision"]], label=f"Best F1={best['f1']:.3f} @ {best['threshold']:.3f}", color='red', zorder=5)
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision-Recall Curve")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(threshold_vals, clean_acc, label="Clean Accuracy", color="green")
    ax2.axhline(clean_target, color="red", linestyle="--", label=f"Target ({clean_target})")
    ax2.scatter([best["threshold"]], [best["clean_accuracy"]], label=f"Chosen Threshold", color='red', zorder=5)
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Clean Accuracy")
    ax2.set_title("Clean Accuracy vs Threshold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig("outputs/validation_pr_curve.png", dpi=160)
    plt.close(fig)
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
