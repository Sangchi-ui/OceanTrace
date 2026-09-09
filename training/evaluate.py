import argparse
import json
import os
import sys

# Add project root to sys.path before importing project modules.
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from app.models.unet import get_segmentation_model
from training.dataset import SARDataset

matplotlib.use('Agg')


def evaluate_model(config_path: str = "config.yaml", split: str = "test", output_path: str = "outputs/metrics.json"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print("OCEANTRACE: U-NET SEGMENTATION EVALUATION")
    print("==================================================")
    print(f"Evaluating model on device: {device}")

    model_path = cfg["training"].get(
        "best_model_path", "models/unet_oil_spill.pth")
    target_size = tuple(cfg["preprocessing"].get("target_size", [256, 256]))

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at '{model_path}'. Please run training first.")

    model = get_segmentation_model(
        architecture=cfg["model"].get("architecture", "unet"),
        in_channels=cfg["model"].get("input_channels", 2),
        num_classes=cfg["model"].get("num_classes", 1),
        features=cfg["model"].get("features", [32, 64, 128, 256]),
        classification_head=cfg["model"].get("classification_head", False)
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Determine split directory
    if split == "test" and os.path.exists("data/raw/test/images") and len(os.listdir("data/raw/test/images")) > 0:
        eval_img_dir = "data/raw/test/images"
        eval_mask_dir = "data/raw/test/masks"
        split_name = "UNTOUCHED TEST SET"
    else:
        eval_img_dir = "data/raw/val/images"
        eval_mask_dir = "data/raw/val/masks"
        split_name = "VALIDATION SET"

    print(f"Target Split: {split_name} ({eval_img_dir})")
    eval_dataset = SARDataset(
        eval_img_dir, eval_mask_dir, target_size=target_size, is_train=False,
        mask_format=cfg["training"].get("mask_format", "binary"),
        in_channels=cfg["model"].get("input_channels", 2))
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    threshold = cfg["postprocessing"].get("probability_threshold", 0.5)
    print(f"Decision Threshold: {threshold}")

    results = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    cached_positive_samples = []

    with torch.no_grad():
        for images, masks, paths in eval_loader:
            img_path = paths[0]
            sample_id = os.path.basename(img_path)

            images = images.to(device)
            masks = masks.to(device)

            probs = model.predict(images)
            preds = (probs >= threshold).float()

            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            tp = int((preds_flat * masks_flat).sum().item())
            fp = int((preds_flat * (1 - masks_flat)).sum().item())
            fn = int(((1 - preds_flat) * masks_flat).sum().item())
            tn = int(((1 - preds_flat) * (1 - masks_flat)).sum().item())

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            gt_foreground = int(masks_flat.sum().item())
            pred_foreground = int(preds_flat.sum().item())

            is_empty_gt = (gt_foreground == 0)

            if is_empty_gt:
                # Clean background patch (no oil ground truth)
                sample_iou = None
                sample_dice = None
                sample_precision = 0.0 if pred_foreground > 0 else 1.0
                sample_recall = 1.0
                sample_f1 = 0.0 if pred_foreground > 0 else 1.0
                correct_clean_patch = (pred_foreground == 0)
            else:
                # Patch containing oil spill
                sample_iou = tp / (tp + fp + fn + 1e-7)
                sample_dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7)
                sample_precision = tp / (tp + fp + 1e-7)
                sample_recall = tp / (tp + fn + 1e-7)
                if sample_precision + sample_recall > 0:
                    sample_f1 = (2 * sample_precision * sample_recall) / \
                        (sample_precision + sample_recall)
                else:
                    sample_f1 = 0.0
                correct_clean_patch = False

                cached_positive_samples.append({
                    "sample_id": sample_id,
                    "img": images[0].cpu().permute(1, 2, 0).numpy(),
                    "mask": masks[0, 0].cpu().numpy(),
                    "pred": preds[0, 0].cpu().numpy(),
                    "iou": sample_iou,
                    "dice": sample_dice,
                    "precision": sample_precision,
                    "recall": sample_recall,
                    "f1": sample_f1,
                    "gt_pixels": gt_foreground,
                    "pred_pixels": pred_foreground
                })

            results.append({
                "sample_id": sample_id,
                "is_empty_gt": is_empty_gt,
                "iou": sample_iou if sample_iou is not None else np.nan,
                "dice": sample_dice if sample_dice is not None else np.nan,
                "precision": sample_precision,
                "recall": sample_recall,
                "f1": sample_f1,
                "gt_foreground_pixels": gt_foreground,
                "pred_foreground_pixels": pred_foreground,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "correct_clean_patch": correct_clean_patch if is_empty_gt else False
            })

    df = pd.DataFrame(results)

    csv_path = output_path.replace(".json", ".csv")
    df.to_csv(csv_path, index=False)

    df_pos = df[~df["is_empty_gt"]]
    df_empty = df[df["is_empty_gt"]]

    # Global Pixel-Level Metrics (Dataset-wide TP/FP/FN/TN)
    global_iou = total_tp / (total_tp + total_fp + total_fn + 1e-7)
    global_dice = (2.0 * total_tp) / \
        (2.0 * total_tp + total_fp + total_fn + 1e-7)
    global_precision = total_tp / (total_tp + total_fp + 1e-7)
    global_recall = total_tp / (total_tp + total_fn + 1e-7)
    global_f1 = (2 * global_precision * global_recall) / (global_precision +
                                                          global_recall + 1e-7) if (global_precision + global_recall > 0) else 0.0

    clean_ocean_accuracy = (df_empty["correct_clean_patch"].sum(
    ) / len(df_empty)) * 100.0 if len(df_empty) > 0 else 0.0

    metrics = {
        "model": cfg["model"].get("architecture", "unet"),
        "evaluated_split": split_name,
        "total_test_samples": len(eval_dataset),
        "threshold": threshold,
        "global_pixel_metrics": {
            "global_iou": float(global_iou),
            "global_dice": float(global_dice),
            "global_precision": float(global_precision),
            "global_recall": float(global_recall),
            "global_f1": float(global_f1),
            "total_true_positives": total_tp,
            "total_false_positives": total_fp,
            "total_false_negatives": total_fn,
            "total_true_negatives": total_tn
        },
        "positive_gt_samples": {
            "count": len(df_pos),
            "percentage": float((len(df_pos) / len(df)) * 100.0),
            "mean_iou": float(df_pos["iou"].dropna().mean()) if len(df_pos) > 0 else 0.0,
            "median_iou": float(df_pos["iou"].dropna().median()) if len(df_pos) > 0 else 0.0,
            "mean_dice": float(df_pos["dice"].dropna().mean()) if len(df_pos) > 0 else 0.0,
            "mean_precision": float(df_pos["precision"].dropna().mean()) if len(df_pos) > 0 else 0.0,
            "mean_recall": float(df_pos["recall"].dropna().mean()) if len(df_pos) > 0 else 0.0,
            "mean_f1": float(df_pos["f1"].dropna().mean()) if len(df_pos) > 0 else 0.0
        },
        "empty_gt_samples": {
            "count": len(df_empty),
            "percentage": float((len(df_empty) / len(df)) * 100.0),
            "clean_background_accuracy_pct": float(clean_ocean_accuracy),
            "false_alarm_count": int((~df_empty["correct_clean_patch"]).sum())
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n--------------------------------------------------")
    print(f"EVALUATION SUMMARY ({split_name}):")
    print("--------------------------------------------------")
    print(f"Total Samples Evaluated:         {metrics['total_test_samples']}")
    print(
        f"  - Positive Samples (w/ Oil):   {metrics['positive_gt_samples']['count']} ({metrics['positive_gt_samples']['percentage']:.2f}%)")
    print(
        f"  - Clean Ocean Samples (Empty): {metrics['empty_gt_samples']['count']} ({metrics['empty_gt_samples']['percentage']:.2f}%)")
    if len(df_empty) == 0:
        print("  WARNING: No clean-ocean masks are present; false-alarm performance cannot be measured.")
    print("\nGlobal Pixel-Level Performance:")
    print(
        f"  Global IoU:                    {metrics['global_pixel_metrics']['global_iou']:.4f}")
    print(
        f"  Global Dice (F1):              {metrics['global_pixel_metrics']['global_dice']:.4f}")
    print(
        f"  Global Precision:              {metrics['global_pixel_metrics']['global_precision']:.4f}")
    print(
        f"  Global Recall:                 {metrics['global_pixel_metrics']['global_recall']:.4f}")
    print("\nPositive-Sample Metrics (Oil Spill Present):")
    print(
        f"  Mean IoU:                      {metrics['positive_gt_samples']['mean_iou']:.4f}")
    print(
        f"  Mean Dice:                     {metrics['positive_gt_samples']['mean_dice']:.4f}")
    print(
        f"  Mean Precision:                {metrics['positive_gt_samples']['mean_precision']:.4f}")
    print(
        f"  Mean Recall:                   {metrics['positive_gt_samples']['mean_recall']:.4f}")
    print("\nClean Ocean Patch Performance:")
    print(
        f"  Clean Accuracy:                {metrics['empty_gt_samples']['clean_background_accuracy_pct']:.2f}%")
    print(
        f"  False Alarms:                  {metrics['empty_gt_samples']['false_alarm_count']} / {metrics['empty_gt_samples']['count']}")

    print(f"\nDetailed per-sample records saved to: '{csv_path}'")
    print(f"JSON metrics saved to:                '{output_path}'")

    # Generate Visualizations for Best & Worst Samples
    if len(cached_positive_samples) > 0:
        sorted_samples = sorted(cached_positive_samples,
                                key=lambda x: x["iou"], reverse=True)
        top_best = sorted_samples[:3]
        top_worst = sorted_samples[-3:]
        samples_to_plot = [("Top Best", s) for s in top_best] + \
            [("Top Worst", s) for s in top_worst]

        _fig, axes = plt.subplots(len(samples_to_plot),
                                 4, figsize=(16, 4 * len(samples_to_plot)))
        if len(samples_to_plot) == 1:
            axes = np.expand_dims(axes, axis=0)

        for row, (group, s) in enumerate(samples_to_plot):
            # Input
            input_image = s["img"]
            if input_image.shape[-1] == 1:
                input_image = input_image[..., 0]
            elif input_image.shape[-1] == 2:
                input_image = input_image[..., 0]
            axes[row, 0].imshow(np.clip(input_image, 0, 1), cmap="gray")
            axes[row, 0].set_title(
                f"[{group}] {s['sample_id']}\nInput Satellite Image", fontsize=9)
            axes[row, 0].axis("off")

            # Ground Truth
            axes[row, 1].imshow(s["mask"], cmap="copper")
            axes[row, 1].set_title(
                f"Ground Truth (Oil Pixels: {s['gt_pixels']})", fontsize=9)
            axes[row, 1].axis("off")

            # Prediction
            axes[row, 2].imshow(s["pred"], cmap="inferno")
            axes[row, 2].set_title(
                f"Prediction (IoU: {s['iou']:.3f} | Dice: {s['dice']:.3f})", fontsize=9)
            axes[row, 2].axis("off")

            # Overlay
            if s["img"].shape[-1] == 2:
                ov = np.repeat(s["img"][..., :1], 3, axis=-1)
            elif s["img"].shape[-1] == 1:
                ov = np.repeat(s["img"], 3, axis=-1)
            else:
                ov = s["img"].copy()
            ov = np.clip(ov, 0, 1)
            # Red for GT, Green for Pred, Yellow for Overlap
            gt_mask = (s["mask"] == 1)
            pred_mask = (s["pred"] == 1)
            overlap = gt_mask & pred_mask

            ov[gt_mask] = [1.0, 0.0, 0.0]        # Red: GT
            ov[pred_mask] = [0.0, 1.0, 0.0]      # Green: Pred
            # Yellow: True Positive Overlap
            ov[overlap] = [1.0, 1.0, 0.0]

            axes[row, 3].imshow(ov)
            axes[row, 3].set_title(
                "Overlay (Red=GT, Green=Pred, Yellow=TP)", fontsize=9)
            axes[row, 3].axis("off")

        plt.tight_layout()
        vis_plot_path = "outputs/evaluation_visualizations.png"
        plt.savefig(vis_plot_path, dpi=150)
        plt.close()
        print(f"Prediction visual gallery saved to:   '{vis_plot_path}'")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate UNet Oil Spill Segmentation Model")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--split", type=str, default="test",
                        help="Split to evaluate: 'test' or 'val'")
    parser.add_argument("--output", type=str, default="outputs/metrics.json")
    args = parser.parse_args()
    evaluate_model(args.config, args.split, args.output)
