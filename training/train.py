import argparse
import os
import sys

# Add project root to sys.path before importing project modules.
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader

from app.models.unet import get_segmentation_model
from training.dataset import SARDataset
from training.losses import CombinedBCETverskyLoss


def train_model(config_path: str = "config.yaml", resume: bool | None = None):
    # Load configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["training"].get("seed", 42)
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        torch.set_num_threads(min(8, os.cpu_count() or 4))

    # Device selection (CUDA GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for training: {device}", flush=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Hyperparameters
    batch_size = cfg["training"].get("batch_size", 16)
    learning_rate = cfg["training"].get("learning_rate", 0.001)
    epochs = cfg["training"].get("epochs", 5)
    target_size = tuple(cfg["preprocessing"].get("target_size", [256, 256]))
    checkpoint_dir = cfg["training"].get(
        "checkpoint_dir", "models/checkpoints")
    best_model_path = cfg["training"].get(
        "best_model_path", "models/unet_oil_spill.pth")
    latest_checkpoint_path = cfg["training"].get(
        "latest_checkpoint_path", os.path.join(checkpoint_dir, "latest.pt"))
    resume = cfg["training"].get("resume", False) if resume is None else resume

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    # Datasets & DataLoaders
    train_img_dir = "data/raw/train/images"
    train_mask_dir = "data/raw/train/masks"
    val_img_dir = "data/raw/val/images"
    val_mask_dir = "data/raw/val/masks"

    if not os.path.exists(train_img_dir):
        print("Training dataset directory not found. Auto-generating synthetic benchmark dataset...", flush=True)
        from training.generate_synthetic_data import generate_benchmark_dataset
        generate_benchmark_dataset()

    train_dataset = SARDataset(
        train_img_dir, train_mask_dir, target_size=target_size, is_train=True,
        positive_crop_probability=cfg["training"].get(
            "positive_crop_probability", 0.75),
        mask_format=cfg["training"].get("mask_format", "binary"),
        speckle_noise_probability=cfg["training"].get("speckle_noise_probability", 0.25),
        intensity_jitter_probability=cfg["training"].get("intensity_jitter_probability", 0.25),
        cache_dir=os.path.join(cfg["training"].get("cache_dir", "data/cache/sar"), "train"),
        in_channels=cfg["model"].get("input_channels", 2))
    val_dataset = SARDataset(val_img_dir, val_mask_dir,
                             target_size=target_size, is_train=False,
                             mask_format=cfg["training"].get("mask_format", "binary"),
                             cache_dir=os.path.join(cfg["training"].get("cache_dir", "data/cache/sar"), "val"),
                             in_channels=cfg["model"].get("input_channels", 2))

    sample_weights = train_dataset.get_sample_weights()
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    num_workers = cfg["training"].get("num_workers", 0)
    loader_options = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_options["persistent_workers"] = True
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, **loader_options)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, **loader_options)

    # Model, Optimizer, Criterion
    model = get_segmentation_model(
        architecture=cfg["model"].get("architecture", "unet"),
        in_channels=cfg["model"].get("input_channels", 3),
        num_classes=cfg["model"].get("num_classes", 1),
        features=cfg["model"].get("features", [64, 128, 256, 512]),
        classification_head=cfg["model"].get("classification_head", False)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=cfg["training"].get("weight_decay", 1e-4))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = CombinedBCETverskyLoss(
        bce_weight=cfg["training"].get("bce_weight", 0.5),
        tversky_weight=cfg["training"].get("tversky_weight", 0.5),
        alpha=cfg["training"].get("tversky_alpha", 0.7),
        beta=cfg["training"].get("tversky_beta", 0.3),
        positive_class_weight=cfg["training"].get("positive_class_weight", 10.0))

    best_val_loss = float("inf")
    best_val_iou = float("-inf")
    patience = cfg["training"].get("patience", 10)
    min_delta = cfg["training"].get("min_delta", 0.001)
    epochs_no_improve = 0
    validation_threshold = cfg["training"].get("validation_threshold", 0.15)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    start_epoch = 1

    if resume and os.path.exists(latest_checkpoint_path):
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        best_val_loss = checkpoint["best_val_loss"]
        best_val_iou = checkpoint.get("best_val_iou", checkpoint.get("best_val_dice", float("-inf")))
        epochs_no_improve = checkpoint["epochs_no_improve"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch} using '{latest_checkpoint_path}'", flush=True)

    print(
        f"Starting UNet training for {epochs} epochs on {len(train_dataset)} training samples...", flush=True)

    total_batches = len(train_loader)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch_idx, (images, masks, _) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    output = model(images)
                    if isinstance(output, tuple):
                        logits, class_logits = output
                        segmentation_loss = criterion(logits, masks)
                        class_targets = (masks.flatten(1).max(dim=1).values > 0).float().unsqueeze(1)
                        classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            class_logits, class_targets)
                        loss = segmentation_loss + cfg["training"].get("classification_loss_weight", 0.5) * classification_loss
                    else:
                        loss = criterion(output, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_loss = loss.item()
            train_loss += batch_loss * images.size(0)

            if batch_idx % 10 == 0 or batch_idx == total_batches:
                print(
                    f"  Epoch [{epoch:02d}/{epochs:02d}] Batch [{batch_idx:03d}/{total_batches:03d}] - Loss: {batch_loss:.4f}", flush=True)

        train_loss /= max(len(train_dataset), 1)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_tp = val_fp = val_fn = 0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    output = model(images)
                    logits = output[0] if isinstance(output, tuple) else output
                    loss = criterion(logits, masks)
                val_loss += loss.item() * images.size(0)
                predictions = torch.sigmoid(logits) >= validation_threshold
                val_tp += int((predictions & (masks >= 0.5)).sum().item())
                val_fp += int((predictions & (masks < 0.5)).sum().item())
                val_fn += int(((~predictions) & (masks >= 0.5)).sum().item())

        val_loss /= max(len(val_dataset), 1)
        val_iou = val_tp / (val_tp + val_fp + val_fn + 1e-7)
        val_dice = (2.0 * val_tp) / (2.0 * val_tp + val_fp + val_fn + 1e-7)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] Summary -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f}", flush=True)

        scheduler.step()

        # Save checkpoint if best validation loss
        if epoch == 1 or val_iou > best_val_iou + min_delta:
            best_val_loss = val_loss
            best_val_iou = val_iou
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(
                f"  --> Saved new best checkpoint to '{best_model_path}' (Val IoU: {best_val_iou:.4f})", flush=True)
        else:
            epochs_no_improve += 1
            print(
                f"  --> Early stopping counter: {epochs_no_improve} out of {patience}")
            if epochs_no_improve >= patience:
                print("Early stopping triggered!", flush=True)
                break

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_val_iou": best_val_iou,
            "best_val_dice": val_dice,
            "epochs_no_improve": epochs_no_improve,
            "config": cfg,
        }, latest_checkpoint_path)

    print("Training complete!", flush=True)
    return best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PyTorch UNet for SAR Oil Spill Detection")
    parser.add_argument("--config", type=str,
                        default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest epoch checkpoint")
    args = parser.parse_args()
    train_model(args.config, resume=True if args.resume else None)
