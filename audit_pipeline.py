import hashlib
import os

import matplotlib.pyplot as plt
import torch

from app.models.unet import get_segmentation_model
from training.dataset import SARDataset


def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def check_dataset_leaks():
    print("="*50)
    print("DATASET LEAKAGE AUDIT")
    print("="*50)

    splits = ["train", "val"]
    data_info = {}

    for split in splits:
        img_dir = f"data/raw/{split}/images"
        mask_dir = f"data/raw/{split}/masks"

        imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(
            ('.png', '.tif'))] if os.path.exists(img_dir) else []
        masks = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(
            ('.png', '.tif'))] if os.path.exists(mask_dir) else []

        hashes = set()
        for img in imgs:
            hashes.add(get_file_hash(img))

        data_info[split] = {
            "imgs": imgs,
            "masks": masks,
            "hashes": hashes
        }

        print(f"{split.upper()}:")
        print(f"images = {len(imgs)}")
        print(f"masks = {len(masks)}\n")

    print("Overlap:")
    train_val_overlap = data_info["train"]["hashes"].intersection(
        data_info["val"]["hashes"])
    print(f"train-val image hash overlap = {len(train_val_overlap)}")


def audit_mask_loading():
    print("\n" + "="*50)
    print("TEST DATALOADER & MASK AUDIT")
    print("="*50)

    test_img_dir = "data/raw/val/images"
    test_mask_dir = "data/raw/val/masks"

    dataset = SARDataset(test_img_dir, test_mask_dir,
                         target_size=(256, 256), is_train=False)
    print(f"Total samples loaded: {len(dataset)}")

    empty_gt_count = 0
    positive_gt_count = 0
    unique_vals = set()

    for i in range(min(5, len(dataset))):
        img_tensor, mask_tensor, img_path = dataset[i]
        mask_path = dataset.mask_paths[i]
        print(f"\nsample ID: {i}")
        print(f"image path: {img_path}")
        print(f"mask path: {mask_path}")
        print(f"image shape: {img_tensor.shape}")
        print(f"mask shape: {mask_tensor.shape}")
        unique = torch.unique(mask_tensor).tolist()
        print(f"unique mask values: {unique}")
        unique_vals.update(unique)

    for i in range(len(dataset)):
        _, mask_tensor, _ = dataset[i]
        if mask_tensor.sum() == 0:
            empty_gt_count += 1
        else:
            positive_gt_count += 1

    print(f"\nGT positive samples: {positive_gt_count}")
    print(f"GT empty samples: {empty_gt_count}")
    print(f"Unique mask values in dataset: {unique_vals}")


def generate_visualisations():
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS & METRICS")
    print("="*50)

    os.makedirs("outputs/audit", exist_ok=True)

    test_img_dir = "data/raw/val/images"
    test_mask_dir = "data/raw/val/masks"
    dataset = SARDataset(test_img_dir, test_mask_dir,
                         target_size=(256, 256), is_train=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/unet_oil_spill.pth"
    model = get_segmentation_model(in_channels=2, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    indices_to_viz = list(range(20))
    _fig, axes = plt.subplots(20, 4, figsize=(16, 60))

    for row, idx in enumerate(indices_to_viz):
        if idx >= len(dataset):
            break

        img_tensor, mask_tensor, img_path = dataset[idx]

        with torch.no_grad():
            img_input = img_tensor.unsqueeze(0).to(device)
            probs = model.predict(img_input)
            preds = (probs >= 0.5).float().cpu().squeeze(0)

        img_disp = img_tensor.permute(1, 2, 0).numpy()
        mask_disp = mask_tensor.squeeze(0).numpy()
        pred_disp = preds.squeeze(0).numpy()

        axes[row, 0].imshow(img_disp)
        axes[row, 0].set_title(f"Input: {os.path.basename(img_path)}")
        axes[row, 0].axis('off')

        axes[row, 1].imshow(mask_disp, cmap='gray', vmin=0, vmax=1)
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis('off')

        axes[row, 2].imshow(pred_disp, cmap='gray', vmin=0, vmax=1)
        axes[row, 2].set_title("Prediction")
        axes[row, 2].axis('off')

        axes[row, 3].imshow(img_disp)
        axes[row, 3].imshow(pred_disp, cmap='Reds', alpha=0.5)
        axes[row, 3].set_title("Overlay")
        axes[row, 3].axis('off')

    plt.tight_layout()
    plt.savefig("outputs/audit/visualizations.png")
    print("Visualizations saved to outputs/audit/visualizations.png")


if __name__ == "__main__":
    check_dataset_leaks()
    audit_mask_loading()
    generate_visualisations()
