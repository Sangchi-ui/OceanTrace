import glob
import os
import random

import cv2
import numpy as np
from PIL import Image


def generate_pseudo_mask(img_np: np.ndarray) -> np.ndarray:
    """
    Generates a binary segmentation mask for Class 1 (oil slick) chips.
    Since SAR oil slicks are dark regions of low backscatter intensity,
    adaptive thresholding + morphological filtering isolates the dark slick region.
    """
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np.copy()

    # Gaussian blur to reduce speckle noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu dark thresholding
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned


def prepare_kaggle_dataset(
    kaggle_dir: str = r"D:\PROJECTS\Datasets\kaggle\data",
    output_dir: str = "data/raw",
    sample_limit: int = 1000
):
    """
    Discovers CSIRO Kaggle Sentinel-1 SAR dataset (Class_0: non-oil, Class_1: oil),
    generates binary ground truth segmentation masks, and splits into train/val/test.
    """
    print(f"Inspecting Kaggle dataset at: '{kaggle_dir}'")

    class_0_files = sorted(
        glob.glob(os.path.join(kaggle_dir, "Class_0", "*.jpg")))
    class_1_files = sorted(
        glob.glob(os.path.join(kaggle_dir, "Class_1", "*.jpg")))

    print(f"Found {len(class_0_files)} Class_0 (non-oil) images.")
    print(f"Found {len(class_1_files)} Class_1 (oil spill) images.")

    if not class_1_files:
        raise FileNotFoundError(
            f"No Class_1 images found in {os.path.join(kaggle_dir, 'Class_1')}")

    # Select balanced subset
    if sample_limit and sample_limit < len(class_1_files):
        class_1_selected = class_1_files[:sample_limit]
        class_0_selected = class_0_files[:sample_limit]
    else:
        class_1_selected = class_1_files
        class_0_selected = class_0_files[:len(class_1_files)]

    all_pairs = [(f, 1) for f in class_1_selected] + [(f, 0)
                                                      for f in class_0_selected]
    random.seed(42)
    random.shuffle(all_pairs)

    n_total = len(all_pairs)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)

    splits = {
        "train": all_pairs[:n_train],
        "val": all_pairs[n_train:n_train + n_val],
        "test": all_pairs[n_train + n_val:]
    }

    for split_name, file_list in splits.items():
        img_out_dir = os.path.join(output_dir, split_name, "images")
        mask_out_dir = os.path.join(output_dir, split_name, "masks")
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(mask_out_dir, exist_ok=True)

        for img_path, label_cls in file_list:
            base_name = os.path.basename(img_path)
            name_no_ext = os.path.splitext(base_name)[0]

            # Read image
            img_pil = Image.open(img_path).convert("L")
            img_np = np.array(img_pil)

            # Copy image
            dst_img_path = os.path.join(img_out_dir, f"{name_no_ext}.png")
            img_pil.save(dst_img_path)

            # Create mask: Class 1 -> pseudo-mask of dark slick; Class 0 -> all black background mask
            if label_cls == 1:
                mask_np = generate_pseudo_mask(img_np)
            else:
                mask_np = np.zeros_like(img_np, dtype=np.uint8)

            dst_mask_path = os.path.join(
                mask_out_dir, f"{name_no_ext}_mask.png")
            Image.fromarray(mask_np).save(dst_mask_path)

        print(
            f"Processed '{split_name}' split: {len(file_list)} images saved to '{os.path.join(output_dir, split_name)}'")

    print("Kaggle dataset preparation complete!")


if __name__ == "__main__":
    prepare_kaggle_dataset()
