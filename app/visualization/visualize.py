import os
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend


def visualize_pipeline_results(
    original_img: np.ndarray,
    preprocessed_img: np.ndarray,
    prob_map: np.ndarray,
    binary_mask: np.ndarray,
    region_records: list[dict[str, Any]],
    output_path: str = "outputs/visualizations/spill_overlay.png",
    title_suffix: str = ""
) -> str:
    """
    Generates a 5-panel comparison figure visualizing the complete Agent 1 detection flow:
    [1. Original SAR] -> [2. Preprocessed SAR] -> [3. Model Probability Map] -> [4. Binary Mask] -> [5. Final Spill Polygon Overlay]
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(
        f"OceanTrace Agent 1 — Sentinel-1 SAR Oil Spill Detection {title_suffix}", fontsize=14, fontweight="bold", y=0.98)

    # 1. Original SAR
    disp_orig = original_img[0] if original_img.ndim == 3 else original_img
    if original_img.ndim == 3 and original_img.shape[0] == 3:
        disp_orig = np.transpose(original_img, (1, 2, 0))
    axes[0].imshow(disp_orig, cmap="gray")
    axes[0].set_title("1. Original SAR Input", fontsize=11)
    axes[0].axis("off")

    # 2. Preprocessed SAR
    disp_preprocessed = preprocessed_img[0] if preprocessed_img.ndim == 3 else preprocessed_img
    if preprocessed_img.ndim == 3 and preprocessed_img.shape[0] == 3:
        disp_preprocessed = np.transpose(preprocessed_img, (1, 2, 0))
    axes[1].imshow(disp_preprocessed, cmap="gray")
    axes[1].set_title("2. Preprocessed (dB + Speckle)", fontsize=11)
    axes[1].axis("off")

    # 3. Model Probability Map
    im_prob = axes[2].imshow(prob_map, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[2].set_title("3. UNet Probability Map", fontsize=11)
    axes[2].axis("off")
    fig.colorbar(im_prob, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Binary Mask
    axes[3].imshow(binary_mask, cmap="binary")
    axes[3].set_title("4. Cleaned Binary Mask", fontsize=11)
    axes[3].axis("off")

    # 5. Final Overlay
    axes[4].imshow(disp_preprocessed, cmap="gray")
    axes[4].imshow(binary_mask, cmap="Reds", alpha=0.45)

    # Overlay centroids and bounding boxes
    for rec in region_records:
        rec.get("measurements", {})
        cx = rec.get("geometry", {}).get("centroid", {}).get("pixel_x")
        cy = rec.get("geometry", {}).get("centroid", {}).get("pixel_y")
        spill_id = rec.get("spill_id", "SPILL")
        conf = rec.get("detection", {}).get("confidence", 0.0)

        if cx is not None and cy is not None:
            axes[4].plot(cx, cy, "r*", markersize=8)
            axes[4].text(cx + 4, cy - 4, f"{spill_id}\nConf: {conf:.2f}", color="yellow", fontsize=8,
                         fontweight="bold", bbox={"boxstyle": "square,pad=0.1", "fc": "black", "ec": "none", "alpha": 0.6})

    axes[4].set_title(
        f"5. Spill Overlay ({len(region_records)} slicks)", fontsize=11)
    axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path
