import os

import numpy as np
from PIL import Image, ImageDraw

try:
    import rasterio
    from rasterio.transform import from_origin
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


def generate_sar_scene(
    width: int = 512,
    height: int = 512,
    num_slicks: int = 2,
    seed: int = 42,
    inject_hard_negative: bool = False
) -> tuple:
    """
    Simulates a realistic Sentinel-1 SAR scene of open ocean with dark oil slick candidates.
    SAR features simulated:
    - High ocean backscatter background (~0.6 - 0.8 intensity)
    - Speckle noise (multiplicative Rayleigh/Gamma noise)
    - Dark oil slicks with suppressed backscatter (~0.1 - 0.2 intensity)
    - Irregular organic slick shapes with thin tails
    - Hard negative look-alikes if requested (wind-shadow zones or biogenic slicks)
    """
    np.random.seed(seed)

    # 1. Base ocean background with slight spatial intensity gradient (wind field variation)
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x, y)
    ocean_background = 0.65 + 0.1 * np.sin(3 * xx) * np.cos(2 * yy)

    # 2. Add SAR Speckle Noise (multiplicative noise)
    speckle = np.random.gamma(shape=4.0, scale=0.25, size=(height, width))
    sar_image = ocean_background * speckle

    # 3. Create ground truth mask and draw oil slicks
    mask_pil = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_pil)

    for i in range(num_slicks):
        cx = np.random.randint(100, width - 100)
        cy = np.random.randint(100, height - 100)
        rx = np.random.randint(30, 70)
        ry = np.random.randint(15, 40)
        angle = np.random.randint(0, 180)

        # Draw main slick body
        bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
        draw.ellipse(bbox, fill=255)

        # Draw slick drift tail
        tail_end_x = cx + int(rx * 1.8 * np.cos(np.radians(angle)))
        tail_end_y = cy + int(ry * 1.8 * np.sin(np.radians(angle)))
        draw.line([cx, cy, tail_end_x, tail_end_y],
                  fill=255, width=np.random.randint(8, 16))

    mask = np.array(mask_pil, dtype=np.float32) / 255.0

    # 4. Suppress SAR backscatter where oil slicks exist (damping surface capillary waves)
    slick_damping = 0.15 + 0.05 * np.random.randn(height, width)
    slick_damping = np.clip(slick_damping, 0.05, 0.3)

    sar_image = np.where(mask > 0.5, sar_image * slick_damping, sar_image)

    # 5. Inject hard negative look-alikes (do not update ground truth mask)
    if inject_hard_negative:
        cx = np.random.randint(100, width - 100)
        cy = np.random.randint(100, height - 100)
        rx = np.random.randint(80, 150)
        ry = np.random.randint(40, 80)
        
        hn_mask_pil = Image.new("L", (width, height), 0)
        hn_draw = ImageDraw.Draw(hn_mask_pil)
        bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
        hn_draw.ellipse(bbox, fill=255)
        hn_mask = np.array(hn_mask_pil, dtype=np.float32) / 255.0

        # Hard negative damping is slightly less intense or more uniform than oil
        hn_damping = 0.25 + 0.1 * np.random.randn(height, width)
        hn_damping = np.clip(hn_damping, 0.1, 0.4)
        sar_image = np.where(hn_mask > 0.5, sar_image * hn_damping, sar_image)

    sar_image = np.clip(sar_image, 0.0, 2.5)

    return sar_image.astype(np.float32), (mask > 0.5).astype(np.uint8)


def save_geotiff(filename: str, img_data: np.ndarray, crs: str = "EPSG:4326", origin: tuple = (83.3, 17.7), pixel_size: float = 0.0001):
    """
    Saves numpy 2D array as a georeferenced GeoTIFF.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if RASTERIO_AVAILABLE:
        transform = from_origin(origin[0], origin[1], pixel_size, pixel_size)
        height, width = img_data.shape

        with rasterio.open(
            filename,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=img_data.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(img_data, 1)
    else:
        # Fallback to PNG if rasterio is unavailable
        png_path = filename.replace(".tif", ".png")
        norm = (img_data - img_data.min()) / \
            (img_data.max() - img_data.min() + 1e-6)
        pil = Image.fromarray((norm * 255).astype(np.uint8))
        pil.save(png_path)


def generate_benchmark_dataset(base_dir: str = "data"):
    """
    Generates train, validation, test datasets, and sample image for local prototype runs.
    """
    splits = {
        "train": (15, os.path.join(base_dir, "raw", "train")),
        "val": (5, os.path.join(base_dir, "raw", "val")),
        "test": (5, os.path.join(base_dir, "raw", "test"))
    }

    for split_name, (count, split_path) in splits.items():
        img_dir = os.path.join(split_path, "images")
        mask_dir = os.path.join(split_path, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        for i in range(count):
            seed = 100 if split_name == "train" else (
                200 if split_name == "val" else 300)
            # Inject hard negatives in ~30% of the training images
            inject_hn = (split_name == "train" and np.random.rand() < 0.3)
            sar_img, mask = generate_sar_scene(
                width=512, height=512, num_slicks=np.random.randint(1, 3), seed=seed + i, inject_hard_negative=inject_hn)

            # Save GeoTIFF image
            img_file = os.path.join(img_dir, f"sar_{split_name}_{i+1:03d}.tif")
            save_geotiff(img_file, sar_img, origin=(
                83.3 + i * 0.05, 17.7 + i * 0.05))

            # Save PNG ground truth mask
            mask_file = os.path.join(
                mask_dir, f"sar_{split_name}_{i+1:03d}.png")
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_file)

    # Save a dedicated sample image for quick CLI inference testing
    sample_dir = os.path.join(base_dir, "sample")
    os.makedirs(sample_dir, exist_ok=True)
    sample_sar, sample_mask = generate_sar_scene(
        width=512, height=512, num_slicks=2, seed=999)
    save_geotiff(os.path.join(sample_dir, "sample_sentinel1.tif"),
                 sample_sar, origin=(83.31, 17.68))
    Image.fromarray((sample_mask * 255).astype(np.uint8)
                    ).save(os.path.join(sample_dir, "sample_sentinel1_mask.png"))

    print(
        f"Benchmark synthetic Sentinel-1 SAR dataset generated in '{base_dir}/'")


if __name__ == "__main__":
    generate_benchmark_dataset()
