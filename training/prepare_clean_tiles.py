import argparse
import random
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def add_clean_tiles(split: str, count: int, tile_size: int, seed: int) -> int:
    image_dir = Path("data/raw") / split / "images"
    mask_dir = Path("data/raw") / split / "masks"
    images = sorted(image_dir.glob("*.tif"))
    rng = random.Random(seed)
    created = 0

    if not images:
        raise FileNotFoundError(f"No TIFF images found in {image_dir}")

    for output_index in range(count):
        for _ in range(200):
            image_path = rng.choice(images)
            mask_path = mask_dir / image_path.name
            with rasterio.open(mask_path) as mask_src:
                height, width = mask_src.height, mask_src.width
                if height < tile_size or width < tile_size:
                    continue
                top = rng.randint(0, height - tile_size)
                left = rng.randint(0, width - tile_size)
                window = Window(left, top, tile_size, tile_size)
                mask = mask_src.read(1, window=window)

            if np.any(mask):
                continue

            with rasterio.open(image_path) as image_src:
                image = image_src.read(window=window)
                profile = image_src.profile.copy()
                profile.update(
                    height=tile_size,
                    width=tile_size,
                    transform=rasterio.windows.transform(window, image_src.transform),
                )

            output_name = f"clean_{split}_{output_index:04d}.tif"
            with rasterio.open(image_dir / output_name, "w", **profile) as dst:
                dst.write(image)

            mask_profile = {
                "driver": "GTiff",
                "height": tile_size,
                "width": tile_size,
                "count": 1,
                "dtype": "uint8",
                "transform": profile["transform"],
            }
            with rasterio.open(mask_dir / output_name, "w", **mask_profile) as dst:
                dst.write(np.zeros((1, tile_size, tile_size), dtype=np.uint8))
            created += 1
            break
        else:
            raise RuntimeError(f"Could not find a clean {tile_size}x{tile_size} window for {split}")

    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Add clean SAR background tiles to each split")
    parser.add_argument("--train", type=int, default=192)
    parser.add_argument("--val", type=int, default=24)
    parser.add_argument("--test", type=int, default=24)
    parser.add_argument("--tile-size", type=int, default=512)
    args = parser.parse_args()

    for index, (split, count) in enumerate(
        (("train", args.train), ("val", args.val), ("test", args.test))
    ):
        print(f"{split}: created {add_clean_tiles(split, count, args.tile_size, 100 + index)} clean tiles")


if __name__ == "__main__":
    main()
