import os
import shutil


def setup_directories():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dirs = {
        "train_img": os.path.join(base_dir, "data", "raw", "train", "images"),
        "train_mask": os.path.join(base_dir, "data", "raw", "train", "masks"),
        "val_img": os.path.join(base_dir, "data", "raw", "val", "images"),
        "val_mask": os.path.join(base_dir, "data", "raw", "val", "masks"),
        "test_img": os.path.join(base_dir, "data", "raw", "test", "images"),
        "test_mask": os.path.join(base_dir, "data", "raw", "test", "masks"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def parse_and_copy_split(split_file: str, dataset_root: str, target_img_dir: str, target_mask_dir: str):
    if not os.path.exists(split_file):
        print(f"Warning: Split file not found: {split_file}")
        return

    with open(split_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    copied = 0
    for line in lines:
        # e.g. "Scene_0_1" -> scene="Scene_0", patch="1"
        parts = line.rsplit("_", 1)
        if len(parts) != 2:
            continue
        scene, patch_idx = parts[0], parts[1]

        # Resolution folder where RGB and masks reside
        res_dir = os.path.join(dataset_root, scene, "10")

        # MADOS uses _rgb_ for image and _cl_ for mask (class)
        img_name = f"{scene}_L2R_rgb_{patch_idx}.png"
        mask_name = f"{scene}_L2R_cl_{patch_idx}.tif"

        src_img = os.path.join(res_dir, img_name)
        src_mask = os.path.join(res_dir, mask_name)

        dst_img = os.path.join(target_img_dir, img_name)
        dst_mask = os.path.join(target_mask_dir, mask_name)

        if os.path.exists(src_img) and os.path.exists(src_mask):
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_mask, dst_mask)
            copied += 1
        else:
            print(f"Missing files for {line}: {img_name} or {mask_name}")

    print(f"Copied {copied} files to {target_img_dir}")


def main():
    dataset_root = r"D:\PROJECTS\Datasets\MADOS"
    splits_dir = os.path.join(dataset_root, "splits")
    train_split = os.path.join(splits_dir, "train_X.txt")
    val_split = os.path.join(splits_dir, "val_X.txt")
    test_split = os.path.join(splits_dir, "test_X.txt")

    print(f"Preparing MADOS dataset from {dataset_root}...")
    dirs = setup_directories()

    print("Processing training split...")
    parse_and_copy_split(train_split, dataset_root,
                         dirs["train_img"], dirs["train_mask"])

    print("Processing validation split...")
    parse_and_copy_split(val_split, dataset_root,
                         dirs["val_img"], dirs["val_mask"])

    if os.path.exists(test_split):
        print("Processing test split...")
        parse_and_copy_split(test_split, dataset_root,
                             dirs["test_img"], dirs["test_mask"])

    print("Dataset preparation complete!")


if __name__ == "__main__":
    main()
