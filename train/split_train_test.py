import os
import random
import argparse
from typing import List, Tuple


def _collect_samples(root: str) -> List[Tuple[str, str]]:
    """
    Collect all (image, mask) patch pairs under 'root', returning
    paths relative to 'root' so they can be reused with different
    absolute roots.

    Expected structure per AOI:
        AOI/images_masked_3x_divide/*.tif
        AOI/masks_3x_divide/*.tif
    """
    samples: List[Tuple[str, str]] = []

    aoi_folders = [
        d for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
    ]

    for aoi in aoi_folders:
        aoi_path = os.path.join(root, aoi)

        image_dir = os.path.join(aoi_path, "images_masked_3x_divide")
        mask_dir = os.path.join(aoi_path, "masks_3x_divide")

        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            continue

        image_files = [f for f in os.listdir(image_dir) if f.endswith(".tif")]

        for f in image_files:
            img_path = os.path.join(image_dir, f)
            mask_path = os.path.join(mask_dir, f)

            if not os.path.exists(mask_path):
                continue

            img_rel = os.path.relpath(img_path, root)
            mask_rel = os.path.relpath(mask_path, root)
            samples.append((img_rel, mask_rel))

    return samples


def create_train_test_split(
    root: str,
    out_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> None:
    """
    Create deterministic train/test split files (train_list.txt, test_list.txt).

    Each line in the output has the format:
        <rel_image_path> <rel_mask_path>

    where paths are relative to 'root'.
    """
    os.makedirs(out_dir, exist_ok=True)

    samples = _collect_samples(root)

    if not samples:
        raise RuntimeError(
            f"No patches found under root '{root}'. "
            "Make sure preprocessing (upsampling + divide_patches) has been run."
        )

    rng = random.Random(seed)
    rng.shuffle(samples)

    n_train = int(len(samples) * train_ratio)
    train_samples = samples[:n_train]
    test_samples = samples[n_train:]

    train_path = os.path.join(out_dir, "train_list.txt")
    test_path = os.path.join(out_dir, "test_list.txt")

    with open(train_path, "w") as f_train:
        for img_rel, mask_rel in train_samples:
            f_train.write(f"{img_rel} {mask_rel}\n")

    with open(test_path, "w") as f_test:
        for img_rel, mask_rel in test_samples:
            f_test.write(f"{img_rel} {mask_rel}\n")

    print(f"Train/test split created at '{out_dir}'.")
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Test samples:  {len(test_samples)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a train/test split for SpaceNet7 patches.")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing AOI folders.")
    parser.add_argument("--out-dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "splits"), help="Output directory for split lists.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of training samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    create_train_test_split(
        root=args.root,
        out_dir=args.out_dir,
        train_ratio=args.train_ratio,
        seed=args.seed
    )

