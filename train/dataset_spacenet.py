import os
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset


class SpaceNetDataset(Dataset):

    def __init__(self, root, transform=None, split=None, split_list_dir=None):

        self.samples = []
        self.transform = transform

        # -----------------------------------
        # SPLIT MODE
        # -----------------------------------
        if split is not None:
            if split_list_dir is None:
                raise ValueError("split_list_dir must be provided if split is specified.")

            list_path = os.path.join(split_list_dir, f"{split}_list.txt")

            if not os.path.exists(list_path):
                raise FileNotFoundError(
                    f"Split file '{list_path}' not found."
                )

            with open(list_path, "r") as f:
                for line in f:

                    line = line.strip()

                    if not line:
                        continue

                    img_rel, mask_rel = line.split()

                    img_path = os.path.join(root, img_rel)
                    mask_path = os.path.join(root, mask_rel)

                    if os.path.exists(img_path) and os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path))

            print(f"Total patches in split '{split}':", len(self.samples))
            return

        # -----------------------------------
        # DIRECTORY SCAN MODE
        # -----------------------------------

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

            image_files = [
                f for f in os.listdir(image_dir) if f.endswith(".tif")
            ]

            for f in image_files:

                img_path = os.path.join(image_dir, f)
                mask_path = os.path.join(mask_dir, f)

                if os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path))

        print("Total patches:", len(self.samples))


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):

        img_path, mask_path = self.samples[idx]

        # Load image
        with rasterio.open(img_path) as src:
            image = src.read()[:4].astype(np.float32)

        # Load mask
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)

        # Normalize
        image = image / 255.0

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask