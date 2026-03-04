import os
from tqdm import tqdm
import numpy as np
import rasterio
from rasterio.enums import Resampling


# ---------------------------------------------------
# Utility: safe AOI listing (ignore .DS_Store etc.)
# ---------------------------------------------------
def get_aoi_folders(root):
    return [
        d for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
    ]


# ---------------------------------------------------
# Optional resize before slicing
# ---------------------------------------------------
def resize_raster(data, src, out_h, out_w, is_mask):
    resampling = Resampling.nearest if is_mask else Resampling.bilinear
    resized = src.read(
        out_shape=(src.count, out_h, out_w),
        resampling=resampling
    )
    return resized


# ---------------------------------------------------
# Pad patch to 512×512 if needed
# ---------------------------------------------------
def pad_patch(patch, target_size=512, padding_pixel=255):
    c, h, w = patch.shape

    if h == target_size and w == target_size:
        return patch

    padded = np.full((c, target_size, target_size), padding_pixel, dtype=patch.dtype)
    padded[:, :h, :w] = patch
    return padded


# ---------------------------------------------------
# Divide single raster into patches
# ---------------------------------------------------
def divide_img(
        in_path,
        out_dir,
        patch_size=512,
        stride=512,
        padding_pixel=255,
        pre_height=None,
        pre_width=None,
        is_mask=False):

    with rasterio.open(in_path) as src:

        data = src.read()
        meta = src.meta.copy()

        height = src.height
        width = src.width

        # Optional pre-resize
        if pre_height and pre_width:
            data = resize_raster(data, src, pre_height, pre_width, is_mask)
            height = pre_height
            width = pre_width

        base = os.path.splitext(os.path.basename(in_path))[0]

        for y in range(0, height, stride):
            for x in range(0, width, stride):

                patch = data[:, y:y+patch_size, x:x+patch_size]

                patch = pad_patch(patch, patch_size, padding_pixel)

                patch_name = f"{base}_{y:05d}_{x:05d}.tif"
                out_path = os.path.join(out_dir, patch_name)

                meta_patch = meta.copy()
                meta_patch.update({
                    "height": patch_size,
                    "width": patch_size,
                    "transform": src.transform,
                })

                with rasterio.open(out_path, "w", **meta_patch) as dst:
                    dst.write(patch)


# ---------------------------------------------------
# Main divide function
# ---------------------------------------------------
def divide(
        root,
        scale=3,
        patch_size=512,
        stride=512,
        padding_pixel=255,
        pre_height=None,
        pre_width=None):

    aoi_folders = get_aoi_folders(root)

    for aoi in tqdm(aoi_folders, desc="Dividing AOIs"):

        aoi_path = os.path.join(root, aoi)

        image_dir = os.path.join(aoi_path, f"images_masked_{scale}x")
        mask_dir = os.path.join(aoi_path, f"masks_{scale}x")

        out_img_dir = os.path.join(aoi_path, f"images_masked_{scale}x_divide")
        out_mask_dir = os.path.join(aoi_path, f"masks_{scale}x_divide")

        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            continue

        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)

        image_files = [f for f in os.listdir(image_dir) if f.endswith(".tif")]

        for img_file in tqdm(image_files, desc=aoi, leave=False):

            name_root = os.path.splitext(img_file)[0]

            image_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, name_root + ".tif")

            if not os.path.exists(mask_path):
                continue

            # Divide image
            divide_img(
                image_path,
                out_img_dir,
                patch_size,
                stride,
                padding_pixel,
                pre_height,
                pre_width,
                is_mask=False
            )

            # Divide mask
            divide_img(
                mask_path,
                out_mask_dir,
                patch_size,
                stride,
                padding_pixel,
                pre_height,
                pre_width,
                is_mask=True
            )


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Divide Spacenet upscaled images and masks into patches.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="../data/SN7_buildings_train/train",
        help="Path to the training data root directory containing AOI folders"
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=3,
        help="Scale factor the images were upsampled by (default: 3)"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=512,
        help="Size of the square patches to extract (default: 512)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Stride for extracting patches (default: 512)"
    )
    parser.add_argument(
        "--pre-height",
        type=int,
        default=None,
        help="Optional target height to resize to before patching"
    )
    parser.add_argument(
        "--pre-width",
        type=int,
        default=None,
        help="Optional target width to resize to before patching"
    )

    args = parser.parse_args()

    print(f"Using data root: {args.data_root}")
    print(f"Scale: {args.scale}x")
    print(f"Patch Size: {args.patch_size}x{args.patch_size}, Stride: {args.stride}")

    divide(
        args.data_root,
        scale=args.scale,
        patch_size=args.patch_size,
        stride=args.stride,
        padding_pixel=255,
        pre_height=args.pre_height,
        pre_width=args.pre_width
    )

    print("Patch slicing complete")