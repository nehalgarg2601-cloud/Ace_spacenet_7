import os
from tqdm import tqdm
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation, binary_erosion


# ---------------------------------------------------
# Utility: get AOI folders safely (ignore .DS_Store)
# ---------------------------------------------------
def get_aoi_folders(root):

    return [
        d for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
    ]


# ---------------------------------------------------
# 1. Upsample images
# ---------------------------------------------------
def enlarge_images(root, scale=3):

    aoi_folders = get_aoi_folders(root)

    for aoi in tqdm(aoi_folders, desc="Upsampling images"):

        aoi_path = os.path.join(root, aoi)

        image_dir = os.path.join(aoi_path, "images_masked")
        out_dir = os.path.join(aoi_path, f"images_masked_{scale}x")

        if not os.path.exists(image_dir):
            continue

        os.makedirs(out_dir, exist_ok=True)

        image_files = [f for f in os.listdir(image_dir) if f.endswith(".tif")]

        for img_file in image_files:

            in_path = os.path.join(image_dir, img_file)
            out_path = os.path.join(out_dir, img_file)

            with rasterio.open(in_path) as src:

                new_height = src.height * scale
                new_width = src.width * scale

                data = src.read(
                    out_shape=(src.count, new_height, new_width),
                    resampling=Resampling.bilinear
                )

                transform = src.transform * src.transform.scale(
                    src.width / new_width,
                    src.height / new_height
                )

                meta = src.meta.copy()
                meta.update({
                    "height": new_height,
                    "width": new_width,
                    "transform": transform
                })

                with rasterio.open(out_path, "w", **meta) as dst:
                    dst.write(data)


# ---------------------------------------------------
# 2. Create labels aligned with upscaled images
# ---------------------------------------------------
def create_label(root, scale=3, create_fbc=True):

    aoi_folders = get_aoi_folders(root)

    for aoi in tqdm(aoi_folders, desc="Creating labels"):

        aoi_path = os.path.join(root, aoi)

        image_dir = os.path.join(aoi_path, f"images_masked_{scale}x")
        label_dir = os.path.join(aoi_path, "labels_match")

        mask_dir = os.path.join(aoi_path, f"masks_{scale}x")
        fbc_dir = os.path.join(aoi_path, f"masks_fbc_{scale}x")

        if not os.path.exists(image_dir):
            continue

        os.makedirs(mask_dir, exist_ok=True)

        if create_fbc:
            os.makedirs(fbc_dir, exist_ok=True)

        image_files = [f for f in os.listdir(image_dir) if f.endswith(".tif")]

        for img_file in image_files:

            name_root = os.path.splitext(img_file)[0]

            image_path = os.path.join(image_dir, img_file)
            json_path = os.path.join(label_dir, name_root + "_Buildings.geojson")

            if not os.path.exists(json_path):
                continue

            with rasterio.open(image_path) as src:

                height = src.height
                width = src.width
                transform = src.transform
                crs = src.crs
                meta = src.meta.copy()

            try:
                gdf = gpd.read_file(json_path)
            except:
                gdf = gpd.GeoDataFrame(geometry=[])

            if not gdf.empty:

                if gdf.crs != crs:
                    gdf = gdf.to_crs(crs)

                shapes = [
                    (geom, 1)
                    for geom in gdf.geometry
                    if geom is not None and not geom.is_empty
                ]

                if len(shapes) > 0:
                    footprint = rasterize(
                        shapes,
                        out_shape=(height, width),
                        transform=transform,
                        fill=0,
                        dtype=np.uint8
                    )
                else:
                    footprint = np.zeros((height, width), dtype=np.uint8)

            else:
                footprint = np.zeros((height, width), dtype=np.uint8)

            # ------------------------------------------
            # Save basic footprint mask
            # ------------------------------------------
            meta.update({"count": 1, "dtype": "uint8"})

            mask_path = os.path.join(mask_dir, name_root + ".tif")

            with rasterio.open(mask_path, "w", **meta) as dst:
                dst.write(footprint, 1)

            # ------------------------------------------
            # Optional FBC mask
            # ------------------------------------------
            if create_fbc:

                dilated = binary_dilation(footprint, iterations=1)
                eroded = binary_erosion(footprint, iterations=1)

                boundary = (dilated ^ eroded).astype(np.uint8)

                expanded = binary_dilation(footprint, iterations=3)
                contact = np.logical_and(expanded, np.logical_not(footprint)).astype(np.uint8)

                fbc = np.stack([footprint, boundary, contact], axis=0)

                meta.update({"count": 3})

                fbc_path = os.path.join(fbc_dir, name_root + "_fbc.tif")

                with rasterio.open(fbc_path, "w", **meta) as dst:
                    dst.write(fbc)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upsample Spacenet images and rasterize labels.")
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
        help="Scale factor for upsampling images (default: 3)"
    )
    parser.add_argument(
        "--no-fbc",
        action="store_true",
        help="Flag to disable creating the FBC (Footprint, Boundary, Contact) masks"
    )

    args = parser.parse_args()

    print(f"Using data root: {args.data_root}")
    print(f"Scale: {args.scale}x")

    print("\nStep 1: Upscaling images")
    enlarge_images(args.data_root, scale=args.scale)

    print("\nStep 2: Creating labels")
    create_label(args.data_root, scale=args.scale, create_fbc=not args.no_fbc)

    print("\nPipeline complete")