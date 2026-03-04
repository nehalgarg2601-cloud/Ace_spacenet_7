import os
import numpy as np
import torch
from tqdm import tqdm
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation, binary_erosion


def get_device():
    if torch.backends.mps.is_available():
        print("🚀 Using MPS")
        return torch.device("mps")
    else:
        print("💻 Using CPU")
        return torch.device("cpu")


def generate_fbc_mask(image_path, geojson_path, output_path):
    # Read reference image metadata
    with rasterio.open(image_path) as src:
        meta = src.meta.copy()
        height = src.height
        width = src.width
        transform = src.transform
        image_crs = src.crs

    # Safely load GeoJSON and drop null geometries
    try:
        gdf = gpd.read_file(geojson_path)
    except Exception as e:
        print(f"⚠️  Failed to read geojson {geojson_path}: {e}")
        gdf = gpd.GeoDataFrame(geometry=[])

    if gdf.empty or "geometry" not in gdf or gdf.geometry.isna().all():
        # No buildings: all-zero footprint
        footprint = np.zeros((height, width), dtype=np.uint8)
    else:
        gdf = gdf[gdf.geometry.notnull()]

        # Ensure CRS match (SpaceNet7 uses EPSG:3857 for both image and labels)
        if gdf.crs is not None and image_crs is not None and gdf.crs != image_crs:
            gdf = gdf.to_crs(image_crs)

        shapes = [
            (geom, 1)
            for geom in gdf.geometry
            if geom is not None and not geom.is_empty
        ]

        if len(shapes) == 0:
            footprint = np.zeros((height, width), dtype=np.uint8)
        else:
            footprint = rasterize(
                shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.uint8,
            )

    # -------- Boundary --------
    # Thin outline around buildings (rough analogue to solaris boundary channel)
    dilated = binary_dilation(footprint, iterations=1)
    eroded = binary_erosion(footprint, iterations=1)
    boundary = (dilated ^ eroded).astype(np.uint8)

    # -------- Contact --------
    # Approximate contact regions near building footprints
    expanded = binary_dilation(footprint, iterations=3)
    contact = np.logical_and(expanded, np.logical_not(footprint)).astype(np.uint8)

    # Stack channels (C, H, W)
    fbc = np.stack([footprint, boundary, contact], axis=0)

    meta.update(
        {
            "count": 3,
            "dtype": "uint8",
            "nodata": 0,
        }
    )

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(fbc)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rasterize GeoJSON labels to FBC masks.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="../data/SN7_buildings_train/train",
        help="Path to the training data root directory containing AOI folders"
    )

    args = parser.parse_args()

    device = get_device()
    train_root = args.data_root

    print(f"Using data root: {train_root}")

    # Use a safely filtered list of AOI folders to avoid .DS_Store
    aoi_folders = [
        d for d in sorted(os.listdir(train_root))
        if os.path.isdir(os.path.join(train_root, d)) and not d.startswith(".")
    ]

    for aoi in tqdm(aoi_folders, desc="Processing AOIs"):

        aoi_path = os.path.join(train_root, aoi)

        image_dir = os.path.join(aoi_path, "images")
        label_dir = os.path.join(aoi_path, "labels_match")

        # 🔥 New raster folder
        raster_dir = os.path.join(aoi_path, "labels_rasterized")
        os.makedirs(raster_dir, exist_ok=True)

        if not os.path.exists(image_dir):
            continue

        image_files = [f for f in os.listdir(image_dir) if f.endswith(".tif")]

        for img_file in tqdm(image_files, desc=aoi, leave=False):

            name_root = os.path.splitext(img_file)[0]

            image_path = os.path.join(image_dir, img_file)
            # Match SpaceNet7 naming convention: *_Buildings.geojson
            json_path = os.path.join(label_dir, name_root + "_Buildings.geojson")

            if not os.path.exists(json_path):
                continue

            output_path = os.path.join(raster_dir, name_root + "_fbc.tif")

            generate_fbc_mask(image_path, json_path, output_path)

    print("✅ FBC masks generated.")