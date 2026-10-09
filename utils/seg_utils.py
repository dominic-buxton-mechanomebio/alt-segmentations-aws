# © 2026 Mechanome Bio. Proprietary and confidential.
# Shared helpers for the MechanoMap alternative segmentation notebooks:
# loading/cropping the sdata, converting between masks and polygons, and saving results.

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import geopandas
import matplotlib.pyplot as plt
import numpy
import shapely
import spatialdata
from matplotlib.patches import Rectangle
from rasterio import features
from rasterio.transform import Affine
from spatialdata.models import ShapesModel
from spatialdata.transformations import Scale

NOTICE = "© 2026 Mechanome Bio. Proprietary and confidential."

# Repo layout
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INPUTS_DIRECTORY = REPOSITORY_ROOT / "inputs"
OUTPUTS_DIRECTORY = REPOSITORY_ROOT / "outputs"
CONFIG_DIRECTORY = REPOSITORY_ROOT / "config_files"
EXTERNAL_DIRECTORY = REPOSITORY_ROOT / "external"  # cloned method repos that aren't pip packages
S3_DATASETS = "s3://mechanome-cellpose-airr-lon-161012475647-eu-west-2/benchmark/_datasets"

# Element names in the sdata (spatialdata-io Xenium defaults)
IMAGE_KEY = "morphology_focus"
TRANSCRIPTS_KEY = "transcripts"
NUCLEI_KEY = "nucleus_boundaries"
CELLS_KEY = "cell_boundaries"
GENE_COLUMN = "feature_name"
DAPI_CHANNEL = "DAPI"
BOUNDARY_CHANNELS = ["ATP1A1/CD45/E-Cadherin", "18S", "AlphaSMA/Vimentin"]

# Xenium morphology image pixel size. The 'global' coordinate system is in these pixels;
# transcripts and shapes are stored in microns with a 1/PIXEL_SIZE_MICRONS scale to 'global'.
PIXEL_SIZE_MICRONS = 0.2125


# ---------- loading and cropping ----------

def zarr_path(dataset_name):
    return INPUTS_DIRECTORY / "zarrs" / f"{dataset_name}.zarr"


def load_sdata(dataset_name):
    return spatialdata.read_zarr(zarr_path(dataset_name))


def image_bounds_microns(sdata):
    # {"x": (min, max), "y": (min, max)} of the morphology image, in microns
    extent_pixels = spatialdata.get_extent(sdata[IMAGE_KEY], coordinate_system="global")
    return {axis: (round(float(extent_pixels[axis][0] * PIXEL_SIZE_MICRONS), 2),
                   round(float(extent_pixels[axis][1] * PIXEL_SIZE_MICRONS), 2))
            for axis in ("x", "y")}


def bounding_box_in_image(sdata, bounding_box_microns):
    # bounding_box_microns = (x_min, y_min, x_max, y_max)
    bounds = image_bounds_microns(sdata)
    x_min, y_min, x_max, y_max = bounding_box_microns
    inside = (bounds["x"][0] <= x_min < x_max <= bounds["x"][1]
              and bounds["y"][0] <= y_min < y_max <= bounds["y"][1])
    if not inside:
        print(f"bounding box not possible for image, image bounds are: {bounds}")
    return inside


def preview_bounding_box(sdata, bounding_box_microns, scale_level=None):
    # Left: whole image with the box drawn. Right: just the box region.
    # Uses a downsampled pyramid level (default: the lowest resolution one) so it renders fast.
    levels = list(sdata[IMAGE_KEY].keys())
    level = levels[-1] if scale_level is None else f"scale{scale_level}"
    image = sdata[IMAGE_KEY][level]["image"].sel(c=DAPI_CHANNEL).values
    bounds = image_bounds_microns(sdata)
    full_extent = (bounds["x"][0], bounds["x"][1], bounds["y"][1], bounds["y"][0])
    contrast_limit = numpy.percentile(image, 99.5)

    if bounding_box_microns is None:
        figure, axis = plt.subplots(figsize=(8, 8))
        axis.imshow(image, cmap="gray", vmax=contrast_limit, extent=full_extent)
        axis.set_xlabel("x (µm)")
        axis.set_ylabel("y (µm)")
        plt.show()
        return
    if not bounding_box_in_image(sdata, bounding_box_microns):
        return

    x_min, y_min, x_max, y_max = bounding_box_microns
    figure, (overview_axis, zoom_axis) = plt.subplots(1, 2, figsize=(16, 8))
    overview_axis.imshow(image, cmap="gray", vmax=contrast_limit, extent=full_extent)
    overview_axis.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                      fill=False, edgecolor="red", linewidth=1.5))

    # microns -> row/column indices of this pyramid level
    microns_per_level_pixel_x = (bounds["x"][1] - bounds["x"][0]) / image.shape[1]
    microns_per_level_pixel_y = (bounds["y"][1] - bounds["y"][0]) / image.shape[0]
    column_start = int((x_min - bounds["x"][0]) / microns_per_level_pixel_x)
    column_end = int(numpy.ceil((x_max - bounds["x"][0]) / microns_per_level_pixel_x))
    row_start = int((y_min - bounds["y"][0]) / microns_per_level_pixel_y)
    row_end = int(numpy.ceil((y_max - bounds["y"][0]) / microns_per_level_pixel_y))
    zoom_axis.imshow(image[row_start:row_end, column_start:column_end], cmap="gray",
                     vmax=contrast_limit, extent=(x_min, x_max, y_max, y_min))

    for axis in (overview_axis, zoom_axis):
        axis.set_xlabel("x (µm)")
        axis.set_ylabel("y (µm)")
    plt.show()


def crop_sdata(sdata, bounding_box_microns):
    # None = whole slide
    if bounding_box_microns is None:
        return sdata
    x_min, y_min, x_max, y_max = numpy.array(bounding_box_microns) / PIXEL_SIZE_MICRONS
    return spatialdata.bounding_box_query(
        sdata, axes=("x", "y"), min_coordinate=[x_min, y_min], max_coordinate=[x_max, y_max],
        target_coordinate_system="global",
    )


# ---------- pulling data out of a (cropped) sdata ----------

def image_origin_pixels(sdata):
    # top-left corner of the (cropped) morphology image, in 'global' pixels
    extent_pixels = spatialdata.get_extent(sdata[IMAGE_KEY], coordinate_system="global")
    return int(round(extent_pixels["x"][0])), int(round(extent_pixels["y"][0]))


def get_image(sdata, channel=DAPI_CHANNEL):
    # full resolution, one channel, numpy (y, x)
    return sdata[IMAGE_KEY]["scale0"]["image"].sel(c=channel).values


def get_transcripts(sdata):
    # pandas table; x/y in microns, plus x_pixels/y_pixels relative to the image origin
    transcripts = sdata[TRANSCRIPTS_KEY].compute()
    origin_x, origin_y = image_origin_pixels(sdata)
    transcripts["x_pixels"] = transcripts["x"] / PIXEL_SIZE_MICRONS - origin_x
    transcripts["y_pixels"] = transcripts["y"] / PIXEL_SIZE_MICRONS - origin_y
    return transcripts


# ---------- masks <-> polygons ----------

def _pixel_to_micron_transform(origin_pixels):
    origin_x, origin_y = origin_pixels
    return Affine(PIXEL_SIZE_MICRONS, 0, origin_x * PIXEL_SIZE_MICRONS,
                  0, PIXEL_SIZE_MICRONS, origin_y * PIXEL_SIZE_MICRONS)


def mask_to_polygons(mask, origin_pixels=(0, 0)):
    # label mask (y, x) -> GeoDataFrame in microns, one polygon per label (largest piece kept)
    mask = mask.astype(numpy.int32)
    records = [(int(label), shapely.geometry.shape(geometry))
               for geometry, label in features.shapes(mask, mask=mask > 0,
                                                       transform=_pixel_to_micron_transform(origin_pixels))]
    polygons = geopandas.GeoDataFrame(records, columns=["label", "geometry"], geometry="geometry")
    polygons["area"] = polygons.area
    polygons = polygons.sort_values("area").drop_duplicates("label", keep="last")
    return polygons.set_index("label").drop(columns="area").sort_index()


def polygons_to_mask(polygons_microns, origin_pixels, shape):
    # GeoDataFrame in microns -> int32 label mask (y, x); labels are 1..n in row order
    shapes_and_labels = zip(polygons_microns.geometry, range(1, len(polygons_microns) + 1))
    return features.rasterize(shapes_and_labels, out_shape=shape, fill=0, dtype="int32",
                              transform=_pixel_to_micron_transform(origin_pixels))


def pixel_polygons_to_microns(polygons_pixels, origin_pixels):
    # polygons in image pixels (relative to origin_pixels) -> microns
    origin_x, origin_y = origin_pixels
    polygons = polygons_pixels.copy()
    polygons.geometry = polygons.geometry.affine_transform(
        [PIXEL_SIZE_MICRONS, 0, 0, PIXEL_SIZE_MICRONS,
         origin_x * PIXEL_SIZE_MICRONS, origin_y * PIXEL_SIZE_MICRONS])
    return polygons


# ---------- saving ----------

def save_segmentation(sdata, polygons_microns, method, dataset_name, bounding_box_microns=None):
    # sdata must be the full, on-disk sdata (not a crop).
    # Writes shapes element '<method>_boundaries' into the zarr, a parquet + metadata json into
    # outputs/<dataset>/, then uploads all of it to S3.
    element_name = f"{method}_boundaries"
    polygons_microns = geopandas.GeoDataFrame(geometry=polygons_microns.geometry.values,
                                              index=polygons_microns.index.astype(str))

    # same convention as cell_boundaries: microns, scaled to 'global' pixels
    shapes = ShapesModel.parse(polygons_microns.copy(), transformations={
        "global": Scale([1 / PIXEL_SIZE_MICRONS, 1 / PIXEL_SIZE_MICRONS], axes=("x", "y"))})
    sdata[element_name] = shapes
    sdata.write_element(element_name, overwrite=True)
    sdata.write_consolidated_metadata()

    output_directory = OUTPUTS_DIRECTORY / dataset_name
    output_directory.mkdir(parents=True, exist_ok=True)
    parquet_path = output_directory / f"{element_name}.parquet"
    metadata_path = output_directory / f"{element_name}_metadata.json"
    polygons_microns.to_parquet(parquet_path)
    metadata = {
        "notice": NOTICE,
        "pipeline": "MechanoMap - alternative segmentation",
        "method": method,
        "dataset": dataset_name,
        "units": "microns",
        "bounding_box_microns": None if bounding_box_microns is None else list(bounding_box_microns),
        "number_of_cells": len(polygons_microns),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    s3_dataset = f"{S3_DATASETS}/{dataset_name}"
    subprocess.run(["aws", "s3", "cp", str(parquet_path), f"{s3_dataset}/{parquet_path.name}"], check=True)
    subprocess.run(["aws", "s3", "cp", str(metadata_path), f"{s3_dataset}/{metadata_path.name}"], check=True)
    # new element + the zarr's (consolidated) metadata files
    subprocess.run(["aws", "s3", "sync", str(zarr_path(dataset_name)),
                    f"{s3_dataset}/zarr/{dataset_name}.zarr",
                    "--exclude", "*", "--include", "zarr.json", "--include", ".zmetadata",
                    "--include", "shapes/*"], check=True)
    print(f"saved {len(polygons_microns)} cells as '{element_name}'")
