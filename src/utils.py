import os
from io import BytesIO
from typing import Tuple
import string
import random
from owslib.wms import WebMapService
from pyproj import Transformer, CRS
import rasterio
import numpy as np
from PIL import Image


def save_numpy_as_geotiff(array, template_file, output_file, count=1):
    # Read the template file
    with rasterio.open(template_file) as src:
        template_profile = src.profile
        # template_transform = src.transform
        # template_crs = src.crs

    # Update the template profile with the array data
    template_profile.update(dtype=np.float32, count=count)

    # Create the output GeoTIFF file
    with rasterio.open(output_file, "w", **template_profile) as dst:
        dst.write(array, 1)


def format_float(f):
    return "%.6f" % (float(f),)


def shape_to_table_row(row):
    bounds = row.bounds.values[0]
    if row.geom_type.values[0] == "Point":
        return {
            "x_min": format_float(bounds[0]),
            "y_min": format_float(bounds[1]),
            "id": str(row["_leaflet_id"].values[0]),
        }
    return {
        "x_min": format_float(bounds[0]),
        "y_min": format_float(bounds[1]),
        "x_max": format_float(bounds[2]),
        "y_max": format_float(bounds[3]),
        "id": str(row["_leaflet_id"].values[0]),
    }


def bounds_to_table_row(polygon):
    bounds = polygon.bounds
    return {
        "x_min": format_float(bounds[0]),
        "y_min": format_float(bounds[1]),
        "x_max": format_float(bounds[2]),
        "y_max": format_float(bounds[3]),
        "id": "bbox",
    }


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def download_from_wms(
    wms_url: str,
    bbox: Tuple,
    layer: str,
    image_format: str,
    output_dir: str,
    resolution: int,
):
    wms = WebMapService(wms_url)
    # Specify the desired bounding box
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    xmin_t, ymin_t = transformer.transform(xmin, ymin)  # pylint: disable=E0633
    xmax_t, ymax_t = transformer.transform(xmax, ymax)  # pylint: disable=E0633
    width = int((xmax_t - xmin_t) / resolution)
    height = int((ymax_t - ymin_t) / resolution)

    # Request the image from the WMS
    image = wms.getmap(
        layers=[layer],
        srs="EPSG:4326",
        bbox=bbox,
        size=(width, height),
        format=image_format,
    )
    output_filename = id_generator(size=8)
    output_filepath = os.path.join(output_dir, output_filename + ".tif")
    img = np.array(Image.open(BytesIO(image.read())))
    pixel_size_x = (xmax - xmin) / width
    pixel_size_y = (ymax - ymin) / height
    transform = rasterio.transform.from_origin(xmin, ymax, pixel_size_x, pixel_size_y)
    crs_to = CRS.from_epsg(4326)
    with rasterio.open(
        output_filepath,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=img.dtype,
        crs=crs_to,
        transform=transform,
    ) as dst:
        dst.write(np.moveaxis(img, 2, 0))  # Assuming img is a 3-band RGB image

    return output_filepath


def geographic_to_pixel(
    bbox_geo: np.array,
    image_width: int,
    image_height: int,
    min_latitude: float,
    max_latitude: float,
    min_longitude: float,
    max_longitude: float,
) -> np.array:
    # Calculate the conversion factors
    lat_range = max_latitude - min_latitude
    lon_range = max_longitude - min_longitude
    # lat_factor = image_height / lat_range
    # lon_factor = image_width / lon_range

    # Convert the bounding box coordinates to pixel coordinates
    x_min = ((bbox_geo[:, 0] - min_longitude) / lon_range * image_width).astype(int)
    y_min = ((max_latitude - bbox_geo[:, 3]) / lat_range * image_height).astype(int)
    x_max = ((bbox_geo[:, 2] - min_longitude) / lon_range * image_width).astype(int)
    y_max = ((max_latitude - bbox_geo[:, 1]) / lat_range * image_height).astype(int)

    # Create the pixel bounding box array
    pixel_bbox = np.column_stack((x_min, y_min, x_max, y_max))

    return pixel_bbox


# if __name__ == "__main__":
#     from config import *

#     download_from_wms(
#         WMS_URL,
#         (1.25, 43.5, 1.5, 43.75),
#         "s2cloudless-2020",
#         "image/jpeg",
#         "/Users/syam/Documents/code/dino-sam/src/",
#         10,
#     )
