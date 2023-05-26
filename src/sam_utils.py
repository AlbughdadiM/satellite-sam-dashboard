import os
from typing import List, Tuple, Union
import torch
from PIL import Image
import tifffile as tiff
from matplotlib import pyplot as plt
import numpy as np
from sam.sam.predictor import SamPredictor
from sam.sam.build_sam import sam_model_registry
from sam.sam.automatic_mask_generator import SamAutomaticMaskGenerator
from utils import (
    geographic_to_pixel_bbox,
    geographic_to_pixel_point,
    save_numpy_as_geotiff,
)
from config import TORCH_DEVICE, WEIGHTS_PATH


def create_mask(img, anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    mask = np.zeros((img.shape[:-1]), dtype=np.float32)
    for idx, ann in enumerate(sorted_anns):
        m = ann["segmentation"].astype(np.uint8)
        mask[m == 1] = idx + 1
    return mask


def build_model(model_filename: str):
    device = torch.device(TORCH_DEVICE)  # pylint: disable=E1101
    model_type = "_".join(model_filename.split("_")[1:3])
    checkpoint_path = os.path.join(WEIGHTS_PATH, model_filename)
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    return sam


def generate_automatic_mask(
    image_path: str,
    model_filename: str,
    pred_iou_thresh: float,
    stability_score_thresh: float,
) -> Tuple[str, str]:
    sam = build_model(model_filename)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        stability_score_thresh=stability_score_thresh,
        pred_iou_thresh=pred_iou_thresh,
    )
    img = tiff.imread(image_path)
    annotation = mask_generator.generate(img)
    one_band_mask = create_mask(img, annotation)
    normalized_img = (one_band_mask - np.min(one_band_mask)) / (
        np.max(one_band_mask) - np.min(one_band_mask)
    )
    colored_img = plt.cm.tab20b(normalized_img)[:, :, :3]  # pylint: disable=E1101
    scaled_img = (colored_img * 255).astype(np.uint8)
    # transparent_pixels = one_band_mask == 1
    # scaled_img = np.concatenate(
    #     [
    #         scaled_img,
    #         255
    #         * np.ones(
    #             (scaled_img.shape[0], scaled_img.shape[1], 1), dtype=scaled_img.dtype
    #         ),
    #     ],
    #     axis=2,
    # )
    # scaled_img[transparent_pixels, 3] = 0
    pil_image = Image.fromarray(scaled_img)
    # one_band_mask = one_band_mask[..., np.newaxis]
    mask_output_path = image_path.split(".")[0] + "_automatic_mask.tif"
    mask_png_path = mask_output_path.split(".")[0] + ".png"
    save_numpy_as_geotiff(one_band_mask, image_path, mask_output_path)
    pil_image.save(mask_png_path)
    return mask_output_path, mask_png_path


def sam_prompt_bbox(
    image_path: str,
    bboxes_geo: Union[np.array, None],
    points_geo: Union[np.array, None],
    labels: Union[np.array, None],
    model_filename: str,
    roi_bbox: List[float],
) -> Tuple[str, str]:
    sam = build_model(model_filename)
    predictor = SamPredictor(sam)
    img = tiff.imread(image_path)

    if bboxes_geo is not None:
        bboxes_geo = geographic_to_pixel_bbox(
            bboxes_geo,
            img.shape[1],
            img.shape[0],
            roi_bbox[1],
            roi_bbox[3],
            roi_bbox[0],
            roi_bbox[2],
        )
        bboxes_geo = torch.tensor(bboxes_geo, device=predictor.device)
        bboxes_geo = predictor.transform.apply_boxes_torch(bboxes_geo, img.shape[:2])
    if points_geo is not None:
        points_geo = geographic_to_pixel_point(
            points_geo,
            img.shape[1],
            img.shape[0],
            roi_bbox[1],
            roi_bbox[3],
            roi_bbox[0],
            roi_bbox[2],
        )
        points_geo = torch.tensor(points_geo, device=predictor.device)
        points_geo = predictor.transform.apply_coords_torch(points_geo, img.shape[:2])
        labels = torch.tensor(labels, device=predictor.device)
    if bboxes_geo is not None and points_geo is not None:
        intersections = (
            (points_geo >= bboxes_geo[:, None, :2])
            & (points_geo <= bboxes_geo[:, None, 2:])
        ).all(2)
        list_points = []
        list_labels = []
        for i in range(bboxes_geo.shape[0]):
            tmp_points = points_geo[intersections[i, :]]
            tmp_labels = labels[intersections[i, :]]
            if i == 0:
                min_point_box = tmp_points.shape[0]
            if tmp_points.shape[0] < min_point_box:
                min_point_box = tmp_points.shape[0]
            list_points.append(tmp_points)
            list_labels.append(tmp_labels)
        points_geo = np.zeros((bboxes_geo.shape[0], min_point_box, 2), dtype=np.float32)
        labels = np.zeros((bboxes_geo.shape[0], min_point_box), dtype=np.float32)
        for i in range(bboxes_geo.shape[0]):
            points_geo[i, :, :] = list_points[i][0:min_point_box]
            labels[i, :] = list_labels[i][0:min_point_box]
        points_geo = torch.tensor(points_geo, device=sam.device)
        labels = torch.tensor(labels, device=sam.device)

    if bboxes_geo is not None or points_geo is not None:
        predictor.set_image(img)
        masks, _, _ = predictor.predict_torch(
            point_coords=points_geo,
            point_labels=labels,
            boxes=bboxes_geo,
            multimask_output=False,
        )
        masks = np.array(masks)
        one_band_mask_1 = np.argmax(masks, axis=0)[0, ...]
        one_band_mask_2 = masks.sum(axis=0)[0, ...]
        one_band_mask = one_band_mask_1 + one_band_mask_2
        normalized_img = (one_band_mask - np.min(one_band_mask)) / (
            np.max(one_band_mask) - np.min(one_band_mask)
        )
        colored_img = plt.cm.tab20b(normalized_img)[:, :, :3]  # pylint: disable=E1101
        scaled_img = (colored_img * 255).astype(np.uint8)
        transparent_pixels = one_band_mask == 0
        scaled_img = np.concatenate(
            [
                scaled_img,
                255
                * np.ones(
                    (scaled_img.shape[0], scaled_img.shape[1], 1),
                    dtype=scaled_img.dtype,
                ),
            ],
            axis=2,
        )
        scaled_img[transparent_pixels, 3] = 0
        pil_image = Image.fromarray(scaled_img)
        # one_band_mask = one_band_mask[..., np.newaxis]
        mask_output_path = image_path.split(".")[0] + "_bbox_mask.tif"
        mask_png_path = mask_output_path.split(".")[0] + ".png"
        save_numpy_as_geotiff(one_band_mask, image_path, mask_output_path)
        # array2raster(mask_output_path, image_path, one_band_mask, "Byte")
        pil_image.save(mask_png_path)
        return mask_output_path, mask_png_path
    else:
        raise RuntimeError("Something Went Wrong")
