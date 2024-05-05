import os
import config
import numpy as np
import cv2


def create_sectors(columns: int = 16, rows: int = 9) -> np.array:
    """Splitting image into sectors."""
    sector_w = int(config.IMG_WIDTH / columns)
    sector_h = int(config.IMG_WIDTH / rows)
    sectors_arrays = []
    for c in range(columns):
        for r in range(rows):
            bbox = [c * sector_w, r * sector_h, (c + 1) * sector_w, (r + 1) * sector_h]
            sectors_arrays.append(np.expand_dims(bbox, axis=0))
    return np.array(sectors_arrays)


def calculate_image_occupancy(file: str) -> float:
    """Calculation the proportion of non-white pixels in image."""
    img = cv2.imread(os.path.join(config.DATASET_PATH, file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nonwhite_pix = np.sum(img != 255)
    occupancy = nonwhite_pix / config.IMG_AREA
    return occupancy


def box_area(boxes: np.array) -> np.array:
    """Calculation area of the input bboxes."""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def get_mean_center_distance(bboxes: np.array) -> float:
    """Calculation of the average distance between the center of the image and all centers of the input bboxes."""
    img_center = (config.IMG_WIDTH // 2, config.IMG_HEIGHT // 2)
    centers_x = bboxes[:, 0] + (bboxes[:, 2] - bboxes[:, 0]) / 2
    centers_y = bboxes[:, 1] + (bboxes[:, 3] - bboxes[:, 1]) / 2
    mean_distance = np.mean((((centers_x - img_center[0]) ** 2) + ((centers_y - img_center[1]) ** 2)) ** 0.5)
    return mean_distance


def box_intersetion(boxes1: np.array, boxes2: np.array) -> float:
    """Calculation of bboxes intersection."""
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = np.clip(rb-lt, a_min=0, a_max=None)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    return intersection


def box_union(boxes1: np.array, boxes2: np.array) -> float:
    """Calculation of bboxes union."""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = area1[:, None] + area2 - box_intersetion(boxes1, boxes2)
    return union
