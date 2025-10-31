# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import cv2
import glob
import torch
import argparse
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm
import argparse
import glob
import os
import cv2 as cv
import torch
from skimage.morphology import skeletonize

def fill_hole(input_mask) -> np.array:
    """Fill holes in matrix.

        Input:
            [[0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 1, 0, 0, 0, 1, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0]]
        Output:
            [[0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0]]

    Args:
        input_mask (ArrayLike): The input mask.

    Returns:
        np.array: The output mask that has been filled.
    """
    input_mask = np.array(input_mask)
    h, w = input_mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:h + 1, 1:w + 1] = input_mask.copy()

    mask = np.zeros((h + 4, w + 4), np.uint8)

    cv.floodFill(canvas, mask, (0, 0), 1)
    canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool_)

    return ~canvas | input_mask

def _centralize(points_yx: np.ndarray,
                normal_cos: torch.Tensor,
                normal_sin: torch.Tensor,
                radius: torch.Tensor,
                contour_mask: np.ndarray,
                step_ratio: float = 0.03) -> np.ndarray:
    """Centralize the points.

    Args:
        points_yx (np.array): The points in yx order.
        normal_cos (torch.Tensor): The normal cosine of the points.
        normal_sin (torch.Tensor): The normal sine of the points.
        radius (torch.Tensor): The radius of the points.
        contour_mask (np.array): The contour mask of the points.
        step_ratio (float): The step ratio of the centralization.
            Defaults to 0.03.

    Returns:
        np.ndarray: The centralized points.
    """

    h, w = contour_mask.shape
    top_yx = bot_yx = points_yx
    step_flags = np.ones((len(points_yx), 1), dtype=np.bool_)
    step = step_ratio * radius * np.hstack([normal_cos, normal_sin])
    while np.any(step_flags):
        next_yx = np.array(top_yx + step, dtype=np.int32)
        next_y, next_x = next_yx[:, 0], next_yx[:, 1]
        step_flags = (next_y >= 0) & (next_y < h) & (next_x > 0) & (
            next_x < w) & contour_mask[np.clip(next_y, 0, h - 1),
                                        np.clip(next_x, 0, w - 1)]
        top_yx = top_yx + step_flags.reshape((-1, 1)) * step
    step_flags = np.ones((len(points_yx), 1), dtype=np.bool_)
    while np.any(step_flags):
        next_yx = np.array(bot_yx - step, dtype=np.int32)
        next_y, next_x = next_yx[:, 0], next_yx[:, 1]
        step_flags = (next_y >= 0) & (next_y < h) & (next_x > 0) & (
            next_x < w) & contour_mask[np.clip(next_y, 0, h - 1),
                                        np.clip(next_x, 0, w - 1)]
        bot_yx = bot_yx - step_flags.reshape((-1, 1)) * step
    centers = np.array((top_yx + bot_yx) * 0.5, dtype=np.int32)
    return centers

def _merge_disks(disks: np.ndarray, disk_overlap_thr: float) -> np.ndarray:
    """Merging overlapped disks.

    Args:
        disks (np.ndarray): The predicted disks.
        disk_overlap_thr (float): The radius overlap threshold for merging
            disks.

    Returns:
        np.ndarray: The merged disks.
    """
    xy = disks[:, 0:2]
    radius = disks[:, 2]
    scores = disks[:, 3]
    order = scores.argsort()[::-1]

    merged_disks = []
    while order.size > 0:
        if order.size == 1:
            merged_disks.append(disks[order])
            break
        i = order[0]
        d = norm(xy[i] - xy[order[1:]], axis=1)
        ri = radius[i]
        r = radius[order[1:]]
        d_thr = (ri + r) * disk_overlap_thr

        merge_inds = np.where(d <= d_thr)[0] + 1
        if merge_inds.size > 0:
            merge_order = np.hstack([i, order[merge_inds]])
            merged_disks.append(np.mean(disks[merge_order], axis=0))
        else:
            merged_disks.append(disks[i])

        inds = np.where(d > d_thr)[0] + 1
        order = order[inds]
    merged_disks = np.vstack(merged_disks)

    return merged_disks

def postprocess(pred_results, min_text_region_confidence=0.6, min_center_region_confidence=0.2, min_center_area=30, disk_overlap_thr=0.03, radius_shrink_ratio=1.03):
    pred_results = torch.Tensor(pred_results[0])
    assert pred_results.dim() == 3

    pred_results[:2, :, :] = torch.sigmoid(pred_results[:2, :, :])
    pred_results = pred_results.detach().cpu().numpy()

    pred_text_score = pred_results[0]
    pred_text_mask = pred_text_score > min_text_region_confidence
    pred_center_score = pred_results[1] * pred_text_score
    pred_center_mask = \
        pred_center_score > min_center_region_confidence
    pred_sin = pred_results[2]
    pred_cos = pred_results[3]
    pred_radius = pred_results[4]
    mask_sz = pred_text_mask.shape

    scale = np.sqrt(1.0 / (pred_sin**2 + pred_cos**2 + 1e-8))
    pred_sin = pred_sin * scale
    pred_cos = pred_cos * scale

    pred_center_mask = fill_hole(pred_center_mask).astype(np.uint8)
    center_contours, _ = cv.findContours(pred_center_mask, cv.RETR_TREE,
                                          cv.CHAIN_APPROX_SIMPLE)
    
    result = []

    for contour in center_contours:
        if cv.contourArea(contour) < min_center_area:
            continue
        instance_center_mask = np.zeros(mask_sz, dtype=np.uint8)
        cv.drawContours(instance_center_mask, [contour], -1, 1, -1)
        skeleton = skeletonize(instance_center_mask)
        skeleton_yx = np.argwhere(skeleton > 0)
        y, x = skeleton_yx[:, 0], skeleton_yx[:, 1]
        cos = pred_cos[y, x].reshape((-1, 1))
        sin = pred_sin[y, x].reshape((-1, 1))
        radius = pred_radius[y, x].reshape((-1, 1))

        center_line_yx = _centralize(skeleton_yx, cos, -sin, radius,
                                          instance_center_mask)
        y, x = center_line_yx[:, 0], center_line_yx[:, 1]
        radius = (pred_radius[y, x] * radius_shrink_ratio).reshape(
            (-1, 1))
        score = pred_center_score[y, x].reshape((-1, 1))
        instance_disks = np.hstack(
            [np.fliplr(center_line_yx), radius, score])
        instance_disks = _merge_disks(instance_disks,
                                           disk_overlap_thr)

        instance_mask = np.zeros(mask_sz, dtype=np.uint8)
        for x, y, radius, score in instance_disks:
            if radius > 1:
                cv.circle(instance_mask, (int(x), int(y)), int(radius), 1,
                           -1)
        contours, _ = cv.findContours(instance_mask, cv.RETR_TREE,
                                       cv.CHAIN_APPROX_SIMPLE)

        score = np.sum(instance_mask * pred_text_score) / (
            np.sum(instance_mask) + 1e-8)
        if (len(contours) > 0 and cv.contourArea(contours[0]) > 0
                and contours[0].size > 8):
            polygon = contours[0].flatten().tolist()
        
        if score > 0.5:
            result.append(polygon)
    return result


def rescale_polygon(polygon,
                    scale_factor,
                    mode: str = 'mul') -> np.ndarray:
    """Rescale a polygon according to scale_factor.

    The behavior is different depending on the mode. When mode is 'mul', the
    coordinates will be multiplied by scale_factor, which is usually used in
    preprocessing transforms such as :func:`Resize`.
    The coordinates will be divided by scale_factor if mode is 'div'. It can be
    used in postprocessors to recover the polygon in the original
    image size.

    Args:
        polygon (ArrayLike): A polygon. In any form can be converted
            to an 1-D numpy array. E.g. list[float], np.ndarray,
            or torch.Tensor. Polygon is written in
            [x1, y1, x2, y2, ...].
        scale_factor (tuple(int, int)): (w_scale, h_scale).
        model (str): Rescale mode. Can be 'mul' or 'div'. Defaults to 'mul'.

    Returns:
        np.ndarray: Rescaled polygon.
    """
    assert len(polygon) % 2 == 0
    assert mode in ['mul', 'div']
    polygon = np.array(polygon, dtype=np.float32)
    poly_shape = polygon.shape
    reshape_polygon = polygon.reshape(-1, 2)
    scale_factor = np.array(scale_factor, dtype=float)
    if mode == 'div':
        scale_factor = 1 / scale_factor
    polygon = (reshape_polygon * scale_factor[None]).reshape(poly_shape)
    return polygon

if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="VAMP EVAL")
    parse.add_argument("--src_dir", type=str, default="/path/to/datasets/ocr/ctw1500/test_images/", help="hr gt image folder")
    parse.add_argument("--input_npz_path", type=str, default="/path/to/datasets/ocr/ctw1500/npz_datalist.txt", help="npz text file")
    parse.add_argument("--out_npz_dir", type=str, default="../../source_code/npz_output/", help="vamp output folder")
    parse.add_argument("--input_shape", nargs='+', type=int, default=[512, 512], help="vamp input shape")

    args = parse.parse_args()
    print(args)

    output_npz_list = glob.glob(os.path.join(args.out_npz_dir, "*.npz"))
    output_npz_list.sort()
    
    with open(args.input_npz_path, 'r') as f:
        file_lines = f.readlines()
        for i, line in enumerate(tqdm(file_lines)):
            npz_name = os.path.basename(line.strip('\n'))
            file_name = npz_name.replace(".npz", ".png")

            # src image
            ori_image  = cv2.imread(os.path.join(args.src_dir, file_name.replace(".png", ".jpg")))

            # load npy
            npz_file = output_npz_list[i]
            heatmap = np.load(npz_file, allow_pickle=True)["output_0"].astype(np.float32)
            
            h, w, _  = ori_image.shape
            scale = [w / heatmap.shape[3], h / heatmap.shape[2]]
            result = postprocess(heatmap)
            # print(results)
            f = open('./vamp_pred/' + file_name.split('.')[0] + '.txt', 'w')
            for i in range(len(result)):
                box = result[i]
                box = rescale_polygon(box, scale)
                b = [str(bb) for bb in box]
                f.writelines(' '.join(b) + '\n')
            
            f.close()

            
