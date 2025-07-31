# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import numpy as np
import mmcv
import math
import cv2

import os
import torch 

from hrnet_postprocess import forward_test
from data_process import data_process_old
from mmcv.visualization.image import imshow
from mmcv.image import imwrite
from utils.coco import dataset_info as coco_datainfo
from utils.dataset_info import DatasetInfo

import argparse

MODEL_H = 512
MODEL_W = 512
dataset_info_new = DatasetInfo(coco_datainfo)


def oks_iou(g, d, a_g, a_d, sigmas=None, vis_thr=None):
    """Calculate oks ious.

    Args:
        g: Ground truth keypoints.
        d: Detected keypoints.
        a_g: Area of the ground truth object.
        a_d: Area of the detected object.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.

    Returns:
        list: The oks ious.
    """
    if sigmas is None:
        sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
            .87, .87, .89, .89
        ]) / 10.0
    vars = (sigmas * 2)**2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros(len(d), dtype=np.float32)
    for n_d in range(0, len(d)):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx**2 + dy**2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if vis_thr is not None:
            ind = list(vg > vis_thr) and list(vd > vis_thr)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / len(e) if len(e) != 0 else 0.0
    return ious

def oks_nms(kpts_db, thr, sigmas=None, vis_thr=None, score_per_joint=False):
    """OKS NMS implementations.

    Args:
        kpts_db: keypoints.
        thr: Retain overlap < thr.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.
        score_per_joint: the input scores (in kpts_db) are per joint scores

    Returns:
        np.ndarray: indexes to keep.
    """
    if len(kpts_db) == 0:
        return []

    if score_per_joint:
        scores = np.array([k['score'].mean() for k in kpts_db])
    else:
        scores = np.array([k['score'] for k in kpts_db])

    kpts = np.array([k['keypoints'].flatten() for k in kpts_db])
    areas = np.array([k['area'] for k in kpts_db])

    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]],
                          sigmas, vis_thr)

        inds = np.where(oks_ovr <= thr)[0]
        order = order[inds + 1]

    keep = np.array(keep)

    return keep

def image_affine(width, height, x, y):
    r = min(MODEL_W / width, MODEL_H / height)
    unpad_w = int(round(width * r))
    unpad_h = int(round(height * r))
    dw = MODEL_W - unpad_w
    dh = MODEL_H - unpad_h
    dw /= 2
    dh /= 2
    w = width / unpad_w
    h = height / unpad_h
    
    xx = (x - dw) * w
    yy = (y - dh) * h

    return xx, yy


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

    

def imshow_keypoints(img,
                     pose_result,
                     skeleton=None,
                     kpt_score_thr=0.3,
                     pose_kpt_color=None,
                     pose_link_color=None,
                     radius=4,
                     thickness=1,
                     show_keypoint_weight=False):
    """Draw keypoints and links on an image.

    Args:
            img (str or Tensor): The image to draw poses on. If an image array
                is given, id will be modified in-place.
            pose_result (list[kpts]): The poses to draw. Each element kpts is
                a set of K keypoints as an Kx3 numpy.ndarray, where each
                keypoint is represented as x, y, score.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints. If None,
                the keypoint will not be drawn.
            pose_link_color (np.array[Mx3]): Color of M links. If None, the
                links will not be drawn.
            thickness (int): Thickness of lines.
    """

    img = mmcv.imread(img)
    # img, _, _ = letterbox(img, (512, 512))

    # img = mmcv.imresize(img, (512,512))
    img_h, img_w, _ = img.shape

    for kpts in pose_result:

        kpts = np.array(kpts, copy=False)

        # draw each point on image
        if pose_kpt_color is not None:
            assert len(pose_kpt_color) == len(kpts)

            for kid, kpt in enumerate(kpts):
                # xx , yy = image_affine(img_w, img_h, int(kpt[0]), int(kpt[1]))
                x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]
                # x_coord, y_coord, kpt_score = xx, yy, kpt[2]

                '''if kpt_score < kpt_score_thr or pose_kpt_color[kid] is None:
                    # skip the point that should not be drawn
                    continue'''

                color = tuple(int(c) for c in pose_kpt_color[kid])
                if show_keypoint_weight:
                    img_copy = img.copy()
                    cv2.circle(img_copy, (int(x_coord), int(y_coord)), radius,
                               color, -1)
                    transparency = max(0, min(1, kpt_score))
                    cv2.addWeighted(
                        img_copy,
                        transparency,
                        img,
                        1 - transparency,
                        0,
                        dst=img)
                else:
                    cv2.circle(img, (int(x_coord), int(y_coord)), radius,
                               color, 1)

        # draw links
        if skeleton is not None and pose_link_color is not None:
            assert len(pose_link_color) == len(skeleton)

            for sk_id, sk in enumerate(skeleton):
                # pos1_x, pos1_y = image_affine(img_w, img_h, kpts[sk[0], 0], kpts[sk[0], 1])
                # pos2_x, pos2_y = image_affine(img_w, img_h, kpts[sk[1], 0], kpts[sk[1], 1])

                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
                # pos1 = (int(pos1_x), int(pos1_y))
                # pos2 = (int(pos2_x), int(pos2_y))
                if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                        or pos1[1] >= img_h or pos2[0] <= 0 or pos2[0] >= img_w
                        or pos2[1] <= 0 or pos2[1] >= img_h
                        or kpts[sk[0], 2] < kpt_score_thr
                        or kpts[sk[1], 2] < kpt_score_thr
                        or pose_link_color[sk_id] is None):
                    # skip the link that should not be drawn
                    continue
                color = tuple(int(c) for c in pose_link_color[sk_id])
                if show_keypoint_weight:
                    img_copy = img.copy()
                    X = (pos1[0], pos2[0])
                    Y = (pos1[1], pos2[1])
                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                    angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                    stickwidth = 2
                    polygon = cv2.ellipse2Poly(
                        (int(mX), int(mY)), (int(length / 2), int(stickwidth)),
                        int(angle), 0, 360, 1)
                    cv2.fillConvexPoly(img_copy, polygon, color)
                    transparency = max(
                        0, min(1, 0.5 * (kpts[sk[0], 2] + kpts[sk[1], 2])))
                    cv2.addWeighted(
                        img_copy,
                        transparency,
                        img,
                        1 - transparency,
                        0,
                        dst=img)
                else:
                    cv2.line(img, pos1, pos2, color, thickness=thickness)

    return img

def show_result(img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color=None,
                    pose_kpt_color=None,
                    pose_link_color=None,
                    radius=4,
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    show_keypoint_weight=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            skeleton (list[list]): The connection of keypoints.
                skeleton is 0-based indexing.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M links.
                If None, do not draw links.
            radius (int): Radius of circles.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            show_keypoint_weight (bool): Whether to change the transparency
                using the predicted confidence scores of keypoints.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized image only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        img_h, img_w, _ = img.shape

        pose_result = []
        for res in result:
            pose_result.append(res['keypoints'])

        img = imshow_keypoints(img, pose_result, skeleton, kpt_score_thr,
                         pose_kpt_color, pose_link_color, radius, thickness)

        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img

def vis_pose_result(
                    img,
                    result,
                    radius=2,
                    thickness=2,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    dataset='TopDownCocoDataset',
                    dataset_info=None,
                    show=False,
                    out_file=None):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str | np.ndarray): Image filename or loaded image.
        result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
        radius (int): Radius of circles.
        thickness (int): Thickness of lines.
        kpt_score_thr (float): The threshold to visualize the keypoints.
        skeleton (list[tuple()]): Default None.
        show (bool):  Whether to show the image. Default True.
        out_file (str|None): The filename of the output visualization image.
    """

    # get dataset info
    # if (dataset_info is None and hasattr(model, 'cfg')
    #         and 'dataset_info' in model.cfg):
    # dataset_info = DatasetInfo(model.cfg.dataset_info)

    if dataset_info_new is not None:
        skeleton = dataset_info_new.skeleton
        pose_kpt_color = dataset_info_new.pose_kpt_color
        pose_link_color = dataset_info_new.pose_link_color
    else:
        # warnings.warn(
        #     'dataset is deprecated.'
        #     'Please set `dataset_info` in the config.'
        #     'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
        #     DeprecationWarning)
        # TODO: These will be removed in the later versions.
        palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255],
                            [255, 0, 0], [255, 255, 255]])

        if dataset in ('TopDownCocoDataset', 'BottomUpCocoDataset',
                       'TopDownOCHumanDataset', 'AnimalMacaqueDataset'):
            # show the results
            skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                        [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                        [3, 5], [4, 6]]

            pose_link_color = palette[[
                0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
            ]]
            pose_kpt_color = palette[[
                16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
            ]]

        else:
            NotImplementedError()

    # if hasattr(model, 'module'):
    #     model = model.module

    img = show_result(
        img,
        result,
        skeleton,
        radius=radius,
        thickness=thickness,
        pose_kpt_color=pose_kpt_color,
        pose_link_color=pose_link_color,
        kpt_score_thr=kpt_score_thr,
        bbox_color=bbox_color,
        show=show,
        out_file=out_file)

    return img

def main(args):
    img_or_path = "configs/000000000785.jpg"


    '''toutput = np.load("./Desktop/valid_hrnet_output.npz",allow_pickle=True)
    output = []
    output.append(torch.Tensor(toutput["input_0"]))'''

    # toutput = np.load("./test_hrnet_output.npz",allow_pickle=True)
    toutput = np.load("./Desktop/hrnet_out/000000000785.npz",allow_pickle=True)

    output = []
    output.append(torch.Tensor(toutput["output_0"]))

    toutput_flip = np.load("./Desktop/valid_flip_out.npz")
    output_flip = []
    output_flip.append(torch.Tensor(toutput_flip["input_0"]))
    img_metas = data_process_old(img_or_path)
    ### 500 333
    img_metas[0]['base_size'] = (640, 425)
    # img_metas[0]['center'] = (256, 256)
    img_metas['scale'] = np.array([3.2, 3.2])
    # img_metas['center'] = 
    img_metas['center'] = (320,212)
    # img_metas[0]['scale'] = np.array([2.22, 1.665])
    # print(output_flip[0].shape)
    # print(output[0].shape)
    result = forward_test(outputs=output,
        # img=img,
        outputs_flipped = output_flip,
        img_metas=img_metas,
        return_heatmap=False,
        flip_test=args.flip_test
    )

    pose_results = []
    for idx, pred in enumerate(result['preds']):
        area = (np.max(pred[:, 0]) - np.min(pred[:, 0])) * (
            np.max(pred[:, 1]) - np.min(pred[:, 1]))
        pose_results.append({
            'keypoints': pred[:, :3],
            'score': result['scores'][idx],
            'area': area,
        })
    # pose nms
    # score_per_joint = cfg.model.test_cfg.get('score_per_joint', False)
    score_per_joint = False
    keep = oks_nms(
        pose_results,
        0.9,
        sigmas,
        score_per_joint=score_per_joint)
    pose_results = [pose_results[_keep] for _keep in keep]

    out_img_root = "vis_results"
    if not os.path.exists(out_img_root):
        os.makedirs(out_img_root, exist_ok=True)
    out_file = os.path.join(
        out_img_root,
        f'vis_{os.path.splitext(os.path.basename(img_or_path))[0]}.jpg')
    vis_pose_result(
        img_or_path,
        pose_results,
        radius= 3,
        thickness=1,
        dataset="BottomUpCocoDataset",
        dataset_info=dataset_info_new,
        kpt_score_thr=0.3,
        show=False,
        out_file=out_file)
        # rank, _ = get_dist_info()
        


def parse_args():
    parser = argparse.ArgumentParser(description="Convert front model to vacc.")
    parser.add_argument('--flip_test',action="store_true" )
    parser.add_argument('--data_root',default= "./Documents/project/det_data/coco")
    parser.add_argument("--output_data",default= "./Desktop/hrnet_out")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)