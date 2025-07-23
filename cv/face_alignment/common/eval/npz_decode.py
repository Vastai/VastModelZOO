# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
'@Author :        melodylu
'@Email :   algorithm@vastaitech.com
'@Time  :     2025/07/23 17:32:48
'''

import argparse
import numpy as np
import glob
import torch
import math
import os
import json

from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.integrate import simps

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        import argparse
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_preds(scores):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds

def get_transform(center, scale, output_size, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(output_size[1]) / h
    t[1, 1] = float(output_size[0]) / h
    t[0, 2] = output_size[1] * (-float(center[0]) / h + .5)
    t[1, 2] = output_size[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -output_size[1]/2
        t_mat[1, 2] = -output_size[0]/2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t

def transform_pixel(pt, center, scale, output_size, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, output_size, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1

def transform_preds(coords, center, scale, output_size):

    for p in range(coords.size(0)):
        coords[p, 0:2] = torch.tensor(transform_pixel(coords[p, 0:2], center, scale, output_size, 1, 0))
    return coords

def decode_preds(output, center, scale, res):
    coords = get_preds(output)  # float type

    coords = coords.cpu()
    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if (px > 1) and (px < res[0]) and (py > 1) and (py < res[1]):
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords += 0.5
    preds = coords.clone()

    # Transform back
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds

def compute_nme(preds, meta):

    targets = meta['pts']
    preds = preds.numpy()
    # target = targets.numpy()
    target = np.array(targets)

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interocular = meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)

    return rmse

def load_gt(gt_json, ):
    with open(gt_json, 'r') as loadf:
        meta_dict = json.load(loadf)
    return meta_dict

def main(args):
    npz_files = glob.glob(args.result + "/*.npz")
    npz_files.sort()
    img_list = []
    # nme_list = []

    if not args.debug:
        with open(args.npz_txt, 'r') as fr:
            for line in fr:
                img_list.append(os.path.basename(line.strip()).split('.')[0])
    gt_landmark_dict = load_gt(args.gt)

    # if args.show_image:
    #     import shutil
    #     if os.path.exists('show_images'):
    #         shutil.rmtree('show_images')
    #     os.makedirs('show_images')

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0

    for index, file in enumerate(tqdm(npz_files)):
        ## pred landmark
        landmarks = np.load(file, allow_pickle=True)["output_0"]
        landmarks = torch.Tensor(landmarks)

        ## landmark_gt
        if args.debug:
            meta = gt_landmark_dict[os.path.basename(file).split('.')[0]]
        else:
            meta = gt_landmark_dict[img_list[index]]

        preds = decode_preds(landmarks, meta['center'], meta['scale'], [64, 64])
        nme_temp = compute_nme(preds, meta)

        ## draw keypoints
        # if args.show_image:
        #     import cv2
        #     img = cv2.imread("../data/wflw/images/29--Students_Schoolkids/29_Students_Schoolkids_Students_Schoolkids_29_804.jpg")
        #     for i in range(0, 98):
        #         cv2.circle(img, (int(preds[i][0]), int(preds[i][1])), 2, (0, 0, 255), -1)
        #     cv2.imwrite('result_vacc_fp16.jpg', img)
        #     exit()


        failure_008 = (nme_temp > 0.08).sum()
        failure_010 = (nme_temp > 0.10).sum()
        count_failure_008 += failure_008
        count_failure_010 += failure_010

        nme_batch_sum += np.sum(nme_temp)
        nme_count = nme_count + preds.size(0)

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    print('nme: {:.4f}'.format(nme))
    print('failure_rate_008: {:}'.format(failure_008_rate))
    print('failure_rate_010: {:}'.format(failure_010_rate))

    # torch
    # nme: 0.0461
    # failure_rate_008: 0.0824
    # failure_rate_010: 0.0324

    # runmodel
    # nme: 0.0498
    # failure_rate_008: 0.1044
    # failure_rate_010: 0.0452
    


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--result',
                        default="npz_output",
                        type=str)
    parser.add_argument('--npz-txt',
                        default='npz_datalist.txt',
                        type=str)
    parser.add_argument('--gt',
                        default="./wflw_meta.json",
                        type=str)
    parser.add_argument('--show_image', action='store_true')
    parser.add_argument('--images', default='', type=str)
    parser.add_argument('--debug', default=False, type=str2bool)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
