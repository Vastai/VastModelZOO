
# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :    zpwen
@Email  :    algorithm@vastaitech.com
@Time   :    2025/04/24 16:17:38
'''

import os
import math
import cv2
import copy
import glob
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image


def draw_matting(image, mask):
    """
    image (np.uint8) shape (H,W,3)
    mask  (np.float32) range from 0 to 1, shape (H,W)
    """
    # mask = 255*(1.0-mask)
    mask = np.expand_dims(mask, axis=2)
    mask = np.tile(mask, (1,1,3))
    mask = mask.astype(np.uint8)
    image_alpha = cv2.addWeighted(image, src2=mask, alpha=0.5, beta=0.5, gamma=1)
    # image_alpha = cv2.add(image, mask)
    return image_alpha


def get_iou(image_mask, predict):

    height = predict.shape[0]
    weight = predict.shape[1]
    # print(height, weight)
    o = 0
    for row in range(height):
            for col in range(weight):
                if predict[row, col] < 125:
                    predict[row, col] = 0
                else:
                    predict[row, col] = 1
                if predict[row, col] == 0 or predict[row, col] == 1:
                    o += 1

    height_mask = image_mask.shape[0]
    weight_mask = image_mask.shape[1]
    for row in range(height_mask):
            for col in range(weight_mask):
                if image_mask[row, col] < 125:
                    image_mask[row, col] = 0
                else:
                    image_mask[row, col] = 1
                if image_mask[row, col] == 0 or image_mask[row, col] == 1:
                    o += 1
    predict = predict.astype(np.int16)
    interArea = np.multiply(predict, image_mask)
    tem = predict + image_mask
    unionArea = tem - interArea
    inter = np.sum(interArea)
    union = np.sum(unionArea)
    iou_tem = inter / union

    print('iou=%f' % (iou_tem))
    return iou_tem


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="VAMP EVAL")
    parse.add_argument("--src_dir", type=str, default="./AutoPortraitMatting/testing/images", help="hr gt image folder")
    parse.add_argument("--gt_dir", type=str, default="./AutoPortraitMatting/testing/masks", help="hr gt image folder")
    parse.add_argument("--input_npz_path", type=str, default="npz_datalist.txt", help="npz text file")
    parse.add_argument("--out_npz_dir", type=str, default="./outputs", help="vamp output folder")
    parse.add_argument("--input_shape", nargs='+', type=int, default=[128, 128], help="vamp input shape h,w ")
    parse.add_argument("--draw_dir", type=str, default="./npz_draw_result", help="vamp npz draw image reult folder")
    parse.add_argument("--vamp_flag", action='store_true', default=True, help="decode vamp or vamc result")

    args = parse.parse_args()
    print(args)

    os.makedirs(args.draw_dir, exist_ok=True)

    output_npz_list = glob.glob(os.path.join(args.out_npz_dir, "*.npz"))
    output_npz_list.sort()
    
    ious = []
    with open(args.input_npz_path, 'r') as f:
        file_lines = f.readlines()
        for i, line in enumerate(tqdm(file_lines)):
            npz_name = os.path.basename(line.strip('\n'))
            file_name = npz_name.replace(".npz", ".png")

            # src image
            src_image  = cv2.imread(os.path.join(args.src_dir, file_name))

            # load npy
            if args.vamp_flag:
                npz_file = output_npz_list[i]
            else:
                npz_file = os.path.join(args.out_npz_dir, npz_name)

            heatmap = np.load(npz_file, allow_pickle=True)["output_0"].astype(np.float32)

            # draw
            vacc_preds = torch.nn.functional.interpolate(torch.from_numpy(heatmap), size=(src_image.shape[0], src_image.shape[1]), mode='bilinear', align_corners=True)
            vacc_preds = torch.sigmoid(vacc_preds)

            vacc_preds = vacc_preds.squeeze(0).squeeze().numpy()
            vacc_preds = vacc_preds > 0.5
            vacc_preds = vacc_preds * 255
            image_alpha = draw_matting(src_image, vacc_preds)
            cv2.imwrite(os.path.join(args.draw_dir, os.path.basename(file_name)), image_alpha)
            
            # eval
            label_path = os.path.join(args.gt_dir, os.path.basename(file_name).replace(".png", "_matte.png"))
            label = cv2.imread(label_path, 0)
            label = cv2.resize(label, args.input_shape) # , interpolation=cv2.INTER_LINEAR
            preds = torch.sigmoid(torch.from_numpy(heatmap))

            preds = preds.squeeze(0).squeeze().numpy()
            preds = preds > 0.5
            preds = preds * 255
            iou = get_iou(label, preds)
            print('{}, --> iou: {}'.format(os.path.basename(file_name), str(iou)))
            ious.append(iou)

        mean_iou = np.mean(ious)
        print("mean iou: {}".format(mean_iou))


""" 
unet3p-int8-kl_divergence-1_3_128_128-vacc
mean iou: 0.7632711768851099

unet3p_deepsupervision-int8-kl_divergence-1_3_128_128-vacc
mean iou: 0.6824934317812644

"""
