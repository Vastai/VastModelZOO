
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

def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="VAMP EVAL")
    parse.add_argument("--src_dir", type=str, default="./dataset/seg/VOCdevkit/VOC2012/JPEGImages_val", help="hr gt image folder")
    parse.add_argument("--gt_dir", type=str, default="./dataset/seg/VOCdevkit/VOC2012/SegmentationClass", help="hr gt image folder")
    parse.add_argument("--input_npz_path", type=str, default="npz_datalist.txt", help="npz text file")
    parse.add_argument("--out_npz_dir", type=str, default="./code/vamc/vamp/0.2.0/outputs/deeplab_v3_mobilenet", help="vamp output folder")
    parse.add_argument("--input_shape", nargs='+', type=int, default=[513, 513], help="vamp input shape")
    parse.add_argument("--draw_dir", type=str, default="./npz_draw_result", help="vamp npz draw image reult folder")
    parse.add_argument("--vamp_flag", action='store_true', default=True, help="decode vamp or vamc result")

    args = parse.parse_args()
    print(args)

    os.makedirs(args.draw_dir, exist_ok=True)

    output_npz_list = glob.glob(os.path.join(args.out_npz_dir, "*.npz"))
    output_npz_list.sort()
    
    import sys
    _cur_file_path = os.path.split(os.path.realpath(__file__))[0]
    sys.path.append(_cur_file_path + os.sep + '../..')
    from source_code.vainf.score import SegmentationMetric

    metric = SegmentationMetric(21)

    with open(args.input_npz_path, 'r') as f:
        file_lines = f.readlines()
        for i, line in enumerate(tqdm(file_lines)):
            npz_name = os.path.basename(line.strip('\n'))
            file_name = npz_name.replace(".npz", ".png")

            # src image
            ori_image  = Image.open(os.path.join(args.src_dir, file_name.replace(".png", ".jpg")))

            # load npy
            if args.vamp_flag:
                npz_file = output_npz_list[i]
            else:
                npz_file = os.path.join(args.out_npz_dir, npz_name)

            heatmap = np.load(npz_file, allow_pickle=True)["output_0"].astype(np.float32)
            # draw
            tvm_predict = torch.from_numpy(heatmap)
            predict = tvm_predict[0].cpu().numpy()
            predict = np.asarray(predict, dtype="float32").transpose(1, 2, 0)

            predict_mask = cv2.resize(predict, ori_image.size, interpolation=cv2.INTER_CUBIC)

            colors = np.loadtxt("segmentation/deeplab_v3/source_code/vainf/voc2012_colors.txt").astype('uint8')
            color = colorize(predict_mask.argmax(axis=2).astype(np.uint8), colors)
            color.save(os.path.join(args.draw_dir, file_name))

            ########################################################################################################
            # eval
            label_path = os.path.join(args.gt_dir, file_name)
            if not os.path.exists(label_path):
                continue
            gt = Image.open(label_path)
            gt = gt.resize(size=args.input_shape)
            target = np.array(gt).astype('int32')
            target[target == 255] = -1
            target = torch.from_numpy(target).long()

            metric.update(tvm_predict, target)
            pixAcc, mIoU = metric.get()
            print("{:s}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(file_name, pixAcc * 100, mIoU * 100))
            ########################################################################################################

""" 
deeplab_v3_mobilenet-fp16-none-3_513_513-vacc
validation pixAcc: 92.796, mIoU: 69.636
deeplab_v3_mobilenet-int8-kl_divergence-3_513_513-vacc
validation pixAcc: 91.932, mIoU: 67.229



deeplab_v3_plus_mobilenet-fp16-none-3_513_513-vacc
validation pixAcc: 71.853, mIoU: 3.567
deeplab_v3_plus_mobilenet-int8-kl_divergence-3_513_513-vacc
validation pixAcc: 73.147, mIoU: 3.577

deeplab_v3_plus_resnet50-fp16-none-3_513_513-vacc
validation pixAcc: 94.031, mIoU: 74.192
deeplab_v3_plus_resnet50-int8-kl_divergence-3_513_513-vacc
validation pixAcc: 93.704, mIoU: 73.183

"""
