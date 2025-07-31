
# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


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


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="VAMP EVAL")
    parse.add_argument("--src_dir", type=str, default="./UnetPlusPlus/dsb2018_256_val/images", help="hr gt image folder")
    parse.add_argument("--gt_dir", type=str, default="./UnetPlusPlus/dsb2018_256_val/masks", help="hr gt image folder")
    parse.add_argument("--input_npz_path", type=str, default="npz_datalist.txt", help="npz text file")
    parse.add_argument("--out_npz_dir", type=str, default="./outputs/unetpp", help="vamp output folder")
    parse.add_argument("--input_shape", nargs='+', type=int, default=[96, 96], help="vamp input shape")
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
    from source_code.unet_zoo.metrics import get_iou, get_dice

    iou_list = []
    dic_list = []

    with open(args.input_npz_path, 'r') as f:
        file_lines = f.readlines()
        for i, line in enumerate(tqdm(file_lines)):
            npz_name = os.path.basename(line.strip('\n'))
            file_name = npz_name.replace(".npz", ".png")

            # src image
            ori_image  = Image.open(os.path.join(args.src_dir, file_name))

            # load npy
            if args.vamp_flag:
                npz_file = output_npz_list[i]
            else:
                npz_file = os.path.join(args.out_npz_dir, npz_name)

            heatmap = np.load(npz_file, allow_pickle=True)["output_0"].astype(np.float32)
            tvm_output = heatmap[0][0]

            # draw
            predict_mask = tvm_output * 255
            cv2.imwrite(os.path.join(args.draw_dir, file_name), predict_mask)

            ########################################################################################################
            # eval
            label_path = os.path.join(args.gt_dir, file_name)
            if not os.path.exists(label_path):
                continue

            iou = get_iou(label_path, tvm_output)
            dic = get_dice(label_path, tvm_output)

            iou_list.append(iou)
            dic_list.append(dic)

            print("{:s}, validation DIC: {:.3f}, IoU: {:.3f}".format(file_name, dic * 100, iou * 100))

        print("{:s}, mean DIC: {:.3f}, mean IoU: {:.3f}".format(file_name, np.mean(dic_list) * 100, np.mean(iou_list) * 100))
            ########################################################################################################

""" 
unetpp-fp16-none-3_96_96-vacc
mean DIC: 90.183, mean IoU: 83.174
unetpp-int8-kl_divergence-3_96_96-vacc
mean DIC: 90.120, mean IoU: 83.084
"""
