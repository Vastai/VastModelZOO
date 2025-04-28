
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


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="VAMP EVAL")
    parse.add_argument("--src_dir", type=str, default="./carvana/imgs", help="hr gt image folder")
    parse.add_argument("--gt_dir", type=str, default="./carvana/masks", help="hr gt image folder")
    parse.add_argument("--input_npz_path", type=str, default="npz_datalist.txt", help="npz text file")
    parse.add_argument("--out_npz_dir", type=str, default="./code/vamc/vamp/0.2.0/outputs/unet", help="vamp output folder")
    parse.add_argument("--input_shape", nargs='+', type=int, default=[512, 512], help="vamp input shape")
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
    from source_code.milesial.stream_metrics import StreamSegMetrics

    metrics = StreamSegMetrics(2)

    with open(args.input_npz_path, 'r') as f:
        file_lines = f.readlines()
        for i, line in enumerate(tqdm(file_lines)):
            npz_name = os.path.basename(line.strip('\n'))

            # src image
            ori_image  = Image.open(os.path.join(args.src_dir, npz_name.replace(".npz", ".jpg")))

            # load npy
            if args.vamp_flag:
                npz_file = output_npz_list[i]
            else:
                npz_file = os.path.join(args.out_npz_dir, npz_name)

            heatmap = np.load(npz_file, allow_pickle=True)["output_0"].astype(np.float32)
            tvm_predict = torch.from_numpy(heatmap)

            # # draw
            predict = tvm_predict.max(dim=1)[1].cpu().numpy()

            mask = predict * 255
            mask = mask.transpose(1, 2, 0).astype("uint8")
            cv2.imwrite(os.path.join(args.draw_dir, npz_name.replace(".npz", "_vis.png")), mask)
            
            # ########################################################################################################
            # # eval
            # gt 
            label_path = os.path.join(args.gt_dir, npz_name.replace(".npz", "_mask.gif"))
            if not os.path.exists(label_path):
                continue
            gt = Image.open(label_path)
            gt = gt.resize(size = args.input_shape)
            gt = np.expand_dims(gt, 0)

            metrics.update(gt, predict)
            val_score = metrics.get_results()
            print(metrics.to_str(val_score))

            # ########################################################################################################

"""
unet_carvana_scale0.5-int8-kl_divergence-3_512_512-vacc
Overall Acc: 0.994154
Mean Acc: 0.993711
FreqW Acc: 0.988437
Mean IoU: 0.982713

"""
